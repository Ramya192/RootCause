"""Per-client rate limit on starting runs (causal_engine/api/ratelimit.py and its use in app.py).

The limiter is tested with a hand-driven clock, and the HTTP behaviour with the fake runner used by
test_api.py, so nothing here runs the pipeline or waits in real time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from causal_engine.api.app import create_app
from causal_engine.api.ratelimit import RateLimiter, client_key
from causal_engine.models.schemas import CausalGraph, EffectEstimate, Explanation, PipelineResult


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_client_is_allowed_up_to_the_limit_then_told_how_long_to_wait():
    clock = Clock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)

    for _ in range(2):
        assert limiter.retry_after("a") is None
        limiter.record("a")
    clock.now += 20

    assert limiter.retry_after("a") == 40  # the oldest run leaves the window 60 s after it started


def test_the_allowance_returns_as_old_runs_leave_the_window():
    clock = Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    limiter.record("a")
    assert limiter.retry_after("a") is not None

    clock.now += 60

    assert limiter.retry_after("a") is None


def test_clients_are_limited_independently():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=Clock())
    limiter.record("a")

    assert limiter.retry_after("a") is not None
    assert limiter.retry_after("b") is None


def test_checking_does_not_use_up_the_allowance():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=Clock())

    for _ in range(5):
        assert limiter.retry_after("a") is None


def test_a_limit_of_zero_turns_the_limiter_off():
    limiter = RateLimiter(limit=0, window_seconds=60, clock=Clock())
    for _ in range(100):
        limiter.record("a")

    assert limiter.enabled is False
    assert limiter.retry_after("a") is None


def test_a_non_positive_window_is_rejected():
    with pytest.raises(ValueError):
        RateLimiter(limit=1, window_seconds=0)


def test_idle_clients_are_forgotten_so_memory_does_not_grow():
    clock = Clock()
    limiter = RateLimiter(limit=5, window_seconds=60, clock=clock)
    for i in range(50):
        limiter.record(f"client-{i}")
    clock.now += 61
    limiter.record("someone-new")

    assert len(limiter._hits) == 1


class _Req:
    def __init__(self, forwarded=None, host="10.0.0.9"):
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}
        self.client = type("C", (), {"host": host})() if host else None


def test_the_client_key_is_the_address_the_platform_appended_not_one_the_caller_supplied():
    # Cloud Run appends the real address after whatever the caller sent, so a spoofed first entry must not matter.
    assert client_key(_Req("6.6.6.6, 203.0.113.7")) == "203.0.113.7"
    assert client_key(_Req("203.0.113.7")) == "203.0.113.7"


def test_the_client_key_can_skip_extra_trusted_proxies():
    assert client_key(_Req("203.0.113.7, 130.211.0.1"), trusted_proxies=2) == "203.0.113.7"
    assert client_key(_Req("203.0.113.7"), trusted_proxies=5) == "203.0.113.7"  # never indexes off the start


def test_without_a_forwarded_header_the_socket_address_is_used():
    assert client_key(_Req(None, host="192.168.1.5")) == "192.168.1.5"
    assert client_key(_Req(None, host=None)) == "unknown"
    assert client_key(_Req(" , ")) == "10.0.0.9"


def _fake_result(domain_id: str) -> PipelineResult:
    return PipelineResult(
        domain_id=domain_id,
        causal_graph=CausalGraph(nodes=["a", "b"], edges=[("a", "b")], algorithm="fake"),
        effect_estimates=[EffectEstimate(treatment="a", outcome="b", ate=0.1, estimator="fake")],
        counterfactuals=[],
        recommendations=[],
        explanation=Explanation(narrative="fake"),
    )


@pytest.fixture
def limited_client(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "100")  # keep the queue cap out of the way
    clock = Clock()
    app = create_app(
        config_loader=config_loader,
        runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)},
        rate_limiter=RateLimiter(limit=2, window_seconds=60, clock=clock),
    )
    with TestClient(app) as client:
        yield client, clock


def _submit(client, ip: str | None = None):
    headers = {"x-forwarded-for": ip} if ip else {}
    return client.post("/domains/employee_attrition/analyze", headers=headers)


def test_over_the_limit_the_api_answers_429_with_retry_after_and_a_readable_message(limited_client):
    client, clock = limited_client
    assert _submit(client, "203.0.113.7").status_code == 202
    assert _submit(client, "203.0.113.7").status_code == 202

    over = _submit(client, "203.0.113.7")

    assert over.status_code == 429
    assert int(over.headers["retry-after"]) == 60
    assert "saved example" in over.json()["detail"]

    clock.now += 61
    assert _submit(client, "203.0.113.7").status_code == 202


def test_one_clients_limit_does_not_block_another(limited_client):
    client, _ = limited_client
    for _ in range(2):
        _submit(client, "203.0.113.7")

    assert _submit(client, "203.0.113.7").status_code == 429
    assert _submit(client, "198.51.100.2").status_code == 202


def test_rejected_requests_do_not_use_up_the_allowance(limited_client):
    client, clock = limited_client
    # unknown domain and bad dataset are refused before a run is queued, so they must not count
    for _ in range(5):
        assert client.post("/domains/nope/analyze").status_code == 404
        assert client.post("/domains/employee_attrition/analyze", json={"dataset": "nope"}).status_code == 400

    assert _submit(client).status_code == 202
    assert _submit(client).status_code == 202


def test_a_run_refused_because_the_queue_is_full_does_not_count_against_the_client(config_loader, monkeypatch):
    import threading

    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "1")
    gate = threading.Event()

    def blocking(domain_id, config, path):
        gate.wait(10)
        return _fake_result(domain_id)

    limiter = RateLimiter(limit=3, window_seconds=60, clock=Clock())
    app = create_app(config_loader=config_loader, runners={"direct": blocking, "crew": blocking}, rate_limiter=limiter)
    try:
        with TestClient(app) as client:
            statuses = [_submit(client).status_code for _ in range(4)]
            assert 429 in statuses
            queued = statuses.count(202)
            assert len(limiter._hits["testclient"]) == queued  # only queued runs were counted
            gate.set()
    finally:
        gate.set()


def test_the_default_limit_comes_from_the_environment(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "100")
    monkeypatch.setenv("ROOTCAUSE_RATE_LIMIT", "1")
    monkeypatch.setenv("ROOTCAUSE_RATE_WINDOW_SECONDS", "600")
    app = create_app(config_loader=config_loader, runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)})

    with TestClient(app) as client:
        assert _submit(client).status_code == 202
        over = _submit(client)

    assert over.status_code == 429
    assert int(over.headers["retry-after"]) <= 600


def test_a_limit_of_zero_in_the_environment_disables_it(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "100")
    monkeypatch.setenv("ROOTCAUSE_RATE_LIMIT", "0")
    app = create_app(config_loader=config_loader, runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)})

    with TestClient(app) as client:
        assert all(_submit(client).status_code == 202 for _ in range(10))


# --- agent-crew runs: a tighter limit of their own ---


@pytest.fixture
def crew_client(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "100")
    monkeypatch.setenv("ROOTCAUSE_ALLOW_CREW", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")
    clock = Clock()
    app = create_app(
        config_loader=config_loader,
        runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)},
        rate_limiter=RateLimiter(limit=10, window_seconds=60, clock=clock),
        crew_rate_limiter=RateLimiter(limit=1, window_seconds=3600, clock=clock),
    )
    with TestClient(app) as client:
        yield client, clock


def _submit_crew(client, ip="203.0.113.7"):
    return client.post("/domains/employee_attrition/analyze", json={"orchestration": "crew"}, headers={"x-forwarded-for": ip})


def test_crew_runs_have_their_own_tighter_limit_and_say_so(crew_client):
    client, clock = crew_client
    assert _submit_crew(client).status_code == 202

    over = _submit_crew(client)

    assert over.status_code == 429
    assert "Agent-crew runs are limited" in over.json()["detail"]
    assert int(over.headers["retry-after"]) == 3600
    clock.now += 3601
    assert _submit_crew(client).status_code == 202


def test_a_crew_limit_does_not_block_direct_runs_or_other_clients(crew_client):
    client, _ = crew_client
    _submit_crew(client)

    assert _submit_crew(client).status_code == 429
    assert _submit(client, "203.0.113.7").status_code == 202  # direct runs still allowed for the same client
    assert _submit_crew(client, "198.51.100.2").status_code == 202


def test_direct_runs_do_not_use_up_the_crew_allowance(crew_client):
    client, _ = crew_client
    for _ in range(3):
        assert _submit(client, "203.0.113.7").status_code == 202

    assert _submit_crew(client).status_code == 202


def test_a_refused_crew_request_does_not_use_up_the_crew_allowance(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "100")
    monkeypatch.delenv("ROOTCAUSE_ALLOW_CREW", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")
    crew_limiter = RateLimiter(limit=1, window_seconds=3600, clock=Clock())
    app = create_app(
        config_loader=config_loader,
        runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)},
        crew_rate_limiter=crew_limiter,
    )
    with TestClient(app) as client:
        assert _submit_crew(client).status_code == 403  # crew switched off on this server
        monkeypatch.setenv("ROOTCAUSE_ALLOW_CREW", "1")
        assert _submit_crew(client).status_code == 202


def test_capabilities_tell_the_page_what_to_offer_without_exposing_the_key(config_loader, monkeypatch):
    app = create_app(config_loader=config_loader, runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)})
    with TestClient(app) as client:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ROOTCAUSE_ALLOW_CREW", raising=False)
        assert client.get("/capabilities").json() == {"crew": False, "llm_narrative": False}

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")
        assert client.get("/capabilities").json() == {"crew": False, "llm_narrative": True}  # a key alone does not enable crew

        monkeypatch.setenv("ROOTCAUSE_ALLOW_CREW", "1")
        body = client.get("/capabilities")
        assert body.json() == {"crew": True, "llm_narrative": True}
        assert "sk-test" not in body.text


def test_the_crew_limit_defaults_to_two_per_hour(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "100")
    monkeypatch.setenv("ROOTCAUSE_RATE_LIMIT", "0")  # general limit off, so only the crew limit can answer 429
    monkeypatch.setenv("ROOTCAUSE_ALLOW_CREW", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")
    monkeypatch.delenv("ROOTCAUSE_CREW_RATE_LIMIT", raising=False)
    app = create_app(config_loader=config_loader, runners={"direct": lambda d, c, p: _fake_result(d), "crew": lambda d, c, p: _fake_result(d)})

    with TestClient(app) as client:
        assert [_submit_crew(client).status_code for _ in range(3)] == [202, 202, 429]
