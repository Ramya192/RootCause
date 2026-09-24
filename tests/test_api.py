"""FastAPI service (rootcause/api/).

Most tests inject a fake runner so they exercise only the HTTP/job contract
and stay instant. `test_real_direct_run_through_api` is the exception: it
runs the actual 7-stage pipeline through the API (no mocks, like the rest of
this suite), with OPENAI_API_KEY removed so it stays free. The crew path is
never kicked off here -- only its up-front validation is tested -- per the
cost note in tests/test_crew.py.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from rootcause.api.app import create_app
from rootcause.models.schemas import (
    CausalGraph,
    EffectEstimate,
    Explanation,
    PipelineResult,
)


def _fake_result(domain_id: str = "employee_attrition") -> PipelineResult:
    return PipelineResult(
        domain_id=domain_id,
        causal_graph=CausalGraph(nodes=["a", "b"], edges=[("a", "b")], algorithm="fake"),
        effect_estimates=[EffectEstimate(treatment="a", outcome="b", ate=0.1, estimator="fake")],
        counterfactuals=[],
        recommendations=[],
        explanation=Explanation(narrative="fake narrative"),
    )


def _wait_for(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture
def fake_client(config_loader):
    app = create_app(
        config_loader=config_loader,
        runners={"direct": lambda *a: _fake_result(), "crew": lambda *a: _fake_result()},
    )
    return TestClient(app)


def test_health(fake_client):
    assert fake_client.get("/health").json() == {"status": "ok"}


def test_list_domains(fake_client):
    domains = {d["id"]: d for d in fake_client.get("/domains").json()}
    assert domains["employee_attrition"]["runnable"] is True
    assert domains["employee_attrition"]["name"] == "Employee Attrition"


def test_analyze_queues_job_and_result_is_retrievable(fake_client):
    resp = fake_client.post("/domains/employee_attrition/analyze")
    assert resp.status_code == 202
    job = resp.json()
    assert resp.headers["location"] == f"/jobs/{job['id']}"
    assert job["orchestration"] == "direct"
    assert job["result"] is None

    done = _wait_for(fake_client, job["id"])
    assert done["status"] == "succeeded"
    assert done["error"] is None
    assert done["result"]["explanation"]["narrative"] == "fake narrative"
    assert done["started_at"] and done["finished_at"]


def test_failed_run_is_reported_not_raised(config_loader):
    def boom(*_):
        raise RuntimeError("stage 4 exploded")

    client = TestClient(create_app(config_loader=config_loader, runners={"direct": boom}))
    job_id = client.post("/domains/employee_attrition/analyze").json()["id"]

    done = _wait_for(client, job_id)
    assert done["status"] == "failed"
    assert done["result"] is None
    assert "RuntimeError: stage 4 exploded" in done["error"]


def test_unknown_domain_is_404(fake_client):
    assert fake_client.post("/domains/nope/analyze").status_code == 404


def test_unknown_job_is_404(fake_client):
    assert fake_client.get("/jobs/does-not-exist").status_code == 404


def test_non_runnable_domain_is_409(config_loader, monkeypatch):
    monkeypatch.setattr(config_loader.get_domain("employee_attrition"), "status", "coming_soon")
    client = TestClient(create_app(config_loader=config_loader, runners={"direct": lambda *a: _fake_result()}))
    assert client.post("/domains/employee_attrition/analyze").status_code == 409


def test_invalid_orchestration_is_422(fake_client):
    resp = fake_client.post("/domains/employee_attrition/analyze", json={"orchestration": "magic"})
    assert resp.status_code == 422


def test_crew_without_api_key_is_rejected_up_front(fake_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = fake_client.post("/domains/employee_attrition/analyze", json={"orchestration": "crew"})
    assert resp.status_code == 400
    assert "OPENAI_API_KEY" in resp.json()["detail"]


def test_crew_with_api_key_queues(fake_client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")
    resp = fake_client.post("/domains/employee_attrition/analyze", json={"orchestration": "crew"})
    assert resp.status_code == 202
    assert resp.json()["orchestration"] == "crew"


def test_runs_are_serialized(config_loader):
    """Feast writes shared files, so two jobs must never overlap."""
    active, max_active = 0, 0
    lock = threading.Lock()

    def slow(*_):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        return _fake_result()

    client = TestClient(create_app(config_loader=config_loader, runners={"direct": slow}))
    ids = [client.post("/domains/employee_attrition/analyze").json()["id"] for _ in range(3)]
    for job_id in ids:
        assert _wait_for(client, job_id)["status"] == "succeeded"
    assert max_active == 1


def test_list_jobs_omits_results(fake_client):
    job_id = fake_client.post("/domains/employee_attrition/analyze").json()["id"]
    _wait_for(fake_client, job_id)
    listed = fake_client.get("/jobs").json()
    assert [j["id"] for j in listed] == [job_id]
    assert listed[0]["result"] is None


def test_real_direct_run_through_api(config_loader, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # keep free: template narrative
    client = TestClient(create_app(config_loader=config_loader))

    job_id = client.post("/domains/employee_attrition/analyze").json()["id"]
    done = _wait_for(client, job_id)

    assert done["status"] == "succeeded", done["error"]
    result = done["result"]
    assert result["domain_id"] == "employee_attrition"
    assert len(result["causal_graph"]["edges"]) == 6
    assert len(result["effect_estimates"]) == 3
    assert len(result["recommendations"]) == 3  # regression guard: never 0
    assert result["explanation"]["narrative"]
