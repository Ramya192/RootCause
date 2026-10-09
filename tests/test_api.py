"""FastAPI service (causal_engine/api/).

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

from causal_engine.api.app import create_app
from causal_engine.models.schemas import (
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


def test_domains_carry_the_facts_the_ui_needs(fake_client):
    domain = {d["id"]: d for d in fake_client.get("/domains").json()}["employee_attrition"]
    assert domain["outcome"] == "attrition"
    assert {"compensation", "manager_quality", "workload"} <= set(domain["treatments"])
    assert domain["sensitive_attribute"] == "gender"
    assert domain["observational"] is False


def test_domains_hide_datasets_whose_file_is_absent_on_this_host(fake_client, monkeypatch, tmp_path):
    from causal_engine.pipeline import runner

    real = runner.resolve_data_path
    monkeypatch.setattr(
        runner, "resolve_data_path",
        lambda cfg, name=None: tmp_path / "missing.csv" if cfg["domain"]["id"] == "freddie_mac" else real(cfg, name),
    )
    domains = {d["id"]: d for d in fake_client.get("/domains").json()}
    assert domains["freddie_mac"]["datasets"] == [] and domains["freddie_mac"]["runnable"] is False
    assert domains["employee_attrition"]["runnable"] is True


def test_bundled_ui_samples_are_valid_results(fake_client):
    from causal_engine.models.schemas import PipelineResult

    index = fake_client.get("/examples/index.json").json()
    assert index, "run scripts/reports/build_ui_examples.py"
    for meta in index:
        sample = fake_client.get(f"/examples/{meta['domain_id']}__{meta['dataset']}.json").json()
        result = PipelineResult.model_validate(sample["result"])  # same schema the API returns
        graph = result.causal_graph
        assert all(a in graph.nodes and b in graph.nodes for a, b in graph.edges)
        assert sample["meta"]["outcome"] in graph.nodes
        assert {e.treatment for e in result.effect_estimates} <= set(sample["meta"]["treatments"])
        if "truth" in sample["meta"]:  # known-answer sample: the overlay needs the true edges
            assert sample["meta"]["truth"]["edges"]


def test_root_serves_the_browser_ui(fake_client):
    response = fake_client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>RootCause" in response.text and "Ramya A" in response.text
    assert "/domains" in response.text and "/analyze" in response.text


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


def test_crew_is_refused_unless_the_server_switches_it_on(fake_client, monkeypatch):
    monkeypatch.delenv("ROOTCAUSE_ALLOW_CREW", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")  # a key alone must not be enough
    resp = fake_client.post("/domains/employee_attrition/analyze", json={"orchestration": "crew"})
    assert resp.status_code == 403
    assert "ROOTCAUSE_ALLOW_CREW" in resp.json()["detail"]
    assert fake_client.get("/jobs").json() == []  # nothing was queued


def test_crew_without_api_key_is_rejected_up_front(fake_client, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_ALLOW_CREW", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = fake_client.post("/domains/employee_attrition/analyze", json={"orchestration": "crew"})
    assert resp.status_code == 400
    assert "OPENAI_API_KEY" in resp.json()["detail"]


def test_crew_with_api_key_queues(fake_client, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_ALLOW_CREW", "1")
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


def test_a_full_queue_answers_429_with_a_readable_message(config_loader, monkeypatch):
    monkeypatch.setenv("ROOTCAUSE_MAX_PENDING", "1")
    release = threading.Event()

    def blocked(*_):
        release.wait(timeout=10)
        return _fake_result()

    client = TestClient(create_app(config_loader=config_loader, runners={"direct": blocked}))
    first = client.post("/domains/employee_attrition/analyze")
    assert first.status_code == 202
    second = client.post("/domains/employee_attrition/analyze")
    assert second.status_code == 429
    assert "busy" in second.json()["detail"] and second.headers["Retry-After"] == "60"
    release.set()
    assert _wait_for(client, first.json()["id"])["status"] == "succeeded"
    assert client.post("/domains/employee_attrition/analyze").status_code == 202  # room again once it finished


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


def test_list_domains_reports_available_datasets(fake_client):
    domain = {d["id"]: d for d in fake_client.get("/domains").json()}["employee_attrition"]
    assert domain["datasets"] == ["synthetic"]
    assert domain["default_dataset"] == "synthetic"


def test_omitted_dataset_uses_default_and_runner_gets_it(config_loader):
    seen = {}

    def spy(domain_id, config, data_path):
        seen.update(config=config, data_path=data_path)
        return _fake_result()

    client = TestClient(create_app(config_loader=config_loader, runners={"direct": spy}))
    job = client.post("/domains/employee_attrition/analyze").json()
    assert job["dataset"] == "synthetic"
    assert _wait_for(client, job["id"])["status"] == "succeeded"
    # the run config is tagged so Stage 2 can keep this dataset's Feast state apart
    assert seen["config"]["run"] == {"dataset": "synthetic"}
    assert seen["data_path"].replace("\\", "/").endswith("data/employee_attrition/attrition.csv")
    # ...without mutating the shared domain config
    assert "run" not in config_loader.get_domain("employee_attrition").extra


def test_dataset_not_configured_for_domain_is_400(fake_client):
    resp = fake_client.post("/domains/employee_attrition/analyze", json={"dataset": "real"})
    assert resp.status_code == 400
    assert "synthetic" in resp.json()["detail"]  # tells the caller what IS available


def test_configured_dataset_with_missing_file_is_409():
    from causal_engine.utils.config_loader import ConfigLoader

    loader = ConfigLoader()  # fresh: don't mutate the session-scoped loader
    ingestion = loader.get_domain("employee_attrition").extra["ingestion"]
    ingestion["datasets"]["real"] = "data/employee_attrition/not_downloaded_yet.csv"
    client = TestClient(create_app(config_loader=loader, runners={"direct": lambda *a: _fake_result()}))

    resp = client.post("/domains/employee_attrition/analyze", json={"dataset": "real"})
    assert resp.status_code == 409
    assert "not_downloaded_yet.csv" in resp.json()["detail"]


def test_selected_dataset_path_is_the_one_run():
    from causal_engine.utils.config_loader import ConfigLoader

    loader = ConfigLoader()
    ingestion = loader.get_domain("employee_attrition").extra["ingestion"]
    ingestion["datasets"]["semi_synthetic"] = "data/employee_attrition/ground_truth.json"  # any existing file
    seen = {}

    def spy(domain_id, config, data_path):
        seen.update(config=config, data_path=data_path)
        return _fake_result()

    client = TestClient(create_app(config_loader=loader, runners={"direct": spy}))
    job = client.post("/domains/employee_attrition/analyze", json={"dataset": "semi_synthetic"}).json()
    _wait_for(client, job["id"])
    assert job["dataset"] == "semi_synthetic"
    assert seen["config"]["run"]["dataset"] == "semi_synthetic"
    assert seen["data_path"].replace("\\", "/").endswith("ground_truth.json")


def test_the_page_offers_the_agent_crew_only_when_the_server_reports_it(fake_client):
    page = fake_client.get("/").text

    assert 'id="crewOpt" hidden' in page  # hidden until /capabilities says the server allows it
    # `label { display:flex }` would otherwise override the hidden attribute and show the box anyway
    assert "[hidden] { display:none !important; }" in page
    assert "/capabilities" in page
    assert 'orchestration = "direct"' in page  # the default stays the free direct run


def test_the_page_resets_the_crew_box_and_tells_the_truth_about_the_narrative_tier(fake_client):
    page = fake_client.get("/").text

    assert '$("crew").checked = false' in page  # one crew run must not silently make the next one a crew run
    assert 'id="runNote"' in page and "c.llm_narrative" in page  # the note changes when the server has a key
    # the dashed legend entry is only drawn when a node carries that style ("young" is derived, not a node)
    assert "nodes.includes(meta.sensitive_attribute)" in page


def test_the_page_and_saved_examples_are_revalidated_so_a_redeploy_is_not_hidden_by_a_stale_cache(fake_client):
    assert fake_client.get("/").headers["cache-control"] == "no-cache"
    index = fake_client.get("/examples/index.json")
    assert index.status_code == 200
    assert index.headers["cache-control"] == "no-cache"


def test_a_model_provider_error_is_not_shown_to_visitors(config_loader, caplog):
    # The first local crew test with a fake key showed OpenAI's text, which quotes the key it was given.
    from openai import AuthenticationError

    def rejected(*_):
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise AuthenticationError("Incorrect API key provided: sk-proj-abc123SECRET", response=response, body=None)

    client = TestClient(create_app(config_loader=config_loader, runners={"direct": rejected}))
    job_id = client.post("/domains/employee_attrition/analyze").json()["id"]

    done = _wait_for(client, job_id)

    assert done["status"] == "failed"
    assert "SECRET" not in done["error"] and "platform.openai.com" not in done["error"]
    assert done["error"].startswith("AuthenticationError:") and "server log" in done["error"]
    assert "sk-proj-abc123SECRET" in caplog.text  # the detail is still logged for the operator


def test_a_key_that_appears_in_one_of_our_own_errors_is_removed():
    from causal_engine.api.jobs import public_error

    text = public_error(RuntimeError("could not use sk-live_ABC.def-123 for stage 7"))

    assert "ABC" not in text and "[key removed]" in text and "stage 7" in text
