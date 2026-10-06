"""FastAPI service exposing the RootCause pipeline.

    GET  /health                       liveness
    GET  /domains                      configured domains + whether they can run
    POST /domains/{id}/analyze         queue a run -> 202 + job
    GET  /jobs                         recent jobs (no results)
    GET  /jobs/{job_id}                job status; carries the PipelineResult once done
    GET  /jobs/{job_id}/graph          the discovered causal graph as a PNG (?format=dot for Graphviz DOT)

`dataset` picks which of the domain's data sources to run (real, synthetic or
semi_synthetic -- same schema and config, different file); omitted, the
domain's default_dataset is used.

`orchestration` picks how the 7 stages run: "direct" (default; the stage
functions called in order -- fast) or "crew" (the CDIA spec's hierarchical
CrewAI orchestration -- minutes of paid LLM calls, needs OPENAI_API_KEY).
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from causal_engine.api.jobs import Job, JobStatus, JobStore, QueueFull
from causal_engine.models.schemas import PipelineResult
from causal_engine.pipeline import runner
from causal_engine.utils.config_loader import ConfigLoader
from causal_engine.utils.graph_plot import plot_causal_graph, to_dot

Runner = Callable[[str, dict, str], PipelineResult]

DEFAULT_RUNNERS: dict[str, Runner] = {
    "direct": runner.run_direct,
    "crew": runner.run_crew,
}


class AnalyzeRequest(BaseModel):
    orchestration: Literal["direct", "crew"] = "direct"
    # real | synthetic | semi_synthetic; omitted -> the domain's default_dataset
    dataset: Optional[str] = None


class DomainSummary(BaseModel):
    id: str
    name: str
    description: str
    status: str
    runnable: bool
    datasets: list[str]
    default_dataset: str
    # facts the browser UI needs to explain a result in plain language
    outcome: str
    treatments: list[str]
    sensitive_attribute: Optional[str] = None
    observational: bool = False
    entity_noun_plural: str = "records"


def create_app(
    config_loader: Optional[ConfigLoader] = None,
    runners: Optional[dict[str, Runner]] = None,
) -> FastAPI:
    loader = config_loader or ConfigLoader()
    run_fns = runners or DEFAULT_RUNNERS
    # Runs are serialized on one worker, so a public deployment caps how many may wait; without a cap a
    # handful of clicks (or a bot) leaves every other visitor queued behind them.
    jobs = JobStore(max_pending=int(os.environ.get("ROOTCAUSE_MAX_PENDING", "3")))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="RootCause", description="Causal Decision Intelligence Agent", lifespan=lifespan)

    app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).with_name("ui.html"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/domains", response_model=list[DomainSummary])
    def list_domains() -> list[DomainSummary]:
        # Only datasets whose file is on this host: a deployment without the gitignored Freddie Mac
        # downloads must not offer runs that can only fail.
        def present(d) -> list[str]:
            return [n for n in runner.available_datasets(d.extra) if runner.resolve_data_path(d.extra, n).is_file()]

        return [
            DomainSummary(
                id=d.id,
                name=d.name,
                description=d.description,
                status=d.status,
                runnable=d.is_runnable and bool(present(d)),
                datasets=present(d),
                default_dataset=(
                    d.extra["ingestion"]["default_dataset"]
                    if d.extra["ingestion"]["default_dataset"] in present(d)
                    else (present(d) or [d.extra["ingestion"]["default_dataset"]])[0]
                ),
                outcome=d.extra["effect_estimation"]["outcome"],
                treatments=[t["name"] for t in d.extra["effect_estimation"]["treatments"]],
                sensitive_attribute=d.extra["interventions"].get("sensitive_attribute"),
                observational=bool(d.extra.get("explanation", {}).get("observational", False)),
                entity_noun_plural=d.extra["domain"].get("entity_noun_plural", "records"),
            )
            for d in loader.list_domains()
        ]

    @app.post("/domains/{domain_id}/analyze", response_model=Job, status_code=202)
    def analyze(domain_id: str, response: Response, body: Optional[AnalyzeRequest] = None) -> Job:
        body = body or AnalyzeRequest()
        try:
            domain = loader.get_domain(domain_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=exc.args[0]) from None
        if not domain.is_runnable:
            raise HTTPException(
                status_code=409,
                detail=f"Domain '{domain_id}' has status '{domain.status}' and cannot be run yet",
            )
        if body.orchestration == "crew" and not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="orchestration='crew' needs OPENAI_API_KEY (the manager agent is an LLM)",
            )

        try:
            dataset = runner.resolve_dataset(domain.extra, body.dataset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        data_path = runner.resolve_data_path(domain.extra, dataset)
        if not data_path.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"Dataset '{dataset}' is configured but its file was not found at {data_path}",
            )

        run_fn = run_fns[body.orchestration]
        run_config = runner.with_dataset(domain.extra, dataset)
        try:
            job = jobs.submit(
                domain_id,
                body.orchestration,
                lambda: run_fn(domain_id, run_config, str(data_path)),
                dataset=dataset,
            )
        except QueueFull:
            raise HTTPException(
                status_code=429,
                detail="The server is busy with other runs. Try again in a minute, or open a saved example, which loads instantly.",
                headers={"Retry-After": "60"},
            ) from None
        response.headers["Location"] = f"/jobs/{job.id}"
        return job

    @app.get("/jobs", response_model=list[Job])
    def list_jobs() -> list[Job]:
        return jobs.list()

    @app.get("/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
        return job

    @app.get("/jobs/{job_id}/graph")
    def get_job_graph(job_id: str, format: Literal["png", "dot"] = "png") -> Response:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
        if job.status != JobStatus.succeeded or job.result is None:
            raise HTTPException(status_code=409, detail=f"Job '{job_id}' is {job.status.value}; it has no graph yet")
        domain_config = loader.get_domain(job.domain_id).extra
        if format == "dot":
            return Response(to_dot(job.result.causal_graph, domain_config), media_type="text/vnd.graphviz")
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_causal_graph(
                job.result.causal_graph, Path(tmp) / "graph.png", domain_config, title=f"{job.domain_id} ({job.dataset})"
            )
            return Response(path.read_bytes(), media_type="image/png")

    return app
