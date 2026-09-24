"""FastAPI service exposing the RootCause pipeline.

    GET  /health                       liveness
    GET  /domains                      configured domains + whether they can run
    POST /domains/{id}/analyze         queue a run -> 202 + job
    GET  /jobs                         recent jobs (no results)
    GET  /jobs/{job_id}                job status; carries the PipelineResult once done

`orchestration` picks how the 7 stages run: "direct" (default; the stage
functions called in order -- fast) or "crew" (the CDIA spec's hierarchical
CrewAI orchestration -- minutes of paid LLM calls, needs OPENAI_API_KEY).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from rootcause.api.jobs import Job, JobStore
from rootcause.models.schemas import PipelineResult
from rootcause.pipeline import runner
from rootcause.utils.config_loader import ConfigLoader

Runner = Callable[[str, dict, str], PipelineResult]

DEFAULT_RUNNERS: dict[str, Runner] = {
    "direct": runner.run_direct,
    "crew": runner.run_crew,
}


class AnalyzeRequest(BaseModel):
    orchestration: Literal["direct", "crew"] = "direct"


class DomainSummary(BaseModel):
    id: str
    name: str
    description: str
    status: str
    runnable: bool


def create_app(
    config_loader: Optional[ConfigLoader] = None,
    runners: Optional[dict[str, Runner]] = None,
) -> FastAPI:
    loader = config_loader or ConfigLoader()
    run_fns = runners or DEFAULT_RUNNERS
    jobs = JobStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="RootCause", description="Causal Decision Intelligence Agent", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/domains", response_model=list[DomainSummary])
    def list_domains() -> list[DomainSummary]:
        return [
            DomainSummary(
                id=d.id,
                name=d.name,
                description=d.description,
                status=d.status,
                runnable=d.is_runnable,
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

        run_fn = run_fns[body.orchestration]
        data_path = str(runner.resolve_data_path(domain.extra))
        job = jobs.submit(
            domain_id,
            body.orchestration,
            lambda: run_fn(domain_id, domain.extra, data_path),
        )
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

    return app
