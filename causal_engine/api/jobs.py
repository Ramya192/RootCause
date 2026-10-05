"""In-memory job store for analysis runs.

A full run takes seconds (direct) to several minutes (crew), so the API
queues it and lets clients poll. A single worker thread serializes runs:
Stage 2 (Feast) writes a parquet file + SQLite online store per (domain,
dataset), so two concurrent runs of the same pair would corrupt each other's
feature data. Jobs live only for
the process lifetime and the oldest finished ones are evicted past
`max_jobs` -- fine for a V1 service, not a durable queue.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel

from causal_engine.models.schemas import PipelineResult

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Job(BaseModel):
    id: str
    domain_id: str
    orchestration: str
    dataset: Optional[str] = None
    status: JobStatus = JobStatus.queued
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[PipelineResult] = None
    error: Optional[str] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStore:
    def __init__(self, max_jobs: int = 100):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rootcause-job")
        self._max_jobs = max_jobs

    def submit(
        self,
        domain_id: str,
        orchestration: str,
        fn: Callable[[], PipelineResult],
        dataset: Optional[str] = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            domain_id=domain_id,
            orchestration=orchestration,
            dataset=dataset,
            created_at=_now(),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
            snapshot = job.model_copy()  # taken before the worker can touch the job
        self._executor.submit(self._run, job.id, fn)
        return snapshot

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def list(self) -> list[Job]:
        """Newest first, without the (large) result payloads."""
        with self._lock:
            jobs = [j.model_copy(update={"result": None}) for j in self._jobs.values()]
        return list(reversed(jobs))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, fn: Callable[[], PipelineResult]) -> None:
        self._update(job_id, status=JobStatus.running, started_at=_now())
        try:
            result = fn()
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._update(
                job_id,
                status=JobStatus.failed,
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            self._update(job_id, status=JobStatus.succeeded, finished_at=_now(), result=result)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:  # may have been evicted while running
                for name, value in fields.items():
                    setattr(job, name, value)

    def _evict_locked(self) -> None:
        overflow = len(self._jobs) - self._max_jobs
        if overflow <= 0:
            return
        finished = [
            jid
            for jid, j in self._jobs.items()
            if j.status in (JobStatus.succeeded, JobStatus.failed)
        ]
        for jid in finished[:overflow]:
            del self._jobs[jid]
