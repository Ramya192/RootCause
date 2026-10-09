"""Per-client rate limit for the endpoint that starts a run.

The job queue already caps how many runs may wait (JobStore.max_pending), but that cap is shared: one
visitor, or a bot, can fill it and lock everyone else out. This limits how many runs a single client may
start in a sliding window, so the cap stays available to other visitors.

State is in memory, like the job store, so it resets on a restart and is per instance. That is enough for
the single-instance demo; a multi-instance deployment would need a shared store.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Optional

from starlette.requests import Request


class RateLimiter:
    """At most `limit` runs per `window_seconds` for each client key. `limit` <= 0 disables the limit."""

    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if not hits:
            self._hits.pop(key, None)
            return deque()
        return hits

    def retry_after(self, key: str) -> Optional[int]:
        """Seconds until `key` may start another run, or None if it may start one now. Does not count a hit."""
        if not self.enabled:
            return None
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) < self.limit:
                return None
            return max(1, math.ceil(hits[0] + self.window - now))

    def record(self, key: str) -> None:
        """Count one started run against `key`."""
        if not self.enabled:
            return
        with self._lock:
            now = self._clock()
            for other in list(self._hits):  # drop every idle client, or a stream of new addresses grows this forever
                self._prune(other, now)
            self._hits[key].append(now)


def client_key(request: Request, trusted_proxies: int = 1) -> str:
    """The client's address for rate limiting.

    Behind Cloud Run the platform appends the real client address to X-Forwarded-For, after any value the
    client sent itself, so the entry `trusted_proxies` from the right is the one to trust; reading the
    left-most entry would let a caller pick their own key. With no header (local use, tests) the socket
    address is used.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[max(0, len(parts) - max(1, trusted_proxies))]
    return request.client.host if request.client else "unknown"
