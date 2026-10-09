"""The ASGI entry point (causal_engine/api/main.py).

main.py calls load_dotenv() on import so a local `uvicorn causal_engine.api.main:app` picks up .env. These
tests replace load_dotenv so importing it here can never read the developer's real OPENAI_API_KEY.
"""

from __future__ import annotations

import importlib
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _import_main(monkeypatch, calls: list):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append((a, k)) or False)
    monkeypatch.delitem(sys.modules, "causal_engine.api.main", raising=False)
    return importlib.import_module("causal_engine.api.main")


def test_main_loads_dotenv_once_and_exposes_a_working_app(monkeypatch):
    calls: list = []
    main = _import_main(monkeypatch, calls)

    assert len(calls) == 1
    assert isinstance(main.app, FastAPI)
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/").status_code == 200  # the UI page is served from the entry point too
