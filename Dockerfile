# RootCause API. Build:  docker build -t rootcause .    Run:  docker run -p 8000:8000 rootcause
# Python 3.11 is required (causal-learn / DoWhy / CausalML wheel support lags on newer versions).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Compilers for any dependency without a wheel; libgomp1 for the OpenMP runtime scikit-learn/lightgbm use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Two layers, and requirements.txt first on its own: it is the expensive, rarely-changing install
# (dowhy, causalml, feast, crewai, ...), and pip's `--retries` covers a flaky connection dropping a
# single wheel mid-download (seen once as a hash mismatch after a very long build). Splitting means
# a retry of the small dev-only layer does not repeat the big one.
COPY requirements.txt ./
RUN pip install --retries 5 -r requirements.txt

# requirements-dev.txt adds pytest and httpx, which the CI job (docker run ... pytest) needs; the
# API alone does not.
COPY requirements-dev.txt ./
RUN pip install --retries 5 -r requirements-dev.txt

# AutoGen (tier 1 of the Stage 7 narrative chain) goes in LAST, then protobuf is put back: autogen-core
# pins protobuf 5.x, which breaks crewai/feast. Without it tests/test_narrative.py (34 tests, incl. the
# grounding check) is skipped whole. `pip check` reports autogen-core's pin afterwards; harmless.
COPY requirements-autogen.txt ./
RUN pip install --retries 5 -r requirements-autogen.txt \
    && pip install --no-deps "protobuf>=6.33.5,<7"

COPY . .

# OPENAI_API_KEY is read from the environment (docker run -e OPENAI_API_KEY=...); without it the
# pipeline uses the template narrative. Never bake a key into the image.
EXPOSE 8000
CMD ["uvicorn", "rootcause.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
