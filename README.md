# RootCause

A domain-agnostic **causal decision intelligence** pipeline (the CDIA architecture). Instead of predicting *who* will churn, it asks *why* — learns the causal structure of a domain, estimates how much each lever actually moves the outcome, and ranks interventions by ROI with a fairness check.

Phase 1 is a full 7-stage vertical slice on one domain, **employee attrition**, run against a semi-synthetic dataset with a known ground-truth causal graph so every stage can be checked against the truth.

## Pipeline

| # | Stage | What it does | Built with |
|---|-------|--------------|------------|
| 1 | Ingestion | Loads and validates the domain CSV (CSV only for now) | pandas |
| 2 | Feature store | Writes/materializes causal feature vectors and reads them back through the online-serving path | Feast (parquet offline, SQLite online) |
| 3 | Causal discovery | Learns the DAG, seeded with required/forbidden domain-prior edges | causal-learn (PC) |
| 4 | Effect estimation | ATE per treatment with a placebo refutation check | DoWhy |
| 5 | Counterfactuals | "What if satisfaction were high?" via a T-learner | CausalML |
| 6 | Interventions | Ranks candidate actions by ROI; four-fifths-rule fairness check | fairlearn |
| 7 | Explanation | SHAP attribution plus a plain-language narrative (template fallback if no API key) | SHAP, OpenAI |

Each stage is a pure function in `rootcause/pipeline/`. They can run directly, or behind a **CrewAI hierarchical crew** (`rootcause/agents/crew.py`): 7 specialized agents plus a manager LLM that delegates one stage to each.

### What it finds on the bundled dataset

On the 2,000-row synthetic data (`data/employee_attrition/`, seed 42) the pipeline recovers all 6 ground-truth edges with no spurious ones, and the three levers get correctly-signed effects that pass refutation:

| Lever | ATE on attrition |
|-------|------------------|
| manager_quality | −0.135 |
| compensation | −0.129 |
| workload | +0.119 |

Ranked interventions: manager training, then workload rebalancing, then compensation adjustment. The fairness check on `gender` passes for all three, as designed: the synthetic data deliberately does not wire gender into any structural equation.

> These numbers come from synthetic data with a known answer, which is the point — real observational data has no ground truth to validate a causal claim against.

## Setup

Requires **Python 3.11** (causal-learn / DoWhy / CausalML wheel support lags on newer versions).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip      # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
python scripts/generate_synthetic_attrition_data.py         # only needed to regenerate the dataset
```

The dataset is already in `data/employee_attrition/`. The generator rewrites `attrition.csv` and `ground_truth.json`.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want an LLM-written narrative or the crew. Without it, the direct pipeline still runs and uses a template narrative.

## Run the API

```bash
.venv/Scripts/python.exe -m uvicorn rootcause.api.main:app
```

Interactive docs at <http://localhost:8000/docs>.

| Method | Path | |
|--------|------|-|
| GET | `/health` | Liveness |
| GET | `/domains` | Configured domains and whether each can run |
| POST | `/domains/{id}/analyze` | Queue a run → `202` + job. Body: `{"orchestration": "direct" \| "crew"}` (default `direct`) |
| GET | `/jobs` | Recent jobs (no results) |
| GET | `/jobs/{job_id}` | Status; carries the full result once `succeeded` |

```bash
curl -X POST localhost:8000/domains/employee_attrition/analyze
# {"id": "ed1c…", "status": "queued", …}
curl localhost:8000/jobs/ed1c…
# {"status": "succeeded", "result": {"causal_graph": …, "effect_estimates": …, "recommendations": …, "explanation": …}}
```

**Orchestration modes**

- `direct` — calls the stage functions in order. Takes seconds.
- `crew` — the hierarchical CrewAI run. Takes several minutes and makes many paid OpenAI calls per run, far more than the analysis itself needs (manager/delegation overhead). Returns `400` immediately if `OPENAI_API_KEY` isn't set.

**Limits to know about**

- Runs are executed one at a time. Stage 2 writes a single shared Feast store, so concurrent runs would corrupt each other.
- Jobs are kept in memory only (newest 100), so they're lost on restart.
- No auth and no upload: the API always analyzes the dataset named in the domain config.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

51 tests, about 12 seconds. They run real code on the real dataset (Feast round-trip, DoWhy, causal-learn) with no mocks, and assert the pipeline still recovers the ground-truth graph and effect signs. They never call OpenAI. The crew is only checked for wiring; a full crew run is deliberately not in the suite because of its cost.

## Adding a domain

A domain is one YAML file in `rootcause/configs/`. The `domain` section is validated; every stage reads its own section (`ingestion`, `feature_store`, `causal_discovery`, `effect_estimation`, `counterfactuals`, `interventions`, `explanation`). See `employee_attrition.yaml` for a complete example.

One thing isn't validated and fails silently: each name in `effect_estimation.treatments` must match an `interventions.candidates[].target_variable`. If they don't line up, Stage 6 returns an empty recommendation list rather than an error.

## Layout

```
rootcause/
  api/          FastAPI app + in-memory job store
  agents/       CrewAI hierarchical orchestration
  pipeline/     the 7 stage functions + runner (direct / crew)
  feature_repo/ Feast definitions and local stores
  models/       Pydantic contracts passed between stages
  configs/      one YAML per domain
  utils/        config loader
scripts/        synthetic data generator
data/           datasets + ground truth
tests/
```

## Scope

Deliberately not built in this phase, per the architecture spec: GAN-based counterfactuals (V2), audio/image/PDF ingestion, the insurance-claims domain, ChromaDB memory, human-in-the-loop review, and AutoGen for the explanation stage. Counterfactuals use CausalML meta-learners.
