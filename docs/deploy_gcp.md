# Deploying the RootCause UI + API to Google Cloud Run

Cloud Run bills per use and scales to zero; the monthly free grant (180k vCPU-seconds, 2M requests, checked
against Google's pricing page before you commit) should cover a recruiter demo. A card is required.

What the deployed copy can do: the saved examples (committed JSON, including the Freddie Mac ones) and live
runs on the committed datasets (employee attrition, German Credit, carclaims, Illinois wellness). Freddie Mac
live runs are not offered, because that data is gitignored; `/domains` hides datasets whose file is absent.

## One-time setup
1. Install the Google Cloud CLI, then `gcloud auth login`.
2. Create a project and enable billing for it. `gcloud config set project <PROJECT_ID>`.
3. `gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com`
4. **Budget alert first:** Billing -> Budgets & alerts -> create a budget of a few dollars with email alerts.

## Deploy (from the repo root; `.dockerignore` keeps the 1 GB+ Freddie Mac data out of the build)
```
gcloud run deploy rootcause --source . --region us-central1 --allow-unauthenticated \
  --memory 4Gi --cpu 2 --timeout 600 --max-instances 1 --concurrency 4
```
- `--max-instances 1` caps cost and matches the single-worker design (runs are serialized).
- `--timeout 600`: the longest live run seen is well under this; raise it if one is cut off.
- The first build takes 10-20 minutes (dowhy, causalml, feast, autogen layers).
- The URL it prints is the public link. Check `<url>/health`, then open `<url>/`.

## Optional: AutoGen narratives
Without a key the narrative is the template tier (the page labels this). To enable the LLM tiers, store the key
as a secret rather than an env var in plain text, and accept that every visitor run then spends your credits
(there is no rate limit):
```
gcloud run services update rootcause --set-secrets OPENAI_API_KEY=openai-key:latest
```

## Cost control / teardown
`gcloud run services delete rootcause --region us-central1` stops everything. Instances are not kept warm
(`--min-instances` defaults to 0), so the first visitor after idle waits for a cold start.
