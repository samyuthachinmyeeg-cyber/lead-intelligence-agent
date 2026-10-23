# AI-Powered Lead Intelligence and Recovery Agent (Prototype)

Rule-based, explainable lead-triage agent for CRM exports. Python + Streamlit. **No API key, no LLM, no CRM connection.**
It only *proposes* actions and draft messages; every item is marked "Pending human approval". Nothing is sent or written back.

> The bundled `data/Leads_by_Status.csv` is a **classroom/demo Zoho export** (31 rows), not verified business data.
> `data/SYNTHETIC_schema_test.csv` is **invented** (8 rows, `example.invalid` emails) and exists only to exercise date/rating/email logic the real file lacks.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
python -m pytest -q                # run tests (11)
```
Python 3.10+ recommended.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, validation, charts, priority queue, lead detail, downloads |
| `analytics.py` | Column detection, validation, missing-value, status, source, follow-up analytics |
| `agent.py` | Rule-based scoring, explanations, recommended actions, draft messages |
| `tests/` | pytest tests for analytics, agent rules and app start-up |
| `docs/` | Project report, PPT content, demo script |

## How the agent works
1. Load CSV, auto-detect columns by header name (Zoho-style names).
2. Validate; list which analyses the file can/cannot support (e.g. no date column -> follow-up analysis disabled, with reason).
3. Classify records: test/placeholder (test/delete/demo/dummy), Zoho "(Sample)", standard.
4. Score 0-100 with fixed rules: status (30) + value from budget or annual revenue percentile (30) + field completeness (25) + rating (10, only if column exists) + recency (10, only if >=5 valid dates and >=50% of rows). Score is normalised to the signals available.
5. Tier: High >= 65, Medium 40-64, Low < 40. Test, lost, junk and not-qualified records are "Not applicable".
6. Explain every flag, recommend one action, draft a message where appropriate.

## Share a demo link (nothing is deployed by this project)
- **Streamlit Community Cloud (free):** push this folder to a GitHub repo, sign in at share.streamlit.io, choose the repo and `app.py`. Before making a repo public, consider removing `data/Leads_by_Status.csv` (it contains owner/person names) and choosing the synthetic file or upload mode.
- **Temporary tunnel:** with the app running, `cloudflared tunnel --url http://localhost:8501` (or `ngrok http 8501`) prints a public URL while your laptop stays on.

## Known limitations
Rule weights are judgement-based, not statistically fitted; revenue is company size, not deal budget; small, mixed-scale demo data; no conversion outcomes, so no predictive claims.
