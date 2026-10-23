# AI-Powered Lead Intelligence and Recovery Agent (Prototype)

A rule-based, explainable Streamlit prototype for CRM lead triage. It proposes actions and draft messages only; it does not send messages or write back to a CRM.

## Demo data
`data/Leads_by_Status.csv` is classroom/demo data and is loaded by default so reviewers can open the app without uploading a file. It is not verified business data. Check the file for personal or confidential information before placing this repository in public.

## Files
- `app.py`: Streamlit dashboard
- `agent.py`: rule-based scoring, explanations, and recommended actions
- `analytics.py`: schema validation and descriptive analytics
- `requirements.txt`: required packages
- `data/Leads_by_Status.csv`: bundled classroom/demo export

## Deploy
On Streamlit Community Cloud, select this repository, branch `main`, and main file path `app.py`.
