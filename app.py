"""Streamlit front-end: AI-Powered Lead Intelligence and Recovery Agent (prototype).
Run:  streamlit run app.py
"""
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import agent
import analytics as an

BASE = Path(__file__).parent
st.set_page_config(page_title="Lead Intelligence Agent", layout="wide")
st.title("Lead Intelligence & Recovery Agent")
st.caption("Prototype. Rule-based, explainable, no API key. It only PROPOSES actions - "
           "nothing is sent or written to any CRM without human approval.")

# ------------------------------------------------------------------ input
with st.sidebar:
    st.header("1. Data source")
    choice = st.radio("Choose input", ["Upload CRM CSV", "Bundled: your uploaded Zoho export (classroom/demo data)",
                                       "Bundled: SYNTHETIC schema test file"])
    up = st.file_uploader("CRM CSV", type="csv") if choice.startswith("Upload") else None
    as_of = st.date_input("As-of date (used only if date columns exist)", value=date.today())
    st.header("2. Filters")
    excl_test = st.checkbox("Exclude test/placeholder records from ranking views", value=True)

try:
    if up is not None:
        df, label = an.load_csv(up), f"Uploaded file: {up.name}"
    elif choice.startswith("Bundled: your"):
        df, label = an.load_csv(BASE / "data" / "Leads_by_Status.csv"), "Classroom/demo Zoho export (NOT verified business data)"
    elif choice.startswith("Bundled: SYNTH"):
        df, label = an.load_csv(BASE / "data" / "SYNTHETIC_schema_test.csv"), "SYNTHETIC file - invented values for testing date/rating logic only"
    else:
        st.info("Upload a CRM lead CSV, or pick a bundled file in the sidebar.")
        st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read file: {e}")
    st.stop()

st.info(f"**Data in use:** {label}")
val = an.validate(df)
if not val["ok"]:
    st.error("Validation failed: " + " | ".join(val["issues"]))
    st.stop()

ranked = agent.analyze(df, as_of)
summ = an.dataset_summary(df)
view = ranked[ranked["Record Type"] != "Test/placeholder"] if excl_test else ranked

tabs = st.tabs(["Overview & validation", "Data quality", "Status & source", "Follow-up",
                "Priority queue", "Lead detail & draft", "Download"])

# --------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Records", summ["rows"])
    c[1].metric("Test/placeholder", summ["test_records"])
    c[2].metric("Zoho (Sample)", summ["sample_records"])
    c[3].metric("Missing status", summ["status_missing"])
    c[4].metric("High priority", int((ranked["Priority Tier"] == "High").sum()))
    st.subheader("Detected schema")
    st.dataframe(pd.DataFrame([{"Field": k, "Column in file": (v if v else "NOT FOUND")}
                               for k, v in val["mapping"].items() if k != "address"]), hide_index=True)
    if val["warnings"]:
        st.warning("Inconsistencies: " + " | ".join(val["warnings"]))
    st.subheader("What the file can and cannot support")
    for k, ok in val["capabilities"].items():
        st.write(("✅ " if ok else "⛔ ") + k.replace("_", " ").capitalize()
                 + ("" if ok else f" - {val['notes'].get(k, 'unavailable')}"))
    if "value_basis" in val["notes"]:
        st.caption(val["notes"]["value_basis"])

# ------------------------------------------------------------ data quality
with tabs[1]:
    mr = an.missing_report(df)
    st.altair_chart(alt.Chart(mr).mark_bar().encode(
        x=alt.X("Missing %:Q", scale=alt.Scale(domain=[0, 100])), y=alt.Y("Column:N", sort="-x"),
        tooltip=["Column", "Missing", "Missing %"]).properties(height=360), use_container_width=True)
    st.dataframe(mr, hide_index=True)
    if summ["fully_empty_columns"]:
        st.warning("Fully empty columns: " + ", ".join(summ["fully_empty_columns"]))
    dn = an.duplicate_names(df)
    if len(dn):
        st.write("Repeated names (possible duplicates - review, may be test data):")
        st.dataframe(dn, hide_index=True)
    st.write("Record types:")
    st.dataframe(ranked["Record Type"].value_counts().rename_axis("Type").reset_index(name="Records"), hide_index=True)

# ---------------------------------------------------------- status / source
with tabs[2]:
    sd = an.status_distribution(df)
    st.altair_chart(alt.Chart(sd).mark_bar().encode(x=alt.X("Count:Q"), y=alt.Y("Lead Status:N", sort="-x"),
                    tooltip=["Lead Status", "Count", "Share %"]), use_container_width=True)
    st.dataframe(sd, hide_index=True)
    sc = an.source_comparison(df)
    if sc is None:
        st.info("Source comparison unavailable: " + val["notes"].get("source_comparison", ""))
    else:
        st.subheader("Lead-source comparison")
        st.altair_chart(alt.Chart(sc).mark_bar().encode(x="Leads:Q", y=alt.Y("Source:N", sort="-x"),
                        tooltip=list(sc.columns)), use_container_width=True)
        st.dataframe(sc, hide_index=True)
        st.caption("Small counts and many missing sources: treat as descriptive only, not as evidence of source quality.")

# ---------------------------------------------------------------- follow-up
with tabs[3]:
    fu = an.followup_analysis(df, as_of)
    if not fu["available"]:
        st.info("Follow-up analysis not available: " + fu["reason"])
    else:
        st.write(f"As of {fu['as_of']}")
        if fu["buckets"] is not None:
            b = fu["buckets"].rename_axis("Days since last activity").reset_index(name="Leads")
            st.altair_chart(alt.Chart(b).mark_bar().encode(x=alt.X("Days since last activity:N", sort=None), y="Leads:Q"),
                            use_container_width=True)
        if "Follow-up overdue" in fu["table"]:
            st.metric("Overdue follow-ups", int(fu["table"]["Follow-up overdue"].sum()))

# ----------------------------------------------------------- priority queue
with tabs[4]:
    tier_sel = st.multiselect("Tiers", ["High", "Medium", "Low", "Not applicable"],
                              default=["High", "Medium", "Low"])
    q = view[view["Priority Tier"].isin(tier_sel)].copy()
    q.insert(0, "Approve?", False)
    cols = ["Approve?", "Rank", "Full Name", "Company", "Lead Status", "Value", "Priority Score",
            "Priority Tier", "Action Type", "Recommended Action", "Why Flagged"]
    st.caption("Tick 'Approve?' to mark an action as approved by you. The agent never executes anything.")
    edited = st.data_editor(q[cols], hide_index=True, use_container_width=True,
                            disabled=[c for c in cols if c != "Approve?"], key="queue")
    st.session_state["approved"] = q.loc[edited.index[edited["Approve?"]], "Full Name"].tolist()
    st.altair_chart(alt.Chart(ranked).mark_bar().encode(x="Priority Tier:N", y="count():Q",
                    color="Priority Tier:N"), use_container_width=True)

# -------------------------------------------------------------- lead detail
with tabs[5]:
    options = view.apply(lambda r: f"{r['Full Name']} - {r['Company']}", axis=1).tolist()
    pick = st.selectbox("Lead", options)
    r = view.iloc[options.index(pick)]
    st.write(f"**Status:** {r['Lead Status']}  |  **Tier:** {r['Priority Tier']}  |  **Score:** {r['Priority Score']}")
    st.write("**Score breakdown:** " + r["Score Breakdown"])
    st.write("**Why flagged:** " + r["Why Flagged"])
    st.write("**Recommended action:** " + r["Recommended Action"])
    if r["Draft Message"]:
        st.text_area("Draft message (edit freely; NOT sent)", r["Draft Message"], height=240)
    else:
        st.info("No outreach draft for this record type/status.")
    st.write("Approval state: " + r["Human Approval"])

# ----------------------------------------------------------------- download
with tabs[6]:
    st.download_button("Prioritised leads + actions (CSV)", ranked.to_csv(index=False).encode("utf-8"),
                       "lead_priorities_and_actions.csv", "text/csv")
    st.download_button("Data-quality report (CSV)", an.missing_report(df).to_csv(index=False).encode("utf-8"),
                       "data_quality_report.csv", "text/csv")
    st.download_button("Analysis summary (Markdown)", agent.build_summary_markdown(df, ranked).encode("utf-8"),
                       "analysis_summary.md", "text/markdown")
    st.caption("Exports mark every action 'Pending human approval'.")
