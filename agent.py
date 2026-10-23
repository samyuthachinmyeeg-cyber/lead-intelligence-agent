"""agent.py - explainable, rule-based lead triage agent (no API key, no LLM).

Pipeline: validate -> classify records -> score (rules) -> explain -> recommend
-> draft. The agent only PROPOSES. Every action and message is marked
'Pending human approval'; nothing is sent or written back to a CRM.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import analytics as an

# Rule table: opportunity weight of each status (max 30 points).
STATUS_POINTS = {
    "pre-qualified": 30, "not contacted": 26, "attempted to contact": 24,
    "contacted": 20, "contact in future": 12,
    "lost lead": 3, "not qualified": 3, "junk lead": 0,
}
UNKNOWN_STATUS_POINTS = 8   # status blank or unrecognised
CLOSED = {"lost lead", "junk lead", "not qualified"}
MAX_STATUS, MAX_VALUE, MAX_COMPLETE, MAX_RATING, MAX_RECENCY = 30, 30, 25, 10, 10
RATING_POINTS = {"hot": 10, "warm": 6, "cold": 2}
TIERS = [(65, "High"), (40, "Medium"), (0, "Low")]
APPROVAL = "Pending human approval"

SOURCE_LINE = {
    "web download": "Thank you for your interest in the material you downloaded from us.",
    "online store": "Thank you for your recent interest through our online store.",
    "exhibition": "It was good to connect at the exhibition.",
    "employee referral": "A colleague suggested I get in touch.",
    "advertisement": "Thank you for responding to our advertisement.",
}


def _tier(score: float) -> str:
    return next(t for cut, t in TIERS if score >= cut)


def _clean(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _action(status: str, rtype: str, missing: list[str]) -> tuple[str, str]:
    """Return (action_type, recommended action text)."""
    if rtype == "Test/placeholder":
        return "Archive review", "Review and archive/delete this test or placeholder record (needs owner approval)."
    s = status.lower()
    if s == "junk lead":
        return "Confirm closure", "Confirm junk classification and exclude from pipeline reports."
    if s in ("lost lead", "not qualified"):
        return "Confirm closure", "No follow-up now. Record the loss reason (not in this export) before closing."
    if s == "":
        return "Complete data", "Assign a lead status after a quick review; then re-run the agent."
    if s == "not contacted":
        return "First contact", "Make a first contact attempt and log the outcome in the CRM."
    if s == "attempted to contact":
        return "Re-attempt", "Re-attempt via a different channel/time and log the result."
    if s == "contacted":
        return "Qualify", "Qualify the need, then update status (Pre-Qualified or Lost) and set a next step."
    if s == "pre-qualified":
        return "Advance", "Prioritise: propose a meeting or quotation and record the next-step date."
    if s == "contact in future":
        return "Schedule reminder", "Set a dated reminder; the export has no follow-up date field to confirm timing."
    return "Review status", f"Status '{status}' is not recognised by the rules; review manually."


def _draft(first, full, company, source, status, action_type) -> str:
    if action_type not in ("First contact", "Re-attempt", "Qualify", "Advance"):
        return ""
    who = first or full or "there"
    src = SOURCE_LINE.get(source.lower(), "")
    body = {
        "First contact": f"I would like to introduce [Your Company] and learn whether we can support {company}. Would you be open to a short call this week?",
        "Re-attempt": f"I tried to reach you recently regarding {company}. Is there a better time or channel to connect?",
        "Qualify": f"Thanks for speaking with us. To make sure we propose the right solution for {company}, could you share your main needs and timeline?",
        "Advance": f"Based on our conversation, I would like to propose a meeting (or share a quotation) for {company}. What time suits you?",
    }[action_type]
    return (f"[DRAFT - verify context and get human approval before sending]\n"
            f"Subject: Following up - {company}\n\nHi {who},\n\n{src} {body}\n\n"
            f"Kind regards,\n[Your Name]\n[Your Company]").replace("\n\n ", "\n\n")


def analyze(df: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    """Score and explain every lead. Returns one row per lead."""
    v = an.validate(df)
    if not v["ok"]:
        raise ValueError("; ".join(v["issues"]))
    m = v["mapping"]
    n = len(df)
    rtypes = an.record_types(df)

    # --- value signal (budget preferred, else annual revenue)
    vcol = m["budget"] or m["revenue"]
    value = an.to_number(df[vcol]) if vcol else pd.Series(np.nan, index=df.index)
    pool = value.where(rtypes != "Test/placeholder")
    value_pct = pool.rank(pct=True)          # percentile among non-test leads with a value
    # --- optional recency signal
    fu = an.followup_analysis(df, as_of)
    recency_pts = pd.Series(0.0, index=df.index)
    have_recency = fu["available"] and "Days since last activity" in fu["table"]
    if have_recency:
        d = fu["table"]["Days since last activity"]
        recency_pts = np.select([d > 90, d > 30, d > 7], [10, 7, 3], default=0).astype(float)
        recency_pts = pd.Series(recency_pts, index=df.index).where(d.notna(), 0)
    have_rating = m["rating"] is not None

    rows = []
    for i, r in df.iterrows():
        get = lambda k: _clean(r[m[k]]) if m.get(k) else ""
        status, source, first = get("status"), get("source"), get("first_name")
        full, company, owner = get("name"), get("company"), get("owner")
        rtype = rtypes[i]

        key_fields = {"Lead Source": source, "Lead Status": status, "First Name": first}
        if vcol:
            key_fields[vcol] = _clean(r[vcol])
        if m["email"]:
            key_fields["Email"] = get("email")
        if m["phone"]:
            key_fields["Phone"] = get("phone")
        missing = [k for k, val in key_fields.items() if val == ""]
        comp_frac = 1 - len(missing) / len(key_fields)

        s_pts = STATUS_POINTS.get(status.lower(), UNKNOWN_STATUS_POINTS)
        v_pts = MAX_VALUE * (value_pct[i] if pd.notna(value_pct[i]) else 0)
        c_pts = MAX_COMPLETE * comp_frac
        rt_pts = RATING_POINTS.get(get("rating").lower(), 0) if have_rating else 0
        rc_pts = float(recency_pts[i]) if have_recency else 0
        max_pts = MAX_STATUS + MAX_VALUE + MAX_COMPLETE + (MAX_RATING if have_rating else 0) + (MAX_RECENCY if have_recency else 0)
        score = round(100 * (s_pts + v_pts + c_pts + rt_pts + rc_pts) / max_pts, 1)

        a_type, action = _action(status, rtype, missing)
        closed = status.lower() in CLOSED or rtype == "Test/placeholder"
        tier = "Not applicable" if closed else _tier(score)

        # --- explanations (every claim traceable to a field)
        why = []
        if rtype == "Test/placeholder":
            why.append("Name/company matches test or placeholder wording (test/delete/demo).")
        if rtype == "Zoho sample":
            why.append("Contains '(Sample)' - looks like Zoho's built-in sample data, not a verified lead.")
        why.append(f"Status: {status or 'missing'} (+{s_pts} of {MAX_STATUS} status points).")
        if vcol:
            why.append(f"{vcol}: {value[i]:,.0f} (value percentile {value_pct[i]*100:.0f}%)." if pd.notna(value_pct[i])
                       else f"No usable {vcol}; value could not be scored.")
        if missing:
            why.append("Missing: " + ", ".join(missing) + ".")
        if have_rating and get("rating"):
            why.append(f"Rating: {get('rating')}.")
        if have_recency and rc_pts:
            why.append(f"{int(fu['table']['Days since last activity'][i])} days since last activity.")
        if not (m["email"] or m["phone"]):
            why.append("No email/phone field in the export, so outreach channel is unknown.")

        draft = "" if closed else _draft(first, full, company, source, status, a_type)
        rows.append({
            "Full Name": full, "Company": company, "Lead Owner": owner,
            "Lead Source": source or "(Missing)", "Lead Status": status or "(Missing)",
            "Value": value[i] if pd.notna(value[i]) else np.nan,
            "Record Type": rtype, "Priority Score": score, "Priority Tier": tier,
            "Score Breakdown": (f"status {s_pts:.0f}/{MAX_STATUS} | value {v_pts:.1f}/{MAX_VALUE} | "
                                f"completeness {c_pts:.1f}/{MAX_COMPLETE}"
                                + (f" | rating {rt_pts}/{MAX_RATING}" if have_rating else "")
                                + (f" | recency {rc_pts:.0f}/{MAX_RECENCY}" if have_recency else "")),
            "Missing Fields": ", ".join(missing), "Why Flagged": " ".join(why),
            "Action Type": a_type, "Recommended Action": action,
            "Draft Message": draft, "Human Approval": APPROVAL,
        })
    out = pd.DataFrame(rows)
    out.insert(0, "Rank", pd.array([pd.NA] * n, dtype="Int64"))
    act = out["Priority Tier"].isin(["High", "Medium", "Low"])
    order = out[act].sort_values(["Priority Score", "Value"], ascending=[False, False], na_position="last").index
    out.loc[order, "Rank"] = np.arange(1, len(order) + 1)
    out["_o"] = np.where(act, 0, 1)
    out = out.sort_values(["_o", "Rank", "Priority Score"], ascending=[True, True, False],
                          na_position="last").drop(columns="_o").reset_index(drop=True)
    out.attrs["value_column"] = vcol
    return out


def build_summary_markdown(df: pd.DataFrame, ranked: pd.DataFrame) -> str:
    """Plain-text analysis summary for download."""
    v = an.validate(df)
    s = an.dataset_summary(df)
    lines = ["# Lead Intelligence Summary", "",
             "_Generated by a rule-based prototype. Recommendations are proposals that require human approval._", "",
             f"- Rows: {s['rows']}  |  Columns: {s['columns']}",
             f"- Test/placeholder records: {s['test_records']}  |  Zoho '(Sample)' records: {s['sample_records']}",
             f"- Missing status: {s['status_missing']}  |  Missing source: {s['source_missing']}",
             f"- Fully empty columns: {', '.join(s['fully_empty_columns']) or 'none'}", "",
             "## Status distribution", an.status_distribution(df).to_markdown(index=False), "",
             "## Analyses not available", ]
    lines += [f"- {k}: {t}" for k, t in v["notes"].items()] or ["- none"]
    lines += ["", "## Tier counts", ranked["Priority Tier"].value_counts().to_markdown(), "",
              "## Top 10 prioritised leads",
              ranked[ranked["Rank"].notna()].head(10)[["Rank", "Full Name", "Company", "Lead Status", "Priority Score", "Recommended Action"]].to_markdown(index=False)]
    return "\n".join(lines)
