"""analytics.py - validation and descriptive analytics for a CRM lead export.

Nothing here assumes a particular business. Every analysis checks whether the
required columns exist and holds enough valid data; otherwise it returns an
explicit "not available" reason instead of guessing.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

# Canonical field -> accepted header names (lower-case, matched after strip).
ALIASES = {
    "name": ["full name", "name", "lead name"],
    "first_name": ["first name"],
    "last_name": ["last name"],
    "company": ["company", "account name", "organization", "organisation"],
    "owner": ["lead owner", "owner"],
    "source": ["lead source", "source"],
    "status": ["lead status", "status"],
    "revenue": ["annual revenue", "revenue"],
    "budget": ["budget", "deal amount", "expected revenue", "amount"],
    "rating": ["rating", "lead rating", "lead quality", "lead score"],
    "email": ["email", "email address"],
    "phone": ["phone", "mobile", "phone number"],
    "last_activity": ["last activity time", "last activity", "last contacted",
                      "last contact date", "last contacted time"],
    "created": ["created time", "created date", "date created"],
    "next_followup": ["next follow-up", "next follow up", "next follow-up date",
                      "next activity date"],
}
ADDRESS_COLS = ["street", "city", "state", "country", "zip code"]
MIN_VALID_DATES = 5          # absolute minimum valid dates for follow-up analysis
MIN_VALID_DATE_SHARE = 0.5   # and at least this share of rows
TEST_PATTERN = re.compile(r"(?<![a-z])(test|delete|demo|dummy)(?![a-z])", re.I)
SAMPLE_PATTERN = re.compile(r"\(sample\)", re.I)


# ---------------------------------------------------------------- loading
def load_csv(file_or_path) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, trying common encodings."""
    raw = file_or_path.read() if hasattr(file_or_path, "read") else open(file_or_path, "rb").read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Could not read file as CSV: {last_err}")


def blank_mask(s: pd.Series) -> pd.Series:
    """True where a cell is NaN or whitespace-only."""
    txt = s.astype("string").str.strip()
    return (s.isna() | txt.eq("")).fillna(True).astype(bool)


def detect_columns(df: pd.DataFrame) -> dict:
    """Map canonical field names to actual column names (None if absent)."""
    lower = {c.lower().strip(): c for c in df.columns}
    mapping = {k: next((lower[a] for a in v if a in lower), None) for k, v in ALIASES.items()}
    mapping["address"] = [lower[c] for c in ADDRESS_COLS if c in lower]
    return mapping


# ------------------------------------------------------------- validation
def to_number(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.replace(r"[,\s₹$€£]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def validate(df: pd.DataFrame) -> dict:
    """Structural validation: schema, capabilities, inconsistencies."""
    m = detect_columns(df)
    issues, warnings = [], []
    if df.empty:
        issues.append("File has no data rows.")
    for req in ("name", "company", "status"):
        if m[req] is None:
            issues.append(f"Required column for '{req}' not found.")
    if m["status"] is None and m["name"] is None and m["company"] is None:
        issues.append("None of the core CRM columns were recognised - is this a CRM lead export?")

    caps = {
        "status_analysis": m["status"] is not None,
        "source_comparison": m["source"] is not None and m["status"] is not None,
        "value_prioritisation": (m["budget"] or m["revenue"]) is not None,
        "rating_prioritisation": m["rating"] is not None,
        "contactability_check": bool(m["email"] or m["phone"]),
        "followup_analysis": False,
    }
    notes = {}
    # Inconsistency checks
    for key in ("revenue", "budget"):
        col = m[key]
        if col:
            raw_nonblank = ~blank_mask(df[col])
            num = to_number(df[col])
            bad = int((raw_nonblank & num.isna()).sum())
            neg = int((num < 0).sum())
            if bad:
                warnings.append(f"{bad} non-numeric value(s) in '{col}'.")
            if neg:
                warnings.append(f"{neg} negative value(s) in '{col}'.")
    for key in ("last_activity", "created", "next_followup"):
        col = m[key]
        if col:
            nonblank = ~blank_mask(df[col])
            valid = parse_dates(df[col]).notna()
            bad = int((nonblank & ~valid).sum())
            if bad:
                warnings.append(f"{bad} unparseable date(s) in '{col}'.")
    if m["email"]:
        e = df[m["email"]].astype("string")
        bad = int((~blank_mask(df[m["email"]]) & ~e.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)).sum())
        if bad:
            warnings.append(f"{bad} malformed email address(es).")
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        warnings.append(f"{dup_rows} fully duplicated row(s).")

    # Follow-up capability needs enough valid dates
    fu_col = m["last_activity"] or m["next_followup"]
    if fu_col:
        valid = int(parse_dates(df[fu_col]).notna().sum())
        enough = valid >= MIN_VALID_DATES and valid / max(len(df), 1) >= MIN_VALID_DATE_SHARE
        caps["followup_analysis"] = enough
        if not enough:
            notes["followup_analysis"] = (f"Only {valid} valid date(s) in '{fu_col}' out of {len(df)} rows "
                                          f"(need >= {MIN_VALID_DATES} and >= {int(MIN_VALID_DATE_SHARE*100)}%).")
    else:
        notes["followup_analysis"] = "No activity or follow-up date column in this file."
    if not caps["source_comparison"]:
        notes["source_comparison"] = "Lead source and/or status column missing."
    if not caps["value_prioritisation"]:
        notes["value_prioritisation"] = "No budget or annual-revenue column."
    if not caps["rating_prioritisation"]:
        notes["rating_prioritisation"] = "No rating / lead-quality column."
    if not caps["contactability_check"]:
        notes["contactability_check"] = "No email or phone column - drafts cannot be addressed to a channel."
    if m["revenue"] and not m["budget"]:
        notes["value_basis"] = ("'Annual Revenue' describes the prospect company's size, not a deal budget; "
                                "it is used only as a proxy for potential value.")
    return {"rows": len(df), "columns": len(df.columns), "mapping": m, "issues": issues,
            "warnings": warnings, "capabilities": caps, "notes": notes,
            "ok": not issues}


# ---------------------------------------------------------- data quality
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        n = int(blank_mask(df[c]).sum())
        rows.append({"Column": c, "Missing": n, "Missing %": round(100 * n / max(len(df), 1), 1),
                     "Present": len(df) - n, "Fully empty": n == len(df)})
    return pd.DataFrame(rows).sort_values("Missing %", ascending=False).reset_index(drop=True)


def classify_record(row, m) -> str:
    """'Test/placeholder', 'Zoho sample', or 'Standard'."""
    text = " ".join(str(row[m[k]]) for k in ("name", "company") if m.get(k) and pd.notna(row[m[k]]))
    if TEST_PATTERN.search(text):
        return "Test/placeholder"
    if SAMPLE_PATTERN.search(text):
        return "Zoho sample"
    return "Standard"


def record_types(df: pd.DataFrame) -> pd.Series:
    m = detect_columns(df)
    return df.apply(lambda r: classify_record(r, m), axis=1)


def duplicate_names(df: pd.DataFrame) -> pd.DataFrame:
    m = detect_columns(df)
    if not m["name"]:
        return pd.DataFrame(columns=["Name (normalised)", "Records"])
    n = df[m["name"]].astype("string").str.strip().str.lower()
    vc = n.value_counts()
    vc = vc[vc > 1]
    return pd.DataFrame({"Name (normalised)": vc.index.astype(str), "Records": vc.values})


# ---------------------------------------------------------- distributions
def status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    m = detect_columns(df)
    if not m["status"]:
        return pd.DataFrame(columns=["Lead Status", "Count", "Share %"])
    s = df[m["status"]].where(~blank_mask(df[m["status"]]), "(Missing)")
    vc = s.value_counts()
    out = pd.DataFrame({"Lead Status": vc.index.astype(str), "Count": vc.values})
    out["Share %"] = (100 * out["Count"] / out["Count"].sum()).round(1)
    return out


def source_comparison(df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-source counts, status mix and mean value. None if unsupported."""
    m = detect_columns(df)
    if not (m["source"] and m["status"]):
        return None
    d = pd.DataFrame({
        "Source": df[m["source"]].where(~blank_mask(df[m["source"]]), "(Missing)").astype(str),
        "Status": df[m["status"]].where(~blank_mask(df[m["status"]]), "(Missing)").astype(str),
    })
    vcol = m["budget"] or m["revenue"]
    d["Value"] = to_number(df[vcol]) if vcol else np.nan
    g = d.groupby("Source")
    out = g.size().rename("Leads").to_frame()
    out["Status recorded"] = g["Status"].apply(lambda x: int((x != "(Missing)").sum()))
    out["Pre-Qualified"] = g["Status"].apply(lambda x: int((x == "Pre-Qualified").sum()))
    out["Contacted"] = g["Status"].apply(lambda x: int((x == "Contacted").sum()))
    out["Lost/Junk"] = g["Status"].apply(lambda x: int(x.isin(["Lost Lead", "Junk Lead"]).sum()))
    out["Leads with value"] = g["Value"].apply(lambda x: int(x.notna().sum()))
    out["Median value"] = g["Value"].median()
    return out.sort_values("Leads", ascending=False).reset_index()


def followup_analysis(df: pd.DataFrame, as_of: date | None = None) -> dict:
    """Days since last activity / overdue follow-ups, only with valid dates."""
    v = validate(df)
    m = v["mapping"]
    if not v["capabilities"]["followup_analysis"]:
        return {"available": False, "reason": v["notes"].get("followup_analysis", "Not available."), "table": None}
    as_of = pd.Timestamp(as_of or datetime.today().date())
    out = pd.DataFrame(index=df.index)
    if m["last_activity"]:
        last = parse_dates(df[m["last_activity"]])
        out["Days since last activity"] = (as_of - last).dt.days
        bins = [-np.inf, 7, 30, 90, np.inf]
        labels = ["0-7 days", "8-30 days", "31-90 days", "90+ days"]
        out["Recency bucket"] = pd.cut(out["Days since last activity"], bins=bins, labels=labels)
    if m["next_followup"]:
        nxt = parse_dates(df[m["next_followup"]])
        out["Follow-up overdue"] = (nxt < as_of) & nxt.notna()
    buckets = (out["Recency bucket"].value_counts().reindex(
        ["0-7 days", "8-30 days", "31-90 days", "90+ days"]).fillna(0).astype(int)
        if "Recency bucket" in out else None)
    return {"available": True, "reason": "", "table": out, "buckets": buckets, "as_of": as_of.date()}


def dataset_summary(df: pd.DataFrame) -> dict:
    """Headline numbers used by the app and the exported report."""
    m = detect_columns(df)
    rt = record_types(df)
    mr = missing_report(df)
    return {
        "rows": len(df), "columns": len(df.columns),
        "test_records": int((rt == "Test/placeholder").sum()),
        "sample_records": int((rt == "Zoho sample").sum()),
        "status_missing": int(blank_mask(df[m["status"]]).sum()) if m["status"] else None,
        "source_missing": int(blank_mask(df[m["source"]]).sum()) if m["source"] else None,
        "value_present": int((~blank_mask(df[m["budget"] or m["revenue"]])).sum()) if (m["budget"] or m["revenue"]) else None,
        "fully_empty_columns": mr.loc[mr["Fully empty"], "Column"].tolist(),
        "repeated_names": int(duplicate_names(df)["Records"].sum()) if m["name"] else 0,
    }
