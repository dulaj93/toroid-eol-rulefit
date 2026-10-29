"""
Shared loaders and helpers for the Streamlit front-end (streamlit_app.py and app_pages/).
"""
import json
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from toroid_preprocess import load_raw, clean, first_attempts, FeatureBuilder

HERE = Path(__file__).parent
DATA = HERE / "toroid_test_data.csv"
MODELS = HERE / "models"
TEST_START = pd.Timestamp("2026-07-01")

# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "rulefit_eol.joblib")
    fb = FeatureBuilder.from_json(MODELS / "feature_builder.json")
    meta = json.loads((MODELS / "model_meta.json").read_text())
    return model, fb, meta


@st.cache_data(max_entries=2)
def load_units():
    """First EOL attempt of every unit (cleaned)."""
    return first_attempts(clean(load_raw(DATA)))


@st.cache_data(max_entries=2)
def score_frame(df: pd.DataFrame) -> np.ndarray:
    model, fb, _ = load_model()
    return model.predict_proba(fb.transform(df))[:, 1]


@st.cache_data(max_entries=2)
def example_units() -> pd.DataFrame:
    """Curated real units from the untouched test period (Jul–Aug 2026)."""
    fa = load_units()
    te = fa[fa.wind_end_ts >= TEST_START].copy()
    te["risk"] = score_frame(te)
    fails = te[te.y_fail == 1]
    picks = [fails.nlargest(6, "risk")]
    for code in sorted(fails.fail_code.unique()):
        picks.append(fails[fails.fail_code == code].nlargest(2, "risk"))
    passes = te[te.y_fail == 0]
    picks += [passes.nsmallest(4, "risk"), passes.nlargest(3, "risk"),
              passes.sample(5, random_state=7)]
    ex = pd.concat(picks).drop_duplicates("serial_no")
    ex["label"] = (ex.serial_no + " · " + ex.product_code + " · " + ex.winding_machine
                   + " · actual: " + ex.fail_code)
    return ex.sort_values("risk", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- labels
LABELS = {
    "winding_area_RH_pct": "Shop-floor RH (%)",
    "winding_area_temp_C": "Shop-floor temperature (°C)",
    "pri_tension_rel": "Primary tension ÷ recipe",
    "sec_tension_rel": "Secondary tension ÷ recipe",
    "pri_speed_rel": "Primary speed ÷ recipe",
    "sec_speed_rel": "Secondary speed ÷ recipe",
    "pri_tension_N": "Primary tension (N)",
    "sec_tension_N": "Secondary tension (N)",
    "pri_wind_speed_rpm": "Primary speed (rpm)",
    "sec_wind_speed_rpm": "Secondary speed (rpm)",
    "steel_supplier": "Steel supplier",
    "steel_coil_lot": "Steel coil lot",
    "coil_cert_loss_W_kg": "Coil certificate loss (W/kg)",
    "cert_loss_rel_grade": "Coil loss ÷ grade median",
    "anneal_ref_ring_loss_W_kg": "Anneal ref-ring loss (W/kg)",
    "anneal_peak_temp_C": "Anneal peak temp (°C)",
    "anneal_soak_min": "Anneal soak (min)",
    "anneal_dewpoint_C": "Anneal dew point (°C)",
    "anneal_furnace": "Anneal furnace",
    "anneal_to_wind_days": "Anneal → winding (days)",
    "core_deburred": "Core deburred",
    "core_coat_thickness_um": "Core epoxy coat (µm)",
    "core_mass_g": "Core mass (g)",
    "core_mass_rel": "Core mass ÷ product median",
    "tape_overlap_pct": "Tape overlap (%)",
    "interwinding_layers": "Inter-winding layers",
    "interwinding_tape": "Inter-winding tape",
    "operator_id": "Operator",
    "operator_experience_months": "Operator experience (months)",
    "shift": "Shift",
    "winding_machine": "Winding machine",
    "wire_supplier": "Wire supplier",
    "wip_hold_h": "WIP hold before test (h)",
    "inproc_pri_dcr_ohm": "In-process primary DCR (Ω)",
    "inproc_dcr_rel": "In-process DCR ÷ product median",
    "pri_turn_dev": "Primary turn deviation",
    "sec_turn_dev": "Secondary turn deviation",
    "impregnation": "Impregnation",
    "product_code": "Product",
    "core_ID_mm": "Core ID (mm)",
}


def label(f):
    return LABELS.get(f, f.replace("_", " "))


_OPS = {"<=": "≤", ">": ">", "!=": "≠"}


def pretty_rule(term: str) -> str:
    """'pri_tension_rel <= 1.116 and steel_supplier != STL-S3' -> readable text."""
    if term.startswith("linear: "):
        return f"linear effect of {label(term[8:])}"
    parts = []
    for p in term.split(" and "):
        m = re.match(r"^(\S+) (<=|>|!=) (.+)$", p)
        if m:
            f, op, v = m.groups()
            parts.append(f"{label(f)} {_OPS[op]} {v}")
        elif " is missing" in p or " is present" in p:
            f, rest = p.split(" is ")
            parts.append(f"{label(f)} is {rest}")
        elif "=" in p:
            f, v = p.split("=", 1)
            parts.append(f"{label(f)} = {v}")
        else:
            parts.append(p)
    return " AND ".join(parts)


# ---------------------------------------------------------------- decision policy
def risk_band(p, thr):
    """Action policy: threshold from the 10:1 cost analysis on the validation month."""
    if p >= 0.20:
        return "Hold for engineer review", "red", ":material/report:"
    if p >= thr:
        return "Route to in-process check", "orange", ":material/rule:"
    return "Standard flow", "green", ":material/check_circle:"


# ---------------------------------------------------------------- explanations
def contributions(model, X):
    lc, base = model.local_contributions(X)
    return lc.iloc[0], base


def waterfall_chart(contrib: pd.Series, base_logit: float, thr: float, top: int = 10):
    """Horizontal waterfall of per-feature log-odds contributions for one unit."""
    c = contrib[contrib != 0]
    order = list(c.abs().sort_values(ascending=False).index)
    steps = [(label(f), float(c[f])) for f in order[:top]]
    rest = float(c[order[top:]].sum())
    if order[top:]:
        steps.append((f"{len(order) - top} other features", rest))
    rows, level = [("Line average", 0.0, base_logit, "base", base_logit)], base_logit
    for name, v in steps:
        rows.append((name, level, level + v, "raises risk" if v > 0 else "lowers risk", v))
        level += v
    rows.append(("Prediction", 0.0, level, "result", level))
    df = pd.DataFrame(rows, columns=["feature", "x0", "x1", "effect", "value"])
    df["p_after"] = 1 / (1 + np.exp(-df.x1))
    colors = alt.Scale(domain=["base", "raises risk", "lowers risk", "result"],
                       range=["#8C8C8C", "#D1495B", "#2E933C", "#3D5A80"])
    bars = alt.Chart(df).mark_bar(cornerRadius=2).encode(
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("x0:Q", title="log-odds of failure (0 = 50 %)"),
        x2="x1:Q",
        color=alt.Color("effect:N", scale=colors, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("value:Q", title="contribution / level", format="+.3f"),
                 alt.Tooltip("p_after:Q", title="P(fail) after this step", format=".1%")],
    )
    thr_logit = float(np.log(thr / (1 - thr)))
    rule = alt.Chart(pd.DataFrame({"x": [thr_logit], "t": [f"action threshold ({thr:.1%})"]})).mark_rule(
        strokeDash=[4, 4], color="#E39B2D").encode(x="x:Q", tooltip=["t:N"])
    return (bars + rule).properties(height=32 * len(df) + 40)


def build_unit_row(base_row: dict, edits: dict) -> pd.DataFrame:
    """Raw-format one-row frame from a template unit plus the user's edits."""
    row = dict(base_row)
    row.update({k: v for k, v in edits.items() if k not in ("wip_hold_h", "anneal_to_wind_days")})
    wind = pd.Timestamp(row["wind_end_ts"])
    row["test_ts"] = wind + pd.Timedelta(hours=float(edits.get("wip_hold_h", 6.0)))
    row["anneal_date"] = wind.normalize() - pd.Timedelta(days=int(edits.get("anneal_to_wind_days", 2)))
    for k, v in row.items():
        if v is None:
            row[k] = np.nan
    return pd.DataFrame([row])
