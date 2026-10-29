import numpy as np
import pandas as pd
import streamlit as st

from app_common import (load_model, load_units, example_units, build_unit_row, contributions,
                        waterfall_chart, risk_band, pretty_rule, label)

model, fb, meta = load_model()
thr = meta["threshold"]
units = load_units()
examples = example_units()
line_rate = float(units.y_fail.mean())

# editable inputs: key -> (min, max, step, format)
NUM = {
    "coil_cert_loss_W_kg": (0.6, 1.3, 0.005, "%.3f"),
    "core_mass_g": (100.0, 10000.0, 10.0, "%.0f"),
    "core_coat_thickness_um": (50.0, 400.0, 5.0, "%.0f"),
    "anneal_peak_temp_C": (780.0, 880.0, 0.5, "%.1f"),
    "anneal_soak_min": (30.0, 180.0, 1.0, "%.0f"),
    "anneal_dewpoint_C": (-60.0, 0.0, 0.5, "%.1f"),
    "anneal_ref_ring_loss_W_kg": (0.7, 1.4, 0.005, "%.3f"),
    "operator_experience_months": (0.0, 200.0, 0.5, "%.1f"),
    "pri_tension_N": (1.0, 150.0, 0.1, "%.1f"),
    "sec_tension_N": (1.0, 250.0, 0.1, "%.1f"),
    "pri_wind_speed_rpm": (100.0, 1200.0, 5.0, "%.0f"),
    "sec_wind_speed_rpm": (100.0, 1000.0, 5.0, "%.0f"),
    "pri_turns_logged": (1.0, 5000.0, 1.0, "%.0f"),
    "sec_turns_logged": (1.0, 2000.0, 1.0, "%.0f"),
    "interwinding_layers": (1.0, 4.0, 1.0, "%.0f"),
    "tape_overlap_pct": (20.0, 70.0, 0.5, "%.1f"),
    "winding_area_RH_pct": (30.0, 100.0, 0.1, "%.1f"),
    "winding_area_temp_C": (18.0, 40.0, 0.1, "%.1f"),
    "wip_hold_h": (0.0, 200.0, 0.5, "%.1f"),
    "anneal_to_wind_days": (0.0, 30.0, 1.0, "%.0f"),
}
CAT = {
    "steel_supplier": sorted(units.steel_supplier.unique()),
    "core_deburred": ["Y", "N", "UNKNOWN"],
    "anneal_furnace": sorted(units.anneal_furnace.unique()),
    "winding_machine": sorted(units.winding_machine.unique()),
    "operator_id": sorted(units.operator_id.unique()),
    "shift": ["A", "B", "C"],
    "wire_supplier": sorted(units.wire_supplier.unique()),
}
FIXED = ["product_code", "rated_power_VA", "primary_config", "secondary_config", "frequency_Hz",
         "application_segment", "steel_grade", "steel_coil_lot", "core_OD_mm", "core_ID_mm", "core_height_mm",
         "core_insulation", "enamel_thermal_class", "pri_wire_dia_mm", "sec_wire_dia_mm",
         "pri_turns_target", "sec_turns_target", "interwinding_tape", "impregnation"]


def load_into_state(row: pd.Series):
    st.session_state.base_row = row.to_dict()
    for k in NUM:
        if k == "wip_hold_h":
            v = (row.test_ts - row.wind_end_ts).total_seconds() / 3600
        elif k == "anneal_to_wind_days":
            v = (row.wind_end_ts.normalize() - row.anneal_date).days
        else:
            v = row[k]
        st.session_state[f"in_{k}"] = None if pd.isna(v) else float(round(v, 3))
    for k in CAT:
        st.session_state[f"in_{k}"] = str(row[k])


def template_for(product: str) -> pd.Series:
    sub = units[units.product_code == product]
    ok = sub[sub.y_fail == 0]
    return (ok if len(ok) else sub).iloc[-1]


def on_source():
    if st.session_state.source == "Test-period unit":
        on_example()
    else:
        on_product()


def on_example():
    row = examples.loc[examples.label == st.session_state.example].iloc[0]
    load_into_state(units.loc[units.serial_no == row.serial_no].iloc[0])


def on_product():
    load_into_state(template_for(st.session_state.product))


# ---- state initialisation (one place)
st.session_state.setdefault("source", "Test-period unit")
st.session_state.setdefault("example", examples.label.iloc[0])
st.session_state.setdefault("product", "TT0300-230-2X24-A" if "TT0300-230-2X24-A" in set(units.product_code)
                            else sorted(units.product_code.unique())[0])
if "base_row" not in st.session_state:
    on_source()

st.caption("Predict whether a wound unit will fail its first end-of-line electrical test, and see why. "
           "Start from a recorded unit or a product template, then change the process values.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    st.segmented_control("Start from", ["Test-period unit", "Product template"], key="source",
                         required=True, on_change=on_source)
    if st.session_state.source == "Test-period unit":
        st.selectbox("Unit (Jul–Aug 2026, unseen by the model)", examples.label, key="example",
                     on_change=on_example, width=520)
    else:
        st.selectbox("Product", sorted(units.product_code.unique()), key="product", on_change=on_product, width=320)

base = st.session_state.base_row
product = base["product_code"]


def num(k, **kw):
    lo, hi, step, fmt = NUM[k]
    return st.number_input(label(k), min_value=lo, max_value=hi, step=step, format=fmt, key=f"in_{k}", **kw)


def cat(k, segmented=False):
    if segmented:
        return st.segmented_control(label(k), CAT[k], key=f"in_{k}", required=True)
    return st.selectbox(label(k), CAT[k], key=f"in_{k}")


inputs, results = st.columns([1.05, 1], gap="large")

with inputs:
    ref = fb.ref_
    t_steel, t_anneal, t_wind, t_env = st.tabs(["Steel & core", "Anneal", "Winding", "Environment & timing"])
    with t_steel:
        c1, c2 = st.columns(2)
        with c1:
            cat("steel_supplier", segmented=True)
            num("coil_cert_loss_W_kg", help="Mill certificate core loss at 1.5 T, 50 Hz. Leave empty if not entered.")
            cat("core_deburred", segmented=True)
        with c2:
            num("core_mass_g")
            num("core_coat_thickness_um", disabled=base["core_insulation"] != "EPOXY_COAT",
                help="Only for epoxy-coated cores.")
            st.caption(f"Product median core mass: {ref['core_mass_g'].get(product, float('nan')):.0f} g")
    with t_anneal:
        c1, c2 = st.columns(2)
        with c1:
            cat("anneal_furnace", segmented=True)
            num("anneal_peak_temp_C")
            num("anneal_soak_min")
        with c2:
            num("anneal_dewpoint_C", help="N₂ atmosphere dew point; sensor installed Apr-2026 (empty = not recorded).")
            num("anneal_ref_ring_loss_W_kg", help="Batch reference-ring loss test (empty = batch not sampled).")
    with t_wind:
        c1, c2 = st.columns(2)
        with c1:
            cat("winding_machine")
            cat("operator_id")
            num("operator_experience_months")
            cat("shift", segmented=True)
            cat("wire_supplier")
        with c2:
            num("pri_tension_N")
            st.caption(f"Recipe median: {ref['pri_tension_N'].get(product, float('nan')):.1f} N · "
                       f"speed {ref['pri_wind_speed_rpm'].get(product, float('nan')):.0f} rpm")
            num("pri_wind_speed_rpm")
            num("sec_tension_N")
            num("sec_wind_speed_rpm")
            num("tape_overlap_pct")
            num("interwinding_layers")
    with t_env:
        c1, c2 = st.columns(2)
        with c1:
            num("winding_area_RH_pct")
            num("winding_area_temp_C")
        with c2:
            num("wip_hold_h")
            num("anneal_to_wind_days")
    with st.expander("Product design (fixed by product code)", icon=":material/settings_input_component:"):
        st.table(pd.DataFrame({"value": [str(base[k]) for k in FIXED]}, index=[label(k) for k in FIXED]))

# ---- prediction
edits = {k: st.session_state[f"in_{k}"] for k in list(NUM) + list(CAT)}
row = build_unit_row(base, edits)
X = fb.transform(row)
p = float(model.predict_proba(X)[0, 1])
contrib, base_logit = contributions(model, X)
action, color, icon = risk_band(p, thr)

with results:
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.metric("Predicted failure risk", f"{p:.1%}",
                      delta=f"{(p - line_rate) * 100:+.1f} pts vs line average ({line_rate:.1%})",
                      delta_color="inverse")
            with st.container(gap="small"):
                st.badge(action, color=color, icon=icon)
                if st.session_state.source == "Test-period unit":
                    actual = base["fail_code"]
                    st.badge(f"Actual EOL result: {actual}", color="red" if actual != "PASS" else "green",
                             icon=":material/history:")
        st.caption(f"Action threshold {thr:.1%} (cost-optimal for 10 : 1 miss vs check cost); "
                   f"≥ 20 % → hold for review.")
        def shown(v):
            if isinstance(v, (int, float, np.floating)):
                return "not recorded" if pd.isna(v) else f"{v:.4g}"
            return str(v)

        up = contrib[contrib > 0].sort_values(ascending=False).head(3)
        if len(up):
            st.markdown("**Main risk drivers:** " + "; ".join(
                f"{label(f)} = `{shown(X.iloc[0][f])}`" for f in up.index))

    st.subheader("Why this prediction", icon=":material/waterfall_chart:")
    st.caption("Each bar is a feature's additive contribution to the log-odds of failure, starting from the "
               "line average. RuleFit is additive in its rules, so this breakdown is exact.")
    st.altair_chart(waterfall_chart(contrib, base_logit, thr))

st.subheader("Rules for this unit", icon=":material/rule:")
act = model.term_activation(X)
act = act[act.type == "rule"].copy()
act["Rule"] = act.term.map(pretty_rule)
fired = act.value == 1
groups = {
    "Risk rules that fire": act[fired & (act.coef > 0)],
    "Safe windows met": act[fired & (act.coef < 0)],
    "Safe windows missed": act[~fired & (act.coef < 0)],
}
tabs = st.tabs([f"{k} ({len(v)})" for k, v in groups.items()])
cfg = {
    "Rule": st.column_config.TextColumn(width="large"),
    "odds_ratio": st.column_config.NumberColumn("Odds ratio when met", format="%.2f"),
    "support": st.column_config.ProgressColumn("Share of training units", min_value=0, max_value=1, format="percent"),
    "contribution": st.column_config.NumberColumn("Contribution (log-odds)", format="%+.3f"),
}
notes = {
    "Risk rules that fire": "Conditions that raise this unit's failure odds.",
    "Safe windows met": "Protective operating windows this unit is inside (each lowers the odds).",
    "Safe windows missed": "Protective windows this unit is outside of: these explain most of a high risk. "
                           "Bringing the process back inside them is the corrective action.",
}
for tab, (name, g) in zip(tabs, groups.items()):
    with tab:
        st.caption(notes[name])
        if g.empty:
            st.caption("None.")
        else:
            st.dataframe(g.sort_values("contribution", key=abs, ascending=False)[["Rule", "odds_ratio", "support", "contribution"]],
                         hide_index=True, column_config=cfg)
