import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import load_model, load_units, score_frame, risk_band, TEST_START
from toroid_preprocess import clean, first_attempts

model, fb, meta = load_model()
thr = meta["threshold"]

st.caption("Score a whole export from the MES / test-station database (same column format as the raw "
           "dataset). Without an upload, the July–August 2026 test period is used.")

upload = st.file_uploader("Upload a raw export (CSV)", type="csv")
if upload is not None:
    try:
        df = clean(pd.read_csv(upload, low_memory=False, dtype={"ir_Mohm": str}))
        if "test_attempt" in df:
            df = first_attempts(df)
        source = f"{upload.name}: {len(df):,} units"
    except Exception as e:  # malformed or incomplete export
        st.error(f"Could not read this file as a raw export: {e}", icon=":material/error:")
        st.stop()
else:
    fa = load_units()
    df = fa[fa.wind_end_ts >= TEST_START].reset_index(drop=True)
    source = f"Test period Jul–Aug 2026: {len(df):,} units (unseen during training)"

try:
    df = df.assign(risk=score_frame(df))
except KeyError as e:
    st.error(f"The export is missing a required column: {e}", icon=":material/error:")
    st.stop()

df["action"] = [risk_band(p, thr)[0] for p in df.risk]
has_truth = "result" in df and df["result"].notna().any()
flag = df.risk >= thr

st.subheader(source, icon=":material/table_chart:")
with st.container(horizontal=True):
    st.metric("Units scored", f"{len(df):,}", border=True)
    st.metric("Flagged for in-process check", f"{flag.sum():,}", f"{flag.mean():.0%} of units", delta_color="off", border=True)
    st.metric("Hold for review (≥ 20 %)", f"{(df.risk >= 0.2).sum():,}", border=True)
    st.metric("Expected failures", f"{df.risk.sum():.0f}", help="Sum of predicted probabilities.", border=True)
    if has_truth:
        y = df.result.eq("FAIL")
        st.metric("Actual failures", f"{y.sum():,}", border=True)
        st.metric("Failures caught by flag", f"{(y & flag).sum() / max(y.sum(), 1):.0%}",
                  f"precision {(y & flag).sum() / max(flag.sum(), 1):.0%}", delta_color="off", border=True)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Risk distribution**")
    hist = alt.Chart(df).mark_bar(color="#3D5A80").encode(
        x=alt.X("risk:Q", bin=alt.Bin(maxbins=40), title="predicted failure risk", axis=alt.Axis(format="%")),
        y=alt.Y("count():Q", title="units"))
    rule = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color="#E39B2D", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(hist + rule)
with right:
    st.markdown("**Mean predicted risk by winding machine**" + (" vs observed fail rate" if has_truth else ""))
    g = df.groupby("winding_machine").agg(predicted=("risk", "mean"))
    if has_truth:
        g["observed"] = df.result.eq("FAIL").groupby(df.winding_machine).mean()
    long = g.reset_index().melt("winding_machine", var_name="series", value_name="rate")
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("winding_machine:N", title=None),
        xOffset=alt.XOffset("series:N", sort=["predicted", "observed"]),
        y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("series:N", scale=alt.Scale(domain=["predicted", "observed"], range=["#3D5A80", "#D1495B"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["winding_machine", "series", alt.Tooltip("rate:Q", format=".1%")])
    st.altair_chart(bars)

cols = ["serial_no", "product_code", "winding_machine", "shift", "wind_end_ts", "risk", "action"]
if has_truth:
    cols.append("fail_code")
table = df.sort_values("risk", ascending=False)[cols]
st.dataframe(table, hide_index=True, column_config={
    "serial_no": "Serial", "product_code": "Product", "winding_machine": "Machine", "shift": "Shift",
    "wind_end_ts": st.column_config.DatetimeColumn("Wound", format="YYYY-MM-DD HH:mm"),
    "risk": st.column_config.ProgressColumn("Failure risk", min_value=0, max_value=1, format="percent"),
    "action": "Recommended action",
    "fail_code": "Actual EOL result",
})
st.download_button("Download scored units (CSV)", table.to_csv(index=False).encode(), file_name="scored_units.csv",
                   mime="text/csv", icon=":material/download:")
