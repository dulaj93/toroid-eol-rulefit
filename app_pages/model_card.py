import altair as alt
import pandas as pd
import streamlit as st

from app_common import load_model, pretty_rule, label

model, fb, meta = load_model()
m = meta["test_metrics"]["RuleFit"]

with st.container(horizontal=True):
    st.metric("Test ROC-AUC", f"{m['ROC-AUC']:.3f}", border=True)
    st.metric("Test PR-AUC", f"{m['PR-AUC']:.3f}", "base rate 0.069", delta_color="off", border=True)
    st.metric("Rules in the model", meta["n_terms"], border=True)
    st.metric("Recall at threshold", f"{m['Recall@thr']:.0%}", f"threshold {meta['threshold']:.1%}", delta_color="off", border=True)

st.caption(f"Trained on units wound {meta['train_period'][0]} – {meta['train_period'][1]}; "
           "evaluated on Jul–Aug 2026, which the model never saw.")

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Comparison with course models", icon=":material/leaderboard:")
    comp = pd.DataFrame(meta["test_metrics"]).T[["ROC-AUC", "PR-AUC", "Brier"]]
    comp["Model size"] = [f"{meta['n_terms']} rules", "206 coefficients", "600 trees", "215 trees"][:len(comp)]
    st.dataframe(comp, column_config={
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Brier": st.column_config.NumberColumn("Brier (lower = better)", format="%.3f")})
    st.subheader("Global feature importance", icon=":material/bar_chart:")
    fi = model.feature_importance()
    fi = fi[fi > 0].head(15).rename(index=label).reset_index()
    fi.columns = ["feature", "importance"]
    st.altair_chart(alt.Chart(fi).mark_bar(color="#3D5A80").encode(
        x=alt.X("importance:Q", title="Friedman–Popescu importance"),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240))))

with right:
    st.subheader("The complete model", icon=":material/rule:")
    st.caption(f"Intercept {model.intercept_:+.3f} log-odds (≈ {1 / (1 + 2.718281828 ** -model.intercept_):.0%} risk when "
               "no rule applies). Every unit's risk is the intercept plus the coefficients of the rules it satisfies.")
    rules = model.get_rules()
    kind = st.segmented_control("Show", ["All", "Raise risk", "Lower risk"], default="All", required=True)
    if kind == "Raise risk":
        rules = rules[rules.coef > 0]
    elif kind == "Lower risk":
        rules = rules[rules.coef < 0]
    rules = rules.assign(Rule=rules.term.map(pretty_rule))
    st.dataframe(rules[["Rule", "coef", "odds_ratio", "support", "importance"]], hide_index=True, height=560,
                 column_config={
                     "Rule": st.column_config.TextColumn(width="large"),
                     "coef": st.column_config.NumberColumn("Coefficient", format="%+.3f"),
                     "odds_ratio": st.column_config.NumberColumn("Odds ratio", format="%.2f"),
                     "support": st.column_config.ProgressColumn("Support", min_value=0, max_value=1, format="percent"),
                     "importance": st.column_config.NumberColumn("Importance", format="%.3f")})

with st.expander("How RuleFit works", icon=":material/school:"):
    st.markdown("""
1. **Rule generation:** about 400 shallow trees (depth ≤ 3) are grown on random halves of the data, each
   correcting the previous ones. Every tree node becomes a candidate IF–THEN rule (~1,500 candidates).
2. **Sparse selection:** an L1-penalised (lasso) logistic regression over all candidate rules keeps only the
   few that matter. The rest get exactly zero weight.
3. **Prediction:** `logit P(fail) = intercept + Σ coefficient × [rule is satisfied]`. The whole model is
   the table above, so every prediction can be traced to the rules behind it.

Most rules describe **safe operating windows** (negative coefficients), e.g. tension close to recipe and humidity
below about 77 %. A unit outside those windows loses their protection, so its failure risk rises.
""")
