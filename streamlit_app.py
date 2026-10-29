"""
Toroidal transformer EOL failure-risk app (RuleFit model).

Run from the Code folder:
    streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Toroid EOL risk", page_icon=":material/electric_bolt:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/unit_risk.py", title="Unit risk check", icon=":material/fact_check:", default=True),
        st.Page("app_pages/batch_scoring.py", title="Batch scoring", icon=":material/table_chart:"),
        st.Page("app_pages/model_card.py", title="Model & rules", icon=":material/rule:"),
    ],
    position="top",
)

st.title(page.title, icon=page.icon)
page.run()

st.caption(
    "IT5514 Applied Machine Learning · RuleFit (Friedman & Popescu, 2008) trained on "
    "MES / test-station records · academic prototype"
)
