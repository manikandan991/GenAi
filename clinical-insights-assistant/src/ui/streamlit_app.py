# /Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant/src/ui/streamlit_app.py

import os
import pathlib
import streamlit as st
from src.agent.agent_core import run_agent
from src.data_loader import load_data
from src.cohort_analysis import cohort_summary

# --- Resolve a sensible default DATA_PATH on Databricks ---
PROJECT_DIR = "/Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant"
DEFAULT_DATA = f"{PROJECT_DIR}/data/clinical_trial_data.csv"
os.environ.setdefault("DATA_PATH", DEFAULT_DATA)

st.set_page_config(page_title="Clinical Insights Assistant", layout="wide")
st.title("🧪 Clinical Insights Assistant (GenAI POC)")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("DATA_PATH", os.getenv("DATA_PATH", DEFAULT_DATA))
    if st.button("Reload Data"):
        st.session_state["df"] = load_data(data_path)
    st.caption("Tip: set OPENAI_API_KEY as an env var (cluster-level or in notebook).")

# --- Load data safely ---
try:
    df = st.session_state.get("df", load_data(data_path))
except Exception as e:
    st.error(f"Failed to load data from {data_path}\n\n{e}")
    df = None

st.subheader("Dataset Preview")
if df is not None:
    st.dataframe(df.head(20))
else:
    st.info("No data loaded.")

st.subheader("Cohort Snapshot")
if df is not None:
    st.dataframe(cohort_summary(df))

st.markdown("---")
st.subheader("Ask the Assistant")
query = st.text_input("Your request", "Summarize doctor feedback and flag issues for cohort A last 7 days")

if st.button("Run"):
    with st.spinner("Thinking..."):
        try:
            result = run_agent(query)  # run_agent should embed the LangGraph config/thread_id
            st.success("Done")
            st.write("**Selected Tool:**", result.get("selected_tool"))
            st.write("**Result Text:**")
            st.code(result.get("result_text") or "", language="markdown")
            if result.get("summary"):
                st.write("**Regulatory-style Summary:**")
                st.write(result["summary"])
        except Exception as e:
            st.error(f"Agent error:\n\n{e}")
