# 🧪 Clinical Insights Assistant — GenAI POC  
**LangChain · LangGraph · LangSmith · OpenAI · Databricks**

This project is a **GenAI-enabled clinical insights assistant** that performs:

✅ Doctor feedback summarization  
✅ Cohort-level analysis  
✅ Issue detection (non-compliance + adverse events)  
✅ Dosage what-if scenario simulation  
✅ Full agentic reasoning using LangGraph  
✅ Real-time visual agent flow (ipywidgets + Plotly)

Designed and tested on **Databricks Community Edition**.

---

## 🚀 Features

### ✅ 1. Agentic Clinical Assistant
The system routes queries to specialized tools using a LangGraph-powered agent:

| Intent Keywords | Routed Tool |
|----------------|-------------|
| "adverse", "issues", "non-compliance" | issue_detection |
| "cohort", "compare", "difference" | cohort_comparison |
| "simulate", "dosage", "what-if", "increase" | scenario_simulation |
| Everything else | genai_summary |

---

## ✅ 2. Automatic Dosage Parsing
The assistant **parses dosage increases from user queries**:

Examples:
increase by 30 mg  
raise dose +25mg  
up by 10 mg  
add 5 mg

Parsed values feed directly into:
simulate_dosage_adjustment(df, delta_mg=<parsed_value>)

✅ Default is +25 mg  
✅ Supports mg, milligram, milligrams

---

## ✅ 3. Notebook UI (No Streamlit Required)
UI uses ipywidgets + Plotly:

 ✅ Dataset preview  
 ✅ Cohort summary  
 ✅ Query box  
 ✅ Flow diagram  
 ✅ Final results  

---

## ✅ 4. LangSmith Integrated

export LANGCHAIN_TRACING_V2=true  
export LANGCHAIN_API_KEY=<your-langsmith-key>  
export LANGCHAIN_PROJECT="clinical-insights-assistant"

---

## 📁 Project Structure
<pre>
clinical-insights-assistant/
│
├── data/
│   └── clinical_trial_data.csv
│
├── src/
│   ├── agent/
│   │   ├── agent_core.py
│   │   ├── memory.py
│   │   └── __init__.py
│   │
│   ├── ui/
│   │   ├── notebook_ui.ipynb
│   │   └── __init__.py
│   │
│   ├── data_loader.py
│   ├── issue_detection.py
│   ├── cohort_analysis.py
│   ├── scenario_simulation.py
│   ├── genai_interface.py
│   └── __init__.py
│
├── requirements.txt
└── README.md
</pre>

---

## ✅ Installation (Databricks)

%pip install -r requirements.txt  
dbutils.library.restartPython()

---

## ✅ Running the UI

Open:
src/ui/notebook_ui.ipynb

---

## ✅ Running the Agent

from src.agent.agent_core import run_agent  
run_agent("simulate dosage increase by 35 mg")

---

## ✅ Example Queries

flag adverse events  
compare cohorts  
simulate dosage +15mg  
summarize doctor feedback  