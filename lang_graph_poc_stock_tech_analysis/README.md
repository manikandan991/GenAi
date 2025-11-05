# Stock Technical Analysis Agent – Databricks Orchestration

This project provides an automated **technical analysis pipeline** on Databricks using:

- ✅ Natural language questions (e.g., *“Check Infosys tech analysis”*)  
- ✅ **OpenAI** to extract the correct stock ticker  
- ✅ A **child notebook agent** (LangGraph + TA + sentiment + chart generation)  
- ✅ A **caller notebook REPL** that loops, parses your queries, triggers the child, and displays the **latest report.md**

All generated reports are stored in:

```
/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/reports
```

## 1. Project Structure

```
/Workspace/Shared/lang_graph_poc_stock_tech_analysis/
│
├── notebooks/
│   ├── 02_rag_agent_notebook_single_cell        ← Child Notebook (TA Agent)
│   └── caller_notebook                          ← Main REPL caller
│
├── artifacts/
│   ├── chroma/                                   ← KB (Chroma)
│   ├── charts/                                   ← Generated PNG charts
│   └── reports/                                  ← Generated report.md files
│
└── README.md
```

## 2. How It Works

### ✅ Step 1 — User asks a question
Example:

```
Check Thangamayil for entry and stop
```

### ✅ Step 2 — Caller Notebook
- Reads your question  
- Uses **OpenAI** to extract the best ticker (`THANGAMAYIL.NS`)  
- Calls the child notebook.

### ✅ Step 3 — Child Notebook
Performs:
- Fetch price history  
- Compute indicators (SMA, RSI, MACD, Bollinger Bands)  
- Load knowledge base from Chroma  
- Fetch news sentiment  
- Generate chart + markdown report  
- Returns JSON  

### ✅ Step 4 — Caller Notebook
- Reads JSON output  
- Loads **latest .md report**  
- Renders it inside Databricks.

## 3. Setup Instructions

### ✅ Install Dependencies

```
%pip install yfinance ta vaderSentiment google-generativeai sentence-transformers langgraph chromadb langchain-huggingface langchain-chroma matplotlib
dbutils.library.restartPython()
```

If Chroma/Pydantic errors occur:

```
%pip install "pydantic>=2.7,<3"
dbutils.library.restartPython()
```

### ✅ Set OpenAI API Key

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL_PRIMARY=gpt-4o-mini
OPENAI_MODEL_FALLBACK=gpt-4o
```

## 4. Running the Application

Run the caller notebook and enter questions:

```
Ask> check infosys
Ask> analyze thangamayil for entry
Ask> quit
```

## 5. Child Notebook Requirements

- Accepts:
  - `ticker`
  - `AS_CHILD=1`
- Returns output via:

```
dbutils.notebook.exit(json.dumps(payload))
```
