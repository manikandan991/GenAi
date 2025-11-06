# Stock Technical Analysis Agent – Databricks Orchestration

This project provides an automated **technical analysis pipeline** on Databricks using:

- ✅ Natural language questions (e.g., *“Check <stock name> tech analysis”*)  
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
Check index/eft/stock for technical analysis
```

### ✅ Step 2 — Caller Notebook
- Reads your question  
- Uses **OpenAI** to extract the best ticker (`NSE`)  
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

## Output

# Technical Analysis Report — ^NSEI
Generated: 2025-11-05T20:15:02

![Chart](/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/charts/^NSEI_20251105_201502.png)
!(https://github.com/manikandan991/GenAi/blob/main/lang_graph_poc_stock_tech_analysis/artifacts/charts/%5ENSEI_20251105_195524.png)

---

## Analyst Narrative
## Trend
bullish

## Signals
MACD bearish crossover
RSI: 52.76 | SMA50: 25179.07 | SMA200: 24352.83

## Entry
If bullish: enter on pullbacks near SMA20–SMA50 followed by breakout of last swing high.
If bearish: avoid longs until price closes above SMA50 with improving MACD. (Fallback template)

## Stop
Below recent swing low or ~1.5× ATR(14).

## 6-Month Outlook
Bull: continuation if trend holds.
Base: sideways range.
Bear: lower highs persist unless structure improves.

## News Sentiment
Positive: None detected
Negative: None detected

## Disclaimer
Educational technical analysis only — not investment advice.

---

## Layman Verdicts
- **Trend:** Overall direction looks steady.
- **RSI:** RSI suggests momentum is normal; not too high or too low.
- **SMA50:** The 50-day average shows the recent price trend.
- **SMA200:** The 200-day average shows the long-term trend.
- **Signals:** No major buy/sell signals unless noted.

---

## Technical Details Found
- **Trend:** bullish
- **RSI(14):** 52.76
- **SMA50:** 25179.07
- **SMA200:** 24352.83
- **MACD vs Signal:** 194.96 vs 217.69
- **Bollinger Band Width:** 0.0504
- **Signals Detected:**
- MACD bearish crossover

### Recent Data (last 5 rows)
| Date | Open | High | Low | Close | Volume | SMA50 | SMA200 | RSI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-29T00:00:00+05:30 | 25982.00 | 26097.85 | 25960.30 | 26053.90 | 321900 | 25119.68 | 24304.13 | 72.43 |
| 2025-10-30T00:00:00+05:30 | 25984.40 | 26032.05 | 25845.25 | 25877.85 | 257400 | 25139.70 | 24315.88 | 64.14 |
| 2025-10-31T00:00:00+05:30 | 25863.80 | 25953.75 | 25711.20 | 25722.10 | 334400 | 25154.53 | 24327.34 | 57.84 |
| 2025-11-03T00:00:00+05:30 | 25696.85 | 25803.10 | 25645.50 | 25763.35 | 23300 | 25168.79 | 24340.72 | 58.99 |
| 2025-11-04T00:00:00+05:30 | 25744.75 | 25787.40 | 25578.40 | 25597.65 | 305100 | 25179.07 | 24352.83 | 52.76 |

---

## News Sentiment (titles only)
- **Positive:** None
- **Negative:** None

---

**Sources**
- KB: embedded via Chroma
- Chart image: /Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/charts/^NSEI_20251105_201502.png

*This is educational technical analysis, not investment advice.*

