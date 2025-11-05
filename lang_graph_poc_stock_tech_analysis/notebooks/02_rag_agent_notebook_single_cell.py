# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 02 — RAG Agent (Technical Analysis Engine)
# MAGIC ### _Single-Cell Implementation (Child Notebook for Stock/Index/ETF Analysis)_
# MAGIC
# MAGIC Fetches 1-year OHLC data
# MAGIC
# MAGIC Calculates SMA50, SMA200, RSI, MACD, Bollinger Width
# MAGIC
# MAGIC Detects MACD crossovers and RSI extremes
# MAGIC
# MAGIC Generates a clean technical trend classification (“bullish / bearish / neutral”)
# MAGIC This notebook acts as a **self-contained technical-analysis agent** that can be invoked from another Databricks notebook via:
# MAGIC
# MAGIC ```python
# MAGIC dbutils.notebook.run(
# MAGIC     "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/notebooks/02_rag_agent_notebook_single_cell",
# MAGIC     timeout_seconds=0,
# MAGIC     arguments={"ticker": "AAPL", "AS_CHILD": "1"}
# MAGIC )
# MAGIC

# COMMAND ----------

# MAGIC %pip install "pydantic>=2.7,<3" "chromadb>=0.5.4" "langchain>=0.2.14" "langchain-chroma>=0.1.2" "langchain-huggingface>=0.1.2" "sentence-transformers>=2.6.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Databricks notebook source
# CHILD: 02_rag_agent_notebook_single_cell — generates TA, saves report+chart to Workspace AND /tmp, returns JSON

# --- Safe dependency bootstrap (cluster-friendly) ---
import os, sys, re, json, traceback, subprocess, importlib
from datetime import datetime
from typing import Any, Dict, List, Optional

def ensure(mod_name: str, pip_name: Optional[str] = None, install: bool = True):
    try:
        importlib.import_module(mod_name)
        return
    except Exception:
        if not install:
            raise ImportError(f"Missing module: {mod_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])
            importlib.invalidate_caches()
            importlib.import_module(mod_name)
        except Exception as e:
            raise ImportError(f"Failed to install {pip_name or mod_name}: {e}")

# ------------ Widgets ------------
dbutils.widgets.text("ticker", "TCS.NS")
dbutils.widgets.text("AS_CHILD", "0")
TICKER_PARAM = dbutils.widgets.get("ticker").strip()
AS_CHILD = dbutils.widgets.get("AS_CHILD").strip() == "1"

def return_json(payload: dict):
    txt = json.dumps(payload, indent=None if AS_CHILD else 2)
    print(txt)
    if AS_CHILD:
        dbutils.notebook.exit(json.dumps(payload))

def _resolve_ticker(initial: str) -> str:
    v = (initial or "").strip()
    if not v:
        try:
            v = dbutils.widgets.getArgument("ticker")
        except Exception:
            pass
    if not v:
        v = (os.getenv("TICKER", "") or os.getenv("QUESTION", "")).strip()
    if not v:
        v = "TCS.NS"
    return v

TICKER_PARAM = _resolve_ticker(TICKER_PARAM)

# Ensure deps
missing_err = None
try:
    ensure("yfinance"); ensure("matplotlib"); ensure("ta")
    ensure("vaderSentiment", "vaderSentiment")
    ensure("chromadb"); ensure("langchain_huggingface", "langchain-huggingface")
    ensure("langchain_chroma", "langchain-chroma"); ensure("sentence_transformers", "sentence-transformers")
    ensure("requests")
except Exception as e:
    missing_err = str(e)

print("===========================")
print("TICKER_PARAM:", TICKER_PARAM)
print("AS_CHILD:", AS_CHILD)
print("===========================")

if missing_err:
    return_json({
        "status": "error",
        "error": "Missing or failed Python dependency",
        "detail": missing_err,
        "hint": "Install on the cluster with %pip install <package> and restart the Python process."
    })

# --- Imports AFTER deps are ensured ---
import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import requests

# ------------ Paths ------------
PROJECT_ROOT = "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/"
CHROMA_DIR   = f"{PROJECT_ROOT}artifacts/chroma"
CHARTS_DIR   = f"{PROJECT_ROOT}artifacts/charts"
REPORTS_DIR  = f"{PROJECT_ROOT}artifacts/reports"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Local export root for Community Edition (ALLOWED)
LOCAL_EXPORT_ROOT = "/tmp/lang_graph_stock_ta"
LOCAL_CHARTS_DIR  = f"{LOCAL_EXPORT_ROOT}/charts"
LOCAL_REPORTS_DIR = f"{LOCAL_EXPORT_ROOT}/reports"
os.makedirs(LOCAL_CHARTS_DIR, exist_ok=True)
os.makedirs(LOCAL_REPORTS_DIR, exist_ok=True)

# ------------ Optional LLMs (for your draft writer; optional) ------------
try:
    import google.generativeai as genai
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    genai = None

try:
    from litellm import completion
except Exception:
    completion = None

analyzer = SentimentIntensityAnalyzer()

def llm_chat(messages: List[Dict[str, str]], model_preference: str = "gemini") -> str:
    if model_preference == "gemini" and genai and os.getenv("GEMINI_API_KEY"):
        try:
            joined = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(joined)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            print("Gemini failed:", e)
    if completion:
        try:
            resp = completion(model=os.getenv("FALLBACK_LLM", "ollama/llama3.1"), messages=messages)
            return resp.choices[0].message["content"]
        except Exception as e:
            print("LiteLLM fallback failed:", e)
    return ""

# ------------ Layman verdicts via OpenAI (REST) ------------
def layman_verdict(text: str) -> str:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model_primary = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-4o-mini")
    model_fallback = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o")

    if not api_key:
        t = text.lower()
        if "rsi" in t: return "RSI suggests momentum is normal; not too high or too low."
        if "sma50" in t: return "The 50-day average shows the recent price trend."
        if "sma200" in t: return "The 200-day average shows the long-term trend."
        if "trend is" in t: return "Overall direction looks steady."
        if "signals" in t: return "No major buy/sell signals unless noted."
        return "No simple verdict available."

    prompt = f"Explain in one very simple layman sentence: {text}\nReturn only the sentence."
    def _call(model: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only a single short sentence, no extra text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        j = r.json()
        return (j.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
    try:
        sent = _call(model_primary)
        if not sent and model_fallback:
            sent = _call(model_fallback)
        return sent or "No simple verdict available."
    except Exception as e:
        return f"(Verdict unavailable: {e})"

# ------------ Retriever ------------
COLLECTION = "stock_kb"
def get_retriever(chroma_dir: str = CHROMA_DIR):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_dir)
    db = Chroma(client=client, collection_name=COLLECTION, embedding_function=embed)
    return db.as_retriever(search_kwargs={"k": 4})
retriever = get_retriever()

# ------------ TA utilities ------------
def fetch_ohlc(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA50"] = ta.trend.sma_indicator(out["Close"], window=50)
    out["SMA200"] = ta.trend.sma_indicator(out["Close"], window=200)
    out["RSI"] = ta.momentum.rsi(out["Close"], window=14)
    out["MACD"] = ta.trend.macd(out["Close"], window_slow=26, window_fast=12)
    out["MACD_SIGNAL"] = ta.trend.macd_signal(out["Close"])
    bb_h = ta.volatility.bollinger_hband(out["Close"])
    bb_l = ta.volatility.bollinger_lband(out["Close"])
    out["BB_WIDTH"] = (bb_h - bb_l) / out["Close"]
    return out

def plot_chart(df: pd.DataFrame, ticker: str) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close")
    if "SMA50" in df:  ax.plot(df.index, df["SMA50"], label="SMA50")
    if "SMA200" in df: ax.plot(df.index, df["SMA200"], label="SMA200")
    ax.legend(); ax.set_title(f"{ticker} — Technical Chart")
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ticker}_{ts}.png"

    # Save to Workspace (original) AND /tmp (local)
    ws_path   = f"{CHARTS_DIR}/{fname}"
    local_path = f"{LOCAL_CHARTS_DIR}/{fname}"
    plt.savefig(ws_path)
    plt.savefig(local_path)
    plt.close(fig)
    return {"workspace": ws_path, "local": local_path}

def analyze_ta(ticker: str) -> Dict[str, Any]:
    df = compute_indicators(fetch_ohlc(ticker))
    latest = df.dropna().iloc[-1]
    prev   = df.dropna().iloc[-2]
    trend = "bullish" if latest["SMA50"] > latest["SMA200"] else "bearish" if latest["SMA50"] < latest["SMA200"] else "neutral"
    signals = []
    if prev["MACD"] < prev["MACD_SIGNAL"] and latest["MACD"] > latest["MACD_SIGNAL"]: signals.append("MACD bullish crossover")
    if prev["MACD"] > prev["MACD_SIGNAL"] and latest["MACD"] < latest["MACD_SIGNAL"]: signals.append("MACD bearish crossover")
    if latest["RSI"] < 35: signals.append("RSI oversold")
    if latest["RSI"] > 65: signals.append("RSI overbought")

    chart_paths = plot_chart(df, ticker)  # dict with workspace & local

    return {
        "ticker": ticker,
        "sma50": float(latest["SMA50"]),
        "sma200": float(latest["SMA200"]),
        "rsi": float(latest["RSI"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACD_SIGNAL"]),
        "bb_width": float(latest["BB_WIDTH"]),
        "trend": trend,
        "signals": signals,
        "chart_path": chart_paths["workspace"],
        "chart_path_local": chart_paths["local"],
        "df_tail": df.tail(5).reset_index().to_dict(orient="records"),
    }

def fetch_news_sentiment(ticker: str, limit: int = 10) -> Dict[str, Any]:
    news = yf.Ticker(ticker).news or []
    rows = []
    for n in news[:limit]:
        title = n.get("title", ""); url = n.get("link", ""); ts = n.get("providerPublishTime")
        dt = datetime.utcfromtimestamp(ts) if ts else None
        s = analyzer.polarity_scores(title)["compound"]
        rows.append({"title": title, "url": url, "published": dt, "compound": s})
    if not rows:
        return {"positive": [], "negative": [], "all": []}
    df = pd.DataFrame(rows).sort_values("published", ascending=False)
    pos = df[df["compound"] > 0.25][:5].to_dict(orient="records")
    neg = df[df["compound"] < -0.25][:5].to_dict(orient="records")
    return {"positive": pos, "negative": neg, "all": df.to_dict(orient="records")}

# ------------ Report writer (Workspace + /tmp) ------------
def _fmt(x, nd=2):
    try: return f"{float(x):.{nd}f}"
    except Exception: return str(x)

def _format_signals(sig_list):
    if not sig_list: return "None detected"
    return "\n".join([f"- {s}" for s in sig_list])

def _format_df_tail(df_tail_records: List[Dict[str, Any]]) -> str:
    if not df_tail_records: return "_No recent rows available_"
    rows = []
    for r in df_tail_records:
        rr = {}
        for k, v in r.items():
            if hasattr(v, "isoformat"): rr[k] = v.isoformat()
            else: rr[k] = v
        rows.append(rr)
    cols_pref = ["Date", "Datetime", "Open", "High", "Low", "Close", "Volume", "SMA50", "SMA200", "RSI"]
    all_cols = list(rows[0].keys())
    cols = [c for c in cols_pref if c in all_cols] or all_cols[:8]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for r in rows:
        line = []
        for c in cols:
            val = r.get(c, "")
            if isinstance(val, float): line.append(_fmt(val))
            else: line.append(str(val))
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)

def save_markdown_report_dual(
    ticker: str,
    analyst_text: str,
    ta: Dict[str, Any],
    layman: Dict[str, str],
    chart_path_ws: str,
    news: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ticker}_{ts}.md"
    ws_path    = f"{REPORTS_DIR}/{fname}"
    local_path = f"{LOCAL_REPORTS_DIR}/{fname}"

    # Technical details
    trend      = ta.get("trend", "unknown")
    rsi        = _fmt(ta.get("rsi"))
    sma50      = _fmt(ta.get("sma50"))
    sma200     = _fmt(ta.get("sma200"))
    macd       = _fmt(ta.get("macd"))
    macd_sig   = _fmt(ta.get("macd_signal"))
    bb_width   = _fmt(ta.get("bb_width"), nd=4)
    signals_md = _format_signals(ta.get("signals", []))
    tail_md    = _format_df_tail(ta.get("df_tail", []))

    # Layman verdicts
    lay_trend   = layman.get("trend", "")
    lay_rsi     = layman.get("rsi", "")
    lay_sma50   = layman.get("sma50", "")
    lay_sma200  = layman.get("sma200", "")
    lay_signals = layman.get("signals", "")

    news_pos = "; ".join([n.get("title","") for n in (news or {}).get("positive", [])]) or "None"
    news_neg = "; ".join([n.get("title","") for n in (news or {}).get("negative", [])]) or "None"

    md = f"""# Technical Analysis Report — {ticker}
Generated: {datetime.now().isoformat(timespec='seconds')}

![Chart]({chart_path_ws})

---

## Analyst Narrative
{analyst_text}

---

## Layman Verdicts
- **Trend:** {lay_trend}
- **RSI:** {lay_rsi}
- **SMA50:** {lay_sma50}
- **SMA200:** {lay_sma200}
- **Signals:** {lay_signals}

---

## Technical Details Found
- **Trend:** {trend}
- **RSI(14):** {rsi}
- **SMA50:** {sma50}
- **SMA200:** {sma200}
- **MACD vs Signal:** {macd} vs {macd_sig}
- **Bollinger Band Width:** {bb_width}
- **Signals Detected:**
{signals_md}

### Recent Data (last 5 rows)
{tail_md}

---

## News Sentiment (titles only)
- **Positive:** {news_pos}
- **Negative:** {news_neg}

---

**Sources**
- KB: embedded via Chroma
- Chart image: {chart_path_ws}

*This is educational technical analysis, not investment advice.*
"""
    # Write BOTH files
    for path in (ws_path, local_path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            print("Warning: could not write", path, "->", e)

    return {"workspace": ws_path, "local": local_path}

# ------------ Agent ------------
class AgentState(BaseModel):
    query: str
    ticker: str
    kb_context: str = ""
    ta: Optional[Dict[str, Any]] = None
    news: Optional[Dict[str, Any]] = None
    draft: str = ""
    report_path: Optional[Dict[str, str]] = None

def extract_ticker_from_query(q: str) -> str:
    tok = re.sub(r"[^A-Za-z0-9\-\.^=]", "", q).upper().strip()
    return tok if 0 < len(tok) <= 20 else q.strip().upper()

def retrieve_docs(retriever, query: str):
    if hasattr(retriever, "invoke"): return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"): return retriever.get_relevant_documents(query)
    if hasattr(retriever, "_get_relevant_documents"): return retriever._get_relevant_documents(query)
    return []

def build_graph():
    def node_plan(state: AgentState) -> AgentState:
        if not state.ticker:
            state.ticker = extract_ticker_from_query(state.query)
        return state

    def node_retrieve_kb(state: AgentState) -> AgentState:
        q = f"Technical analysis checklist for {state.ticker}. Indicators, patterns, rules."
        docs = retrieve_docs(retriever, q)
        state.kb_context = "\n\n".join([getattr(d, "page_content", "") for d in docs]) if docs else ""
        return state

    def node_tools(state: AgentState) -> AgentState:
        state.ta = analyze_ta(state.ticker)
        state.news = fetch_news_sentiment(state.ticker)
        return state

    def node_compose(state: AgentState) -> AgentState:
        sys_msg = (
            "You are a cautious equities technical analyst. Use ONLY provided facts and KB. "
            "Write concise, structured analysis with exact sections: "
            "## Trend ## Signals ## Entry ## Stop ## 6-Month Outlook ## News Sentiment ## Disclaimer. No advice."
        )
        facts = {
            "ticker": state.ticker,
            "ta": state.ta or {},
            "news_positive": (state.news or {}).get("positive", []),
            "news_negative": (state.news or {}).get("negative", []),
            "chart_path": (state.ta or {}).get("chart_path"),
        }
        user_prompt = (
            "KB:\n{kb}\n\nFACTS:\n{facts}\n\n"
            "Write the analysis now with the exact required sections."
        ).format(kb=state.kb_context, facts=json.dumps(facts, default=str))

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt},
        ]
        draft = llm_chat(messages).strip()

        ta_ = state.ta or {}
        trend = ta_.get("trend", "unknown")
        sigs = ", ".join(ta_.get("signals", [])) or "—"
        def fmt(x): 
            try: return f"{float(x):.2f}"
            except: return str(x)
        required = ["## trend","## signals","## entry","## stop","## 6-month outlook","## news sentiment","## disclaimer"]
        missing = not draft or any(h not in draft.lower() for h in required)
        if missing:
            news_pos = (state.news or {}).get("positive", [])
            news_neg = (state.news or {}).get("negative", [])
            pos_str = "; ".join([n.get("title","") for n in news_pos[:3]]) or "None detected"
            neg_str = "; ".join([n.get("title","") for n in news_neg[:3]]) or "None detected"
            draft = f"""
## Trend
{trend}

## Signals
{sigs}
RSI: {fmt(ta_.get("rsi"))} | SMA50: {fmt(ta_.get("sma50"))} | SMA200: {fmt(ta_.get("sma200"))}

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
Positive: {pos_str}
Negative: {neg_str}

## Disclaimer
Educational technical analysis only — not investment advice.
""".strip()

        state.draft = draft
        return state

    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("retrieve_kb", node_retrieve_kb)
    g.add_node("tools", node_tools)
    g.add_node("compose", node_compose)
    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve_kb")
    g.add_edge("retrieve_kb", "tools")
    g.add_edge("tools", "compose")
    g.add_edge("compose", END)
    return g.compile()

def _to_dict(x):
    try:
        if hasattr(x, "model_dump"): return x.model_dump()
        if hasattr(x, "dict"): return x.dict()
        return dict(x) if isinstance(x, dict) else json.loads(json.dumps(x, default=str))
    except Exception:
        return {"_raw": str(x)}

try:
    from dbruntime.dbutils import NotebookExit
except Exception:
    class NotebookExit(Exception): pass

try:
    graph = build_graph()
    out = graph.invoke(AgentState(query=TICKER_PARAM, ticker=""))
    state = _to_dict(out)
    ta   = state.get("ta", {}) or {}
    news = state.get("news", {}) or {}

    # Layman verdicts
    layman = {
        "rsi":     layman_verdict(f"RSI value is {ta.get('rsi')}"),
        "sma50":   layman_verdict(f"SMA50 value is {ta.get('sma50')}"),
        "sma200":  layman_verdict(f"SMA200 value is {ta.get('sma200')}"),
        "trend":   layman_verdict(f"Trend is {ta.get('trend')}"),
        "signals": layman_verdict(f"Signals are: {ta.get('signals', [])}"),
    }

    # Save report BOTH to Workspace and /tmp
    dual_paths = save_markdown_report_dual(
        ticker=state.get("ticker", ""),
        analyst_text=state.get("draft", ""),
        ta=ta,
        layman=layman,
        chart_path_ws=ta.get("chart_path", ""),
        news=news,
    )

    payload = {
        "status": "ok",
        "ticker": state.get("ticker", ""),
        "analysis": state.get("draft", ""),
        "signals": ta.get("signals", []),
        "trend": ta.get("trend"),
        "chart_image": ta.get("chart_path"),          # Workspace path
        "chart_image_local": ta.get("chart_path_local"),  # /tmp path (CE-safe)
        "news_pos": news.get("positive", []),
        "news_neg": news.get("negative", []),
        "report_path": dual_paths["workspace"],       # Workspace path
        "report_path_local": dual_paths["local"],     # /tmp path (CE-safe)
        "chroma_dir": CHROMA_DIR,
        "charts_dir": CHARTS_DIR,
        "reports_dir": REPORTS_DIR,
        "local_charts_dir": LOCAL_CHARTS_DIR,
        "local_reports_dir": LOCAL_REPORTS_DIR,
        "layman": layman,
    }
    return_json(payload)

except NotebookExit:
    raise
except Exception as e:
    tb = traceback.format_exc()
    err = {"status": "error", "error": str(e), "traceback": tb}
    return_json(err)
