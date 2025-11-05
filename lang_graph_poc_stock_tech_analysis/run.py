# Databricks notebook source
# MAGIC %pip install -r /Workspace/Shared/lang_graph_poc_stock_tech_analysis/requirements.txt
# MAGIC

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
os.environ["GEMINI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_KEY"]="xxxxxxxxxxxxxxxxxxxxxxxxxxx"


# COMMAND ----------

import os

os.makedirs("/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/charts", exist_ok=True)
os.makedirs("/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/chroma", exist_ok=True)
os.makedirs("/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/reports", exist_ok=True)


# COMMAND ----------

# MAGIC %run
# MAGIC /Workspace/Shared/lang_graph_poc_stock_tech_analysis/notebooks/01_build_kb

# COMMAND ----------

# MAGIC %run
# MAGIC /Workspace/Shared/lang_graph_poc_stock_tech_analysis/notebooks/02_rag_agent_notebook_single_cell

# COMMAND ----------

# MAGIC %run
# MAGIC /Workspace/Shared/lang_graph_poc_stock_tech_analysis/notebooks/03_ragas_eval_single_cell

# COMMAND ----------

# Databricks notebook source
# CALLER: parse ticker → run child → show /tmp report + chart (CE-safe)

import os, re, json, traceback, requests
from PIL import Image
import matplotlib.pyplot as plt

CHILD_PATH   = "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/notebooks/02_rag_agent_notebook_single_cell"
OPENAI_MODEL_PRIMARY   = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-4o-mini")
OPENAI_MODEL_FALLBACK  = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o")

# ---------- OpenAI (optional) ----------
def extract_ticker_with_openai(question: str) -> str:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return ""
    prompt = f"""
Extract exactly ONE yfinance-compatible symbol (global stock/ETF/INDEX) from the question.
Return ONLY strict compact JSON like: {{"ticker":"AAPL"}} or {{"ticker":"^GSPC"}} or {{"ticker":"0700.HK"}} or {{"ticker":"7203.T"}} or {{"ticker":"RELIANCE.NS"}}.
If none, return {{"ticker":""}}.

Question: {question}
""".strip()
    def _call(model: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Output only strict compact JSON. No extra text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        j = r.json()
        return (j.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
    text = ""
    try:
        text = _call(OPENAI_MODEL_PRIMARY)
        if not text and OPENAI_MODEL_FALLBACK:
            text = _call(OPENAI_MODEL_FALLBACK)
    except Exception as e:
        print("[OpenAI REST warn]", e)
    if not text:
        return ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    return (d.get("ticker","") or "").upper().strip()

# ---------- Heuristic fallback ----------
COMMON_MAP = {
    "APPLE": "AAPL","AAPL":"AAPL","MICROSOFT":"MSFT","MSFT":"MSFT","NVIDIA":"NVDA","NVDA":"NVDA",
    "ALPHABET":"GOOGL","GOOGL":"GOOGL","GOOG":"GOOG","AMAZON":"AMZN","AMZN":"AMZN","META":"META",
    "TESLA":"TSLA","TSLA":"TSLA","SP500":"^GSPC","S&P500":"^GSPC","S&P 500":"^GSPC","^GSPC":"^GSPC",
    "NASDAQ":"^IXIC","^IXIC":"^IXIC","DOW":"^DJI","DOW JONES":"^DJI","^DJI":"^DJI",
    "SPY":"SPY","QQQ":"QQQ","DIA":"DIA","IWM":"IWM",
    "INFOSYS":"INFY.NS","INFY":"INFY.NS","TCS":"TCS.NS","TATA CONSULTANCY":"TCS.NS","RELIANCE":"RELIANCE.NS",
    "NIFTY":"^NSEI","NIFTY 50":"^NSEI","^NSEI":"^NSEI","BANKNIFTY":"^NSEBANK","^NSEBANK":"^NSEBANK",
    "FTSE":"^FTSE","^FTSE":"^FTSE","DAX":"^GDAXI","^GDAXI":"^GDAXI","CAC":"^FCHI","^FCHI":"^FCHI",
    "NIKKEI":"^N225","^N225":"^N225","TOYOTA":"7203.T","7203":"7203.T","SONY":"6758.T",
    "TENCENT":"0700.HK","700":"0700.HK","0700":"0700.HK","SHOPIFY":"SHOP.TO","SHOP.TO":"SHOP.TO",
    "IBIT":"IBIT","FBTC":"FBTC",
}
STOPWORDS = set("CHECK ANALYZE ANALYSIS OUTLOOK FOR NEXT MONTH MONTHS WEEK WEEKS YEAR YEARS TODAY TOMORROW ENTRY ENTRIES STOP TARGET TP SL PRICE PRICES BUY SELL SHORT LONG TECHNICAL FUNDAMENTAL SENTIMENT RANGE SIDEWAYS BREAKOUT BREAK DIPS PULLBACK PULL BACK IS THE A AN OF AND OR TO WITH PLEASE GIVE SHOW ME DO ON ABOUT".split())
YF_PATTERN = re.compile(r"^[A-Z0-9\-\.^=]{1,20}(\.[A-Z0-9]{1,5})?$")
def _norm(s: str) -> str: return re.sub(r"[^A-Z0-9\-\.^=]", "", s.upper().strip())
def heuristic_ticker(q: str) -> str:
    qU = q.upper()
    for k, v in COMMON_MAP.items():
        if k in qU: return v
    for tok in re.findall(r"[A-Za-z0-9\-\.^=]+", qU):
        if tok.upper() in STOPWORDS: continue
        t = _norm(tok)
        if t and YF_PATTERN.match(t) and t not in ("CHECK","BUY","SELL","LONG","SHORT","ENTRY","STOP"):
            return t
    return ""
def resolve_ticker(q: str) -> str:
    return extract_ticker_with_openai(q) or heuristic_ticker(q)

# ---------- Display helpers (CE-safe: use /tmp paths only) ----------
def _display_report_local(local_md_path: str):
    if not local_md_path or not os.path.exists(local_md_path):
        print(f"(Report markdown not found at: {local_md_path})")
        return
    with open(local_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    displayHTML(
        f'''
        <div style="font-family:ui-sans-serif;line-height:1.6;max-width:900px">
            <h3>Latest report.md</h3>
            <pre style="white-space:pre-wrap">{md_text}</pre>
        </div>
        '''
    )

def _display_chart_local(local_png_path: str):
    if not local_png_path or not os.path.exists(local_png_path):
        print(f"(Chart not found at: {local_png_path})")
        return
    img = Image.open(local_png_path)
    plt.imshow(img); plt.axis("off"); plt.show()

# ---------- Run child and show local artifacts ----------
def call_child_and_show(ticker: str):
    print(f"→ Resolved ticker: {ticker}")
    try:
        raw = dbutils.notebook.run(
            path=CHILD_PATH,
            timeout_seconds=0,
            arguments={"ticker": ticker, "AS_CHILD": "1"}
        )
    except Exception:
        print("Child invocation failed:")
        traceback.print_exc()
        return

    try:
        result = json.loads(raw)
    except Exception:
        print("Child returned non-JSON output:\n", raw)
        return

    if result.get("status") != "ok":
        print("Child reported error:")
        print(json.dumps(result, indent=2))
        return

    print("\n=== Summary from child ===")
    print(json.dumps({
        "ticker": result.get("ticker"),
        "trend": result.get("trend"),
        "signals": result.get("signals", []),
        "report_path_local": result.get("report_path_local"),
        "chart_image_local": result.get("chart_image_local")
    }, indent=2))

    # CE-safe: use ONLY the local (/tmp) paths returned by child
    _display_report_local(result.get("report_path"))
    _display_chart_local(result.get("chart_image"))

# ---------- Simple REPL ----------
print("Type a question about a stock/index/ETF (e.g., 'check S&P 500', 'analyze 7203.T', 'Tencent HK').")
print("Type 'quit'/'exit'/'q' to leave.")
while True:
    try:
        q = input("Ask> ").strip()
    except EOFError:
        break
    if not q:
        continue
    if q.lower() in ("quit", "exit", "q"):
        print("Goodbye 👋")
        break

    ticker = resolve_ticker(q)
    if not ticker:
        print("Sorry, I couldn't figure out a ticker. Try including the symbol (e.g., 0700.HK, 7203.T, AAPL).")
        continue
    if not re.fullmatch(r"[A-Z0-9\-\.^=]{1,20}(\.[A-Z0-9]{1,5})?", ticker):
        print(f"Resolved invalid-looking ticker: {ticker}")
        continue

    call_child_and_show(ticker)
