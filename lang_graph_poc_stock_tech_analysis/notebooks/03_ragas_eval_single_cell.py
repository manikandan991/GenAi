# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 03 — RAGAS Evaluation for Technical Analysis Output  
# MAGIC ### _Automated Quality Check for Model-Generated Stock Analysis_
# MAGIC
# MAGIC This notebook cell uses **RAGAS-style LLM evaluation** to verify whether the generated **Technical Analysis (TA) report** meets all required structural and informational criteria.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Purpose
# MAGIC
# MAGIC Every generated TA report must contain certain mandatory sections.  
# MAGIC This evaluator checks **completeness**, not correctness, and returns only:
# MAGIC
# MAGIC - `"pass"` ✅ — if **all** required components are present  
# MAGIC - `"fail"` ❌ — if **any** component is missing
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ What the evaluator checks
# MAGIC
# MAGIC The model must include all **six** of the following elements:
# MAGIC
# MAGIC 1. **Trend** — clear statement (bullish / bearish / neutral)  
# MAGIC 2. **Entry** — actionable technical entry plan  
# MAGIC 3. **Stop-loss** — where to place a risk control level  
# MAGIC 4. **6-month scenario/outlook** — directional expectation  
# MAGIC 5. **News sentiment** — may include headlines or “None detected”  
# MAGIC 6. **Disclaimer** — must acknowledge this is not investment advice  
# MAGIC
# MAGIC If **any** of these are missing → result is `"fail"`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Evaluation Prompt Template
# MAGIC
# MAGIC The evaluator uses the following instruction to RAGAS / OpenAI:
# MAGIC
# MAGIC ```text
# MAGIC Return 'pass' only if ALL are true:
# MAGIC 1) trend
# MAGIC 2) entry
# MAGIC 3) stop-loss
# MAGIC 4) 6-month scenario
# MAGIC 5) news sentiment (can say 'None detected')
# MAGIC 6) disclaimer
# MAGIC
# MAGIC Question: {question}
# MAGIC Expected Answer: Complete TA plan.
# MAGIC Model Response: {response}
# MAGIC
# MAGIC Return 'pass' or 'fail'.
# MAGIC Evaluation:
# MAGIC

# COMMAND ----------

# Databricks notebook source
# SINGLE-CELL: Validate latest report with widgets + clean return

import os, json, re, traceback
from pathlib import Path
from datetime import datetime
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def layman_verdict(text: str) -> str:
    """Use OpenAI to convert technical metrics into a simple layman's verdict."""
    prompt = f"""
Explain this technical indicator meaning in one simple layman sentence:
{text}
Return only the sentence.
"""
    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL_PRIMARY", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(Verdict unavailable: {e})"


# ------------ Widgets ------------
dbutils.widgets.text("reports_dir", "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/reports")
dbutils.widgets.text("report_file", "")  # optional specific .md or prefix
dbutils.widgets.dropdown("use_ragas", "yes", ["no","yes"])
dbutils.widgets.text("AS_CHILD", "0")   # "1" when called by parent
REPORTS_DIR = dbutils.widgets.get("reports_dir").strip()
REPORT_FILE = dbutils.widgets.get("report_file").strip()
USE_RAGAS   = dbutils.widgets.get("use_ragas") == "yes"
AS_CHILD    = dbutils.widgets.get("AS_CHILD") == "1"

# ------------ Return helper ------------
def return_json(payload: dict):
    txt = json.dumps(payload, indent=None if AS_CHILD else 2)
    print(txt)
    if AS_CHILD:
        dbutils.notebook.exit(json.dumps(payload))

# ------------ Local validator ------------
REQUIRED_SECTIONS = [
    "## trend","## signals","## entry","## stop",
    "## 6-month outlook","## news sentiment","## disclaimer"
]

def local_validate(markdown_text: str) -> dict:
    t = (markdown_text or "").lower()
    missing = [h for h in REQUIRED_SECTIONS if h not in t]
    if missing:
        return {"value":"fail","reason":f"Missing sections: {', '.join(missing)}"}
    if not any(tok in t for tok in ["rsi","sma","macd"]):
        return {"value":"fail","reason":"No common TA tokens (RSI/SMA/MACD) detected"}
    return {"value":"pass","reason":"All required sections present and TA tokens found"}

# ------------ Optional RAGAS try ------------
def try_ragas_validate(question: str, response_text: str):
    try:
        from ragas.metrics import DiscreteMetric
    except Exception as e:
        return {"value": None, "reason": f"ragas not available: {e}"}

    class RagasFunctionLLM:
        class _Resp:
            def __init__(self, value: str, reason: str = ""):
                self.value = value; self.reason = reason
            def model_dump(self): return {"value": self.value, "reason": self.reason}
            def dict(self): return self.model_dump()
        def generate(self, prompt_input, response_model=None):
            res = local_validate(response_text)
            return RagasFunctionLLM._Resp(value=res["value"], reason=res["reason"])

    metric = DiscreteMetric(
        name="ta_report_checks",
        prompt=(
            "Return 'pass' only if ALL are true:\n"
            "1) trend, 2) entry, 3) stop-loss, 4) 6-month scenario, 5) news sentiment (can say 'None detected'), 6) disclaimer.\n\n"
            "Question: {question}\nExpected Answer: Complete TA plan.\nModel Response: {response}\n"
            "Return 'pass' or 'fail'.\nEvaluation:"
        ),
        allowed_values=["pass","fail"],
    )

    # Prefer sync
    try:
        if hasattr(metric, "score"):
            score = metric.score(
                question=question,
                expected_answer="Complete TA plan",
                response=response_text,
                llm=RagasFunctionLLM(),
            )
        else:
            import asyncio, nest_asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    nest_asyncio.apply()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            async def _run():
                return await metric.ascore(
                    question=question,
                    expected_answer="Complete TA plan",
                    response=response_text,
                    llm=RagasFunctionLLM(),
                )
            score = loop.run_until_complete(_run())
        return {"value": getattr(score, "value", None), "reason": getattr(score, "reason", "")}
    except Exception as e:
        return {"value": None, "reason": f"ragas error: {e}"}

# ------------ Find latest (or specific) report ------------
try:
    p = Path(REPORTS_DIR)
    if not p.exists():
        raise FileNotFoundError(f"Reports dir not found: {REPORTS_DIR}")

    if REPORT_FILE:
        cand = p / REPORT_FILE
        if not cand.exists():
            matches = sorted(p.glob(f"{REPORT_FILE}*.md"))
            if not matches:
                raise FileNotFoundError(f"No report matching {REPORT_FILE} in {REPORTS_DIR}")
            report_path = matches[-1]
        else:
            report_path = cand
    else:
        mds = sorted(p.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not mds:
            raise FileNotFoundError(f"No .md files in {REPORTS_DIR}")
        report_path = mds[0]

    text = report_path.read_text(encoding="utf-8")
    m = re.match(r"([A-Za-z0-9\.\-_]+)_\d{8}_\d{6}\.md$", report_path.name)
    ticker = m.group(1) if m else "UNKNOWN"

    local_res = local_validate(text)
    value, reason = local_res["value"], local_res["reason"]
    evaluator = "LOCAL"

    if USE_RAGAS:
        r = try_ragas_validate(f"Create a TA plan for {ticker}", text)
        if r["value"] is None:
            evaluator = f"LOCAL (ragas skipped: {r['reason']})"
        else:
            evaluator = "LOCAL+RAGAS"
            # Both must pass to pass
            if value == "pass" and r["value"] == "pass":
                value, reason = "pass", "Local+RAGAS passed"
            else:
                value = "fail"
                reason = f"Local={local_res['value']}, RAGAS={r['value']}; {local_res['reason']} | {r['reason']}"

    result = {
        "status": "ok",
        "report_path": str(report_path),
        "ticker": ticker,
        "metric": "ta_report_checks",
        "value": value,
        "reason": reason,
        "evaluator": evaluator,
        "reports_dir": REPORTS_DIR,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return_json(result)

except Exception as e:
    tb = traceback.format_exc()
    err = {"status": "error", "error": str(e), "traceback": tb}
    return_json(err)
