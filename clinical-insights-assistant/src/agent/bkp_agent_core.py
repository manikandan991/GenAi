import os
import re
import pandas as pd
from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.data_loader import load_data
from src.issue_detection import flag_non_compliance, extract_adverse_events
from src.cohort_analysis import cohort_summary, compare_cohorts
from src.scenario_simulation import simulate_dosage_adjustment
from src.genai_interface import get_llm, summarize_doctor_feedback, regulatory_style_summary
from .memory import AgentState

# --- simple, robust parser for "increase by +X mg" or "+Xmg" etc. ---
_INC_WORDS = r"(increase|increased|inc|raise|raised|bump|bumped|up|add|added)"
_MG_UNIT   = r"(?:mg|milligram(?:s)?)"

def _parse_delta_mg(text: str) -> Optional[float]:
    """
    Parse dosage increase (delta in mg) from free text.
    Returns a positive float if found; None if not found.

    Matches examples:
      "increase by 25 mg", "increase 25mg", "raise 10 mg", "add 5 mg",
      "up by 15mg", "+25 mg", "+25mg"
    """
    if not text:
        return None
    t = text.lower()

    # 1) Verb-led patterns: "increase by 25 mg", "raise 10mg", "add 5 mg", "up by 15 mg"
    m1 = re.search(
        rf"(?:{_INC_WORDS})\s*(?:by\s*)?(?P<num>[-+]?\d+(?:\.\d+)?)\s*{_MG_UNIT}?",
        t,
    )
    if m1:
        try:
            val = float(m1.group("num"))
            if val > 0:
                return val
        except ValueError:
            pass

    # 2) Signed number patterns: "+25 mg", "+25mg"
    m2 = re.search(
        rf"(?P<sign>[+])\s*(?P<num>\d+(?:\.\d+)?)\s*{_MG_UNIT}",
        t,
    )
    if m2:
        try:
            val = float(m2.group("num"))
            if val > 0:
                return val
        except ValueError:
            pass

    # If you want to support "increase dose 25" without unit, uncomment:
    # m3 = re.search(rf"(?:{_INC_WORDS})\s*(?:by\s*)?(?P<num>\d+(?:\.\d+)?)\b", t)
    # if m3:
    #     val = float(m3.group("num"))
    #     if val > 0:
    #         return val

    return None


def _route(state: AgentState) -> Literal["issues","cohorts","simulate","summarize"]:
    q = state.query.lower()
    if any(k in q for k in ["non-compliance","noncompliance","adverse","side effect","flag issues","issues","alerts"]):
        return "issues"
    if any(k in q for k in ["cohort","compare","outcome comparison","difference"]):
        return "cohorts"
    if any(k in q for k in ["what-if","simulate","dosage","dose"]):
        return "simulate"
    # If the user mentions an increase explicitly, nudge to simulate
    if _parse_delta_mg(q):
        return "simulate"
    return "summarize"


# --- Nodes: return partial dicts so LangGraph merges into AgentState safely ---
def node_issues(state: AgentState):
    df = load_data()
    nonc = flag_non_compliance(df)
    aev = extract_adverse_events(df)
    msg = f"Non-compliance rows: {len(nonc):,}\nAdverse-related rows: {len(aev):,}"
    notes = aev['doctor_notes'].dropna().astype(str).tolist()[:50]
    return {
        "selected_tool": "issue_detection",
        "result_text": msg,
        "notes_sample": notes,
    }

def node_cohorts(state: AgentState):
    df = load_data()
    table = cohort_summary(df).to_markdown(index=False)
    msg = "Cohort comparison complete.\n" + table
    return {
        "selected_tool": "cohort_comparison",
        "result_text": msg,
    }

def node_simulate(state: AgentState):
    df = load_data()
    # Use parsed delta from state if present; else default to 25 mg
    delta = state.delta_mg if state.delta_mg is not None else 25.0
    # Sanity clamp (optional): 0.1 mg to 500 mg
    try:
        delta = float(delta)
    except Exception:
        delta = 25.0
    delta = max(0.1, min(delta, 500.0))

    sim = simulate_dosage_adjustment(df, delta_mg=delta)
    table = (
        sim.groupby('cohort')['projected_outcome']
        .mean().round(2).to_frame()
        .rename(columns={'projected_outcome': 'avg_projected_outcome'})
        .to_markdown()
    )
    msg = f"Dosage +{delta} mg projection (avg_projected_outcome by cohort):\n" + table
    return {
        "selected_tool": "scenario_simulation",
        "result_text": msg,
        "delta_mg": delta,
    }

def node_summarize(state: AgentState):
    df = load_data()
    notes = df['doctor_notes'].dropna().astype(str).tolist()[:50]
    llm = get_llm()
    issues = summarize_doctor_feedback(llm, notes, max_notes=50)
    cohort_table = cohort_summary(df).to_markdown(index=False)
    reg = regulatory_style_summary(llm, cohort_table, issues)
    return {
        "selected_tool": "genai_summary",
        "result_text": issues,
        "summary": reg,
        "notes_sample": notes,
    }

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("issues", node_issues)
    graph.add_node("cohorts", node_cohorts)
    graph.add_node("simulate", node_simulate)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("summarize")
    graph.add_conditional_edges("summarize", _route, {
        "issues": "issues",
        "cohorts": "cohorts",
        "simulate": "simulate",
        "summarize": END,
    })
    graph.add_edge("issues", END)
    graph.add_edge("cohorts", END)
    graph.add_edge("simulate", END)

    memory = MemorySaver()  # in-memory checkpointing; swap with Redis/SQLite in prod
    app = graph.compile(checkpointer=memory)
    return app

def run_agent(query: str) -> dict:
    # Parse dosage delta from the user prompt and add to the initial state
    delta = _parse_delta_mg(query)
    app = build_agent()
    state = {"query": query, "delta_mg": delta}
    cfg = {"configurable": {"thread_id": "poc-session-1", "checkpoint_ns": "default"}}
    out = app.invoke(state, config=cfg)
    return out

if __name__ == "__main__":
    print(run_agent("simulate what-if: increase by 35 mg for cohort A next month"))
