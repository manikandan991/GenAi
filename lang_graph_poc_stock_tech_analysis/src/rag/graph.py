import json
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from langgraph.graph import StateGraph, END

from .vectorstore_init import build_kb_index
from .ta_tools import analyze_ta, TAResult
from .news_tools import fetch_news_sentiment
from .llm import llm_chat

class AgentState(BaseModel):
    query: str
    ticker: str
    kb_context: str = ""
    ta: Optional[TAResult] = None
    news: Optional[Dict[str, Any]] = None
    draft: str = ""

def extract_ticker_from_query(q: str) -> str:
    tok = ''.join([c for c in q if c.isalnum() or c=='.']).upper()
    return tok if 0 < len(tok) <= 12 else q.strip().upper()

def build_graph(kb_dir="./src/rag/kb", chroma_dir="./artifacts/chroma"):
    vectorstore = build_kb_index(kb_dir=kb_dir, chroma_dir=chroma_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def node_plan(state: AgentState) -> AgentState:
        if not state.ticker:
            state.ticker = extract_ticker_from_query(state.query)
        return state

    def node_retrieve_kb(state: AgentState) -> AgentState:
        q = f"Technical analysis checklist for {state.ticker}. Indicators, patterns, rules."
        docs = retriever.get_relevant_documents(q)
        state.kb_context = "\n\n".join([d.page_content for d in docs])
        return state

    def node_tools(state: AgentState) -> AgentState:
        state.ta = analyze_ta(state.ticker, charts_dir="./artifacts/charts")
        state.news = fetch_news_sentiment(state.ticker)
        return state

    def node_compose(state: AgentState) -> AgentState:
        sys = (
            "You are a cautious equities technical analyst. Use ONLY provided facts and KB. "
            "Include: trend, key signals, a candidate entry with justification, stop, 6‑month bull/base/bear scenario "
            "and a brief news sentiment note. Do not give investment advice; include a risk disclaimer."
        )
        facts = {
            "ticker": state.ticker,
            "ta": state.ta.__dict__ if state.ta else {},
            "news_positive": state.news.get("positive", []) if state.news else [],
            "news_negative": state.news.get("negative", []) if state.news else [],
            "chart_path": state.ta.chart_path if state.ta else None,
        }
        messages = [
            {"role":"system","content":sys},
            {"role":"user","content": f"KB:\n{state.kb_context}\n\nFACTS (JSON):\n{json.dumps(facts, default=str)}\n\nWrite the analysis now."}
        ]
        state.draft = llm_chat(messages)
        return state

    builder = StateGraph(AgentState)
    builder.add_node("plan", node_plan)
    builder.add_node("retrieve_kb", node_retrieve_kb)
    builder.add_node("tools", node_tools)
    builder.add_node("compose", node_compose)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "retrieve_kb")
    builder.add_edge("retrieve_kb", "tools")
    builder.add_edge("tools", "compose")
    builder.add_edge("compose", END)
    return builder.compile()
