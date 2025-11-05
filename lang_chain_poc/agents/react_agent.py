# agents/react_agent.py
from __future__ import annotations
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools.render import render_text_description

# Optional web search (safe to remove if you don't need it)
from langchain_community.tools.tavily_search import TavilySearchResults

# ✅ Alpha Vantage tools (you created these in tools/alpha_tools.py)
from tools.alpha_tools import alpha_fetch_statements, analyze_alpha_fundamentals


# -------------------------
# Tools
# -------------------------
def build_tools() -> List[Tool]:
    tools: List[Tool] = []

    # (Optional) Web search for context/news
    try:
        tavily = TavilySearchResults(max_results=5)

        def _web_search(q: str) -> str:
            try:
                results = tavily.invoke({"query": q})
                return "\n\n".join([r.get("content", "") for r in results if r.get("content")])
            except Exception as e:
                return f"[web_search error] {e}"

        tools.append(
            Tool(
                name="web_search",
                description="Search the web for recent facts/news on a company or sector.",
                func=_web_search,
            )
        )
    except Exception as _:
        # Tavily not installed or blocked in the environment — just skip it.
        pass

    # Alpha Vantage: fetch raw statements
    tools.append(
        Tool(
            name="alpha_fetch_statements",
            description=(
                "Fetch quarterly and annual Income Statement, Balance Sheet, and Cash Flow from "
                "Alpha Vantage for the given symbol (e.g., 'TCS.NS', 'INFY.NS', 'RELIANCE.NS'). "
                "Returns JSON."
            ),
            func=alpha_fetch_statements,
        )
    )

    # Alpha Vantage: compute metrics and verdict
    tools.append(
        Tool(
            name="analyze_alpha_fundamentals",
            description=(
                "Analyze fundamentals using Alpha Vantage data: 5Y Revenue CAGR, 5Y PAT CAGR, "
                "Debt/Equity, Interest coverage, Cash conversion (CFO vs PAT). "
                "Input: symbol (e.g., 'TCS.NS'). Returns JSON with metrics and verdict."
            ),
            func=analyze_alpha_fundamentals,
        )
    )

    return tools


# -------------------------
# Prompt (ReAct)
# -------------------------
def _prompt(tools: List[Tool]) -> ChatPromptTemplate:
    tools_md = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])

    # NOTE:
    # - history is OPTIONAL so first call works without passing any history.
    # - agent_scratchpad MUST be a MessagesPlaceholder (ReAct will fill it with tool traces).
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a meticulous equity research analyst for Indian equities. "
                    "Use tools when helpful.\n\n"
                    "Priorities (cite numbers you compute):\n"
                    "  1) Growth: 5Y Revenue & PAT CAGR\n"
                    "  2) Margin stability (discuss if visible)\n"
                    "  3) Leverage: Debt/Equity\n"
                    "  4) Interest coverage\n"
                    "  5) Cash conversion: CFO vs PAT\n\n"
                    "Keep answers concise with bullet points and end with a one-line verdict. "
                    "Prefer Alpha Vantage tool outputs over assumptions.\n\n"
                    "Available tools:\n{tools}\n\nTool names: {tool_names}"
                ),
                MessagesPlaceholder("history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        .partial(tools=tools_md, tool_names=tool_names)
    )


# -------------------------
# Builders
# -------------------------
def build_react_agent(llm_model: str = "gpt-4o-mini") -> AgentExecutor:
    """
    Stateless ReAct agent. Use this if you don't need chat history between calls.
    """
    llm = ChatOpenAI(model=llm_model, temperature=0)
    tools = build_tools()
    prompt = _prompt(tools)
    agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # handle_parsing_errors=True prevents 'Invalid Format: Missing Action:' crashes
    return AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


# -------------------------
# Memory wrapper
# -------------------------
_SESSION_STORE: Dict[str, ChatMessageHistory] = {}


def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_react_agent_with_memory(llm_model: str = "gpt-4o-mini") -> RunnableWithMessageHistory:
    """
    Stateful ReAct agent with message history.
    Call with config containing a session_id:
      cfg = {"configurable": {"session_id": "finance-demo-1"}}
      agent.invoke({"input": "Analyze TCS.NS fundamentals"}, config=cfg)
    """
    base = build_react_agent(llm_model=llm_model)
    return RunnableWithMessageHistory(
        base,
        get_session_history=_get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
