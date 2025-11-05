# agents/react_agent_web.py
from __future__ import annotations
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import StructuredTool  # << use StructuredTool

from tools.web_fundamentals_tool import (
    web_fundamentals_scan_structured,
    PROMPT_GUIDELINES,
)

def build_tools():
    return [
        StructuredTool.from_function(
            name="web_fundamentals_scan",
            description=(
                "Legally gather and summarize public fundamentals for a company using web search. "
                "Args: company (str), symbol (optional str, e.g., 'TCS.NS', 'AAPL'). "
                "Returns JSON with metrics (5Y Revenue/PAT CAGR, D/E, Interest Coverage, CFO vs PAT), "
                "evidence snippets, and source URLs. DO-NOT rules are enforced."
            ),
            func=web_fundamentals_scan_structured,
        )
    ]

SYSTEM_PROMPT = f"""
You are an equity research assistant specialized in fundamental analysis using only public web information.

Follow these priorities and strictly enforce the DO-NOT list:
- Growth: 5Y Revenue & PAT CAGR
- Margin stability (qualitative if only text hints are available)
- Leverage: Debt/Equity
- Interest coverage
- Cash conversion: CFO vs PAT
- Optional: ROE/ROCE

DO-NOT LIST (MUST OBEY):
{PROMPT_GUIDELINES}

Instructions:
- Always call the tool `web_fundamentals_scan` FIRST with arguments: company, and optional symbol.
- Summarize metrics in bullet points; cite 2–4 source URLs (from tool output).
- If values are missing or conflicting, say so and include both sources.
- Finish with a 1-line verdict (Strong / Decent / Weak) and a 1-line caveat if needed.
"""

def _prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
            # 👇 Required by some LangChain versions for tool-calling agent
            MessagesPlaceholder("agent_scratchpad", optional=True),
        ]
    )

def build_agent(llm_model: str = "gpt-4o-mini") -> AgentExecutor:
    llm = ChatOpenAI(model=llm_model, temperature=0)
    tools = build_tools()
    prompt = _prompt()
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Memory wrapper
_SESSION_STORE: Dict[str, ChatMessageHistory] = {}

def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

def build_agent_with_memory(llm_model: str = "gpt-4o-mini") -> RunnableWithMessageHistory:
    base = build_agent(llm_model)
    return RunnableWithMessageHistory(
        base,
        get_session_history=_get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

if __name__ == "__main__":
    agent = build_agent_with_memory()
    cfg = {"configurable": {"session_id": "web-funda-demo-1"}}
    print(agent.invoke(
        {"input": "Analyze fundamentals for Tata Consultancy Services (company='Tata Consultancy Services', symbol='TCS.NS')."},
        config=cfg
    )["output"])
