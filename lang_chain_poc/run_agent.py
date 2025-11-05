# Databricks notebook source
# === CLEAN UNINSTALL SCRIPT ===
# Run this in a Databricks notebook cell before reinstalling pinned requirements.

pkgs = [
    "langchain", "langchain-core", "langchain-community", "langchain-openai",
    "langchain-text-splitters", "langchain-pinecone",
    "pinecone-client", "pinecone", "pinecone-plugin-interface", "pinecone-plugin-assistant",
    "openai", "tiktoken",
    "tavily-python", "google-search-results",
    "python-dotenv",
    "SQLAlchemy", "aiohttp", "aiohttp-retry", "aiohappyeyeballs",
    "httpx", "httpx-sse",
    "pydantic-settings", "dataclasses-json",
    "jsonpatch", "langsmith",
    "requests-toolbelt",
    "regex", "simsimd", "jiter", "orjson",
    "frozenlist", "yarl", "multidict", "aiosignal", "propcache","langchain-tavily"
]

for p in pkgs:
    print(f"Uninstalling {p} …")
    try:
        # -y = yes automatically; ignore errors avoids stopping on system packages
        get_ipython().system(f"pip uninstall -y {p}")
    except Exception as e:
        print(f"Could not uninstall {p}: {e}")

print("\n✅ Uninstall completed. Now run %restart_python before reinstalling.")


# COMMAND ----------

# MAGIC %restart_python
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Shared/lang_chain_poc/requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from dotenv import load_dotenv
load_dotenv("/Workspace/Shared/lang_chain_poc/.env")

# COMMAND ----------

from rag.ingest import ingest
ingest('/Workspace/Shared/lang_chain_poc/data/samples/kb_docs/')


# COMMAND ----------

# MAGIC %skip
# MAGIC # After you restore credits
# MAGIC
# MAGIC # Run a short cell/script to backfill:
# MAGIC from rag.ingest import backfill_backlog
# MAGIC backfill_backlog()
# MAGIC

# COMMAND ----------

# run_cli.py
from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

from rich.console import Console
from rich.panel import Panel

from config import PINECONE_INDEX_NAME

from chains.rag_chain import build_rag_chain
from chains.convo_chain import build_conversational_chain
from agents.react_agent import build_react_agent_with_memory


console = Console()


def _panel(title: str, body: str) -> Panel:
    return Panel(body if isinstance(body, str) else str(body), title=title)


def _as_text(res: Any) -> str:
    """Normalize various LCEL outputs to displayable text."""
    if res is None:
        return ""
    if isinstance(res, dict):
        # Prefer the common 'output' field for AgentExecutor
        if "output" in res:
            return str(res["output"])
        return str(res)
    content = getattr(res, "content", None)
    if content is not None:
        return str(content)
    return str(res)


def _safe_invoke(obj: Any, q: str, config: Dict | None = None) -> Any:
    """
    Be lenient about input shape:
    1) Try dict payload: {"input": q}
    2) If that fails with a type/prompt error, try plain string: q
    And the other way around, depending on first failure.
    """
    # Try dict first
    try:
        return obj.invoke({"input": q}, config=config) if config else obj.invoke({"input": q})
    except Exception as e1:
        # If it failed because the chain actually wants a raw string, try that
        try:
            return obj.invoke(q, config=config) if config else obj.invoke(q)
        except Exception as e2:
            # If both fail, bubble the original but include both messages
            raise RuntimeError(f"Invoke failed (dict and string). First: {e1}; Second: {e2}") from e2


def run_cli() -> None:
    """
    Commands:
      rag: <q>     → force RAG
      agent: <q>   → fundamentals agent (tools)
      chat: <q>    → general chat
      help         → list commands
      quit/exit    → leave

    Default route: RAG
    """
    # Build RAG
    rag = build_rag_chain(PINECONE_INDEX_NAME)

    # Build conversational chain (some code returns (memory, chain); support both)
    convo_chain = None
    try:
        maybe_tuple = build_conversational_chain()
        if isinstance(maybe_tuple, Tuple) and len(maybe_tuple) == 2:
            _, convo_chain = maybe_tuple
        else:
            convo_chain = maybe_tuple
    except Exception as e:
        console.print(_panel("Startup Warning", f"Could not load convo chain: {e}"))

    # Build agent with memory
    agent = None
    try:
        agent = build_react_agent_with_memory()
    except Exception as e:
        console.print(_panel("Startup Warning", f"Could not load agent: {e}"))

    # Stable session IDs for stateful chains
    chat_session_id = "chat-cli-session"
    agent_session_id = "agent-cli-session"

    console.print(
        Panel(
            "LangChain POC — type 'quit' to exit.\n\n"
            "Commands:\n"
            "  rag: <q>     → ask the knowledge base\n"
            "  agent: <q>   → fundamentals agent (tools)\n"
            "  chat: <q>    → general small talk\n\n"
            "No prefix → defaults to RAG",
            title="LCEL + RAG + Agent",
        )
    )

    while True:
        try:
            raw = input("> ").strip()
            if not raw:
                continue

            low = raw.lower()
            if low in {"quit", "exit"}:
                break
            if low in {"help", "commands"}:
                console.print(
                    _panel(
                        "Help",
                        "rag: <q>   → Force RAG\n"
                        "agent: <q> → Fundamentals Agent\n"
                        "chat: <q>  → Chat\n"
                        "quit/exit  → Exit\n\n"
                        "Default (no prefix) → RAG",
                    )
                )
                continue

            # ----- explicit routes -----
            if raw.startswith("agent:"):
                if agent is None:
                    console.print(_panel("Agent", "Agent not available."))
                    continue
                q = raw.split(":", 1)[1].strip()
                res = _safe_invoke(
                    agent,
                    q,
                    config={"configurable": {"session_id": agent_session_id}},
                )
                console.print(_panel("Agent", _as_text(res)))
                continue

            if raw.startswith("chat:"):
                if convo_chain is None:
                    console.print(_panel("Chat", "Chat chain not available."))
                    continue
                q = raw.split(":", 1)[1].strip()
                res = _safe_invoke(
                    convo_chain,
                    q,
                    config={"configurable": {"session_id": chat_session_id}},
                )
                console.print(_panel("Chat", _as_text(res)))
                continue

            if raw.startswith("rag:"):
                q = raw.split(":", 1)[1].strip()
                res = _safe_invoke(rag, q)
                console.print(_panel("RAG", _as_text(res)))
                continue

            # ----- default: RAG -----
            res = _safe_invoke(rag, raw)
            console.print(_panel("RAG", _as_text(res)))

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted[/bold yellow]")
            break
        except Exception as e:
            console.print(_panel("Error", f"{e.__class__.__name__}: {e}"))


if __name__ == "__main__":
    try:
        run_cli()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
