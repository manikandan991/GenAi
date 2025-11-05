from __future__ import annotations
from rich.console import Console
from rich.panel import Panel

from config import PINECONE_INDEX_NAME
from chains.rag_chain import build_rag_chain
from chains.convo_chain import build_conversational_chain
from agents.react_agent import build_react_agent

console = Console()

def run_cli():
    memory, convo_chain = build_conversational_chain()
    rag = build_rag_chain(PINECONE_INDEX_NAME)
    agent = build_react_agent()

    console.print(Panel("LangChain POC — type 'quit' to exit.\nCommands: rag: <q>, agent: <q>, chat: <q>", title="LCEL + RAG + Agent"))

    while True:
        try:
            raw = input("> ").strip()
            if raw.lower() in {"quit", "exit"}:
                break
            if raw.startswith("rag:"):
                q = raw.split(":", 1)[1].strip()
                ans = rag.invoke(q)
                console.print(Panel(ans.content if hasattr(ans, 'content') else str(ans), title="RAG"))
            elif raw.startswith("agent:"):
                q = raw.split(":", 1)[1].strip()
                result = agent.invoke({"input": q})
                console.print(Panel(result.get("output", str(result)), title="Agent"))
            else:
                result = convo_chain.invoke(raw)
                console.print(Panel(result.content, title="Chat"))
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    run_cli()
