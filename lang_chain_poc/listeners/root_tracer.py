# listeners/root_tracer.py

from typing import Any
from langchain_core.messages import AIMessage, BaseMessage

def extract_ai_text(outputs: Any) -> str:
    """Safely extract text from ANY LangChain output shape."""

    # Case 1: dict
    if isinstance(outputs, dict):
        # Check common keys
        for key in ("output", "text", "answer", "content"):
            if key in outputs and isinstance(outputs[key], str):
                return outputs[key]

        # Check messages
        msgs = outputs.get("messages")
        if isinstance(msgs, list) and msgs:
            # Look for AIMessage
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    return m.content
            # Look for assistant-like dict
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("type") in ("ai", "assistant"):
                    return m.get("content", "")

        return str(outputs)

    # Case 2: AIMessage or other message type
    if isinstance(outputs, BaseMessage):
        return getattr(outputs, "content", str(outputs))

    # Case 3: raw string or anything else
    return str(outputs)


class RootListenersTracer:
    """A safe tracer that never breaks your app."""
    
    def on_chain_end(self, run_id: str, outputs: Any, **kwargs):
        try:
            ai_text = extract_ai_text(outputs)
            print(f"[TRACE][chain_end] run_id={run_id} -> {ai_text[:150]}...")
        except Exception as e:
            print(f"[TRACE] Error ignored: {e}")
