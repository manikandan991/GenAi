import os
import google.generativeai as genai
from litellm import completion

def llm_chat(messages, model_preference: str = "gemini") -> str:
    """messages: list of {role, content}. Try Gemini, fallback to LiteLLM (e.g., Ollama)."""
    if model_preference == "gemini" and os.getenv("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            joined = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(joined)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            print("Gemini failed, falling back:", e)
    try:
        resp = completion(model=os.getenv("FALLBACK_LLM", "ollama/llama3.1"), messages=messages)
        return resp.choices[0].message["content"]
    except Exception as e:
        print("LiteLLM fallback failed:", e)
        return "(LLM unavailable)"
