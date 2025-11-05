from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

_summary_prompt = PromptTemplate.from_template(
    """Summarize the following web results into key bullets, citing sources inline.

{content}

Bulleted summary:"""
)

def summarize_search_results(results) -> str:
    text = "\n\n".join([
        f"- {r.get('title','')} — {r.get('url','')}\n{r.get('content','')}" if isinstance(r, dict) else str(r)
        for r in results
    ])
    llm = ChatOpenAI()
    return llm.invoke(_summary_prompt.format(content=text)).content
