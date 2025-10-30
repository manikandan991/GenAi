import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2):
    # Uses OpenAI via langchain_openai
    return ChatOpenAI(model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))

def summarize_doctor_feedback(llm, notes: List[str], max_notes: int = 50) -> str:
    subset = notes[:max_notes]
    joined = "\n".join(f"- {n}" for n in subset if isinstance(n, str))
    prompt = f"""You are a clinical trial analyst.
Extract key themes, adverse events, and anomalies from the doctor's notes below. Be concise and bullet the findings.

Notes:
{joined}
"""
    return llm.invoke(prompt).content

def regulatory_style_summary(llm, cohort_table: str, issues_summary: str) -> str:
    prompt = f"""Draft a *three-paragraph* FDA-style summary:
1) Study design & cohorts (use this table):\n{cohort_table}
2) Efficacy and safety outcomes (mention compliance and adverse rates)
3) Notable risks, mitigations, and next steps

Also incorporate these issues: {issues_summary}
"""
    return llm.invoke(prompt).content
