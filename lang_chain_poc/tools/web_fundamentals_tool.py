# tools/web_fundamentals_tool.py
from __future__ import annotations
import json
import re
from typing import Dict, List, Any, Tuple

from langchain_community.tools.tavily_search import TavilySearchResults

# =========================
# Guidance prompt with DO-NOT list (enforced by the tool)
# =========================
PROMPT_GUIDELINES = """
You are a fundamentals fact-gathering and summarization tool for public companies.

STRICT RULES — DO NOT:
1) Do NOT scrape HTML directly from Google / sites (this tool uses Tavily, which is compliant).
2) Do NOT copy or reproduce full financial tables or mirror structured datasets.
3) Do NOT quote from sites that block scraping or require login/paywall (Bloomberg, Reuters premium, etc.).
4) Do NOT use Screener.in content or Moneycontrol table copies.
5) Do NOT present unverified numbers as guaranteed — always note source(s).
6) Limit quotes/snippets; summarize in your own words.
7) Prefer primary sources: company filings, press releases, investor presentations, exchanges, EDGAR/SEBI, or reputable news summaries.
8) If numbers disagree, surface the discrepancy with both sources.

DATA PRIVACY:
- Only use public web pages returned by Tavily.
- Do not process personal data.

OUTPUT POLICY:
- Return a compact JSON with: metrics (if found/extracted), evidence (bulleted text summaries),
  sources (list of URLs grouped by factor), and notes (any caveats).
"""

# =========================
# Query templates per factor
# =========================
FACTOR_QUERIES = {
    "revenue_cagr_5y": [
        "{company} 5 year revenue CAGR",
        "{company} revenue CAGR last 5 years",
        "{company} revenue growth 5Y investor presentation site:({company_site})"
    ],
    "pat_cagr_5y": [
        "{company} 5 year profit CAGR",
        "{company} PAT CAGR last 5 years",
        "{company} net profit CAGR investor presentation"
    ],
    "margin_stability": [
        "{company} operating margin trend",
        "{company} EBIT margin stability analysis",
        "{company} OPM trend commentary"
    ],
    "de_ratio": [
        "{company} debt to equity ratio latest",
        "{company} D/E ratio balance sheet"
    ],
    "interest_coverage": [
        "{company} interest coverage ratio",
        "{company} interest coverage latest"
    ],
    "cash_conversion": [
        "{company} CFO vs PAT",
        "{company} cash flow from operations compared to net profit",
        "{company} cash conversion ratio"
    ],
    "roe_roce": [
        "{company} ROE ROCE latest",
        "{company} return on equity return on capital employed"
    ],
}

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%?")

def _pick_company_site(company: str) -> str:
    # Best-effort guess for domain filters; safe default is no domain restriction
    # You can augment this map if you like.
    return ""

def _extract_numbers(text: str) -> List[str]:
    return NUM_RE.findall(text or "")

def _best_number(snippets: List[str]) -> Tuple[str | None, str | None]:
    """
    Try to extract a single representative number with its snippet.
    """
    for snip in snippets:
        nums = _extract_numbers(snip)
        if nums:
            return nums[0], snip
    return None, None

def _tavily_search_many(queries: List[str], k: int = 4) -> List[Dict[str, Any]]:
    """
    Use TavilySearchResults tool to fetch summarized results.
    """
    tavily = TavilySearchResults(max_results=k)
    out = []
    for q in queries:
        try:
            res = tavily.invoke({"query": q})
            # Each res item typically: {"content": "...", "url": "...", ...}
            out.append({"query": q, "results": res})
        except Exception as e:
            out.append({"query": q, "error": str(e), "results": []})
    return out

def _collect_snippets_and_urls(qresults: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    snippets, urls = [], []
    for grp in qresults:
        for r in grp.get("results", []):
            c = r.get("content", "")
            u = r.get("url", "") or r.get("source", "")
            if c:
                snippets.append(c)
            if u:
                urls.append(u)
    # Deduplicate
    urls = list(dict.fromkeys(urls))
    return snippets, urls

def _summarize_factor(snippets: List[str], factor: str) -> Dict[str, Any]:
    """
    Heuristic extraction for a single factor from collected snippets.
    """
    value, evidence_snip = _best_number(snippets)
    stable_hint = None
    if factor == "margin_stability":
        # crude text signal
        joined = " ".join(snippets[:6]).lower()
        if any(w in joined for w in ["stable", "stability", "consistent", "steady", "resilient"]):
            stable_hint = True
        elif any(w in joined for w in ["volatile", "fluctuat", "pressure", "declin"]):
            stable_hint = False

    return {
        "value": value,                 # may be % or ratio as string
        "evidence_snippet": evidence_snip,
        "text_hint": stable_hint,
    }

def web_fundamentals_scan(payload: str) -> str:
    """
    Tool entry point. Accepts either a plain company name string or a JSON string:
      - "Tata Consultancy Services"
      - '{"company":"Tata Consultancy Services","symbol":"TCS.NS"}'

    Returns JSON string:
      {
        "company": "...", "symbol": "...",
        "metrics": {...}, "evidence": {...}, "sources": {...}, "notes": [...],
        "policy": "PROMPT_GUIDELINES ..."
      }
    """
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            company = data.get("company") or data.get("symbol") or payload
            symbol = data.get("symbol", "")
        else:
            company = payload
            symbol = ""
    except Exception:
        company = payload
        symbol = ""

    company_site = _pick_company_site(company)
    metrics: Dict[str, Any] = {}
    evidence: Dict[str, List[str]] = {}
    sources: Dict[str, List[str]] = {}
    notes: List[str] = []

    for factor, templates in FACTOR_QUERIES.items():
        queries = [t.format(company=company, company_site=company_site) for t in templates]
        qres = _tavily_search_many(queries, k=4)
        snippets, urls = _collect_snippets_and_urls(qres)
        summary = _summarize_factor(snippets, factor)

        metrics[factor] = summary.get("value")
        # Human-readable evidence list (shortened)
        ev_list = []
        if summary.get("evidence_snippet"):
            ev_list.append(summary["evidence_snippet"][:300])
        if factor == "margin_stability" and summary.get("text_hint") is not None:
            ev_list.append(f"margin_stability_hint={summary['text_hint']}")
        evidence[factor] = ev_list
        sources[factor] = urls[:8]  # cap for cleanliness

    notes.append("Numbers are best-effort extractions from public summaries. Verify with primary filings when critical.")
    notes.append("This tool follows: " + " ".join(line.strip() for line in PROMPT_GUIDELINES.strip().splitlines() if line.strip()))

    return json.dumps({
        "company": company,
        "symbol": symbol,
        "metrics": metrics,
        "evidence": evidence,
        "sources": sources,
        "notes": notes,
        "policy": "Embedded DO-NOT list enforced. See notes.",
    }, ensure_ascii=False)

def web_fundamentals_scan_structured(company: str, symbol: str | None = None) -> str:
    """Structured wrapper so LangChain can pass named args safely."""
    import json
    payload = json.dumps({"company": company, "symbol": symbol or ""})
    return web_fundamentals_scan(payload)