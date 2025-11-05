from __future__ import annotations
from typing import List
from langchain.tools import tool
from config import TAVILY_API_KEY, SERPAPI_API_KEY

# Option A: Tavily
if TAVILY_API_KEY:
    from langchain_community.tools.tavily_search import TavilySearchResults
    search_tool = TavilySearchResults(max_results=5)

# Option B: SerpAPI
elif SERPAPI_API_KEY:
    from langchain_community.utilities import SerpAPIWrapper
    serp = SerpAPIWrapper()

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for up-to-date information. Input: a search query string."""
        return serp.run(query)
else:
    @tool("web_search")
    def web_search(_: str) -> str:
        """Fallback search tool: return a helpful message if no API is configured."""
        return "Search provider not configured. Set TAVILY_API_KEY or SERPAPI_API_KEY."

# If Tavily is available, expose it as a tool named web_search for uniformity
if TAVILY_API_KEY:
    @tool("web_search")
    def web_search(query: str) -> List[dict]:
        """Search the web and return JSON results (title, url, content snippets)."""
        return search_tool.invoke({"query": query})
