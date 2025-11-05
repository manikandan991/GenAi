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

from agents.react_agent_web import build_agent_with_memory
agent = build_agent_with_memory()
cfg = {"configurable": {"session_id": "demo-1"}}

print(agent.invoke(
    {"input": "Any positive news about JSL."},
    config=cfg
)["output"])
