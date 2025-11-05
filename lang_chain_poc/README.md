# LangChain POC (Databricks-friendly)

This POC demonstrates:
- LCEL (RunnableParallel, RunnablePassthrough)
- Conversational Buffer Memory
- RAG (Pinecone)
- ReAct agent with tools: web search, summarize, Screener fundamentals

## Databricks Quickstart

1) Upload this project folder (or the zip) to DBFS (e.g., `/FileStore/langchain-poc`).  
2) In a notebook, run:

```python
# If you uploaded the zip to /FileStore, unzip it:
# dbutils.fs.cp("dbfs:/FileStore/langchain-poc.zip", "file:/tmp/langchain-poc.zip")
# import zipfile; zipfile.ZipFile("/tmp/langchain-poc.zip").extractall("/Workspace/")
# Or just clone from repo / or use %sh to wget.

%pip install -r /dbfs/FileStore/langchain-poc/requirements.txt

import sys
sys.path.append('/dbfs/FileStore/langchain-poc')  # so Python can import the package
```

3) Configure secrets: prefer Databricks Secrets:
```python
# Example: fetching secrets safely (replace scope & keys)
import os
# os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope="my-scope", key="openai")
# os.environ['PINECONE_API_KEY'] = dbutils.secrets.get(scope="my-scope", key="pinecone")
# os.environ['TAVILY_API_KEY'] = dbutils.secrets.get(scope="my-scope", key="tavily")
```

4) Ingest your KB:
```python
from rag.ingest import ingest
ingest('/dbfs/FileStore/langchain-poc/data/samples/kb_docs')
```

5) Run chains/agent:
```python
from chains.rag_chain import build_rag_chain
from chains.convo_chain import build_conversational_chain
from agents.react_agent import build_react_agent
from config import PINECONE_INDEX_NAME

memory, convo = build_conversational_chain()
rag = build_rag_chain(PINECONE_INDEX_NAME)
agent = build_react_agent()

rag.invoke("What are the key points in our policy doc?").content
agent.invoke({"input":"Find fundamentally strong Indian IT largecaps and show PE/ROE"})["output"]
```

