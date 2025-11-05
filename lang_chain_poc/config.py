from __future__ import annotations
import os
from dotenv import load_dotenv

# In Databricks, you might not have a .env file—this call is harmless either way.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL="hf:bge-base-en-v1.5"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-poc-index")

DEFAULT_K = int(os.getenv("DEFAULT_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

OPENAI_EMBEDDING_MODEL="hf:bge-base-en-v1.5"   # (we just reuse this var name)
