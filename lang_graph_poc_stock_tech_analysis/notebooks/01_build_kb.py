# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Build KB Index (Chroma)
# MAGIC Workspace-only paths (CE safe). Uses **langchain-huggingface** + **langchain-chroma**.

# COMMAND ----------

# MAGIC %pip install -q "langchain>=0.2" "langchain-community>=0.2" langchain-text-splitters                langchain-huggingface langchain-chroma sentence-transformers chromadb
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Imports & config
import os, json, traceback
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

PROJECT_ROOT = "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/"
KB_DIR = "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/src/rag/kb"
CHROMA_DIR = "/Workspace/Shared/lang_graph_poc_stock_tech_analysis/artifacts/chroma"

# Ensure artifact dir exists
os.makedirs(CHROMA_DIR, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("KB_DIR:", KB_DIR)
print("CHROMA_DIR:", CHROMA_DIR)

# COMMAND ----------

# Load KB .md files strictly from Workspace path
def load_kb_markdown(kb_dir: str) -> list[Document]:
    docs: list[Document] = []
    kb_root = Path(kb_dir)
    if not kb_root.exists():
        raise FileNotFoundError(
            f"KB_DIR not found at {kb_dir}. Create it and add .md files (e.g., indicators.md, patterns.md)."
        )
    for p in sorted(kb_root.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": p.name}))
    if not docs:
        raise RuntimeError(f"No .md files found in {kb_dir}.")
    return docs

raw_docs = load_kb_markdown(KB_DIR)
print(f"Loaded {len(raw_docs)} KB docs:")
for d in raw_docs:
    print(" -", d.metadata["source"])

# COMMAND ----------

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(raw_docs)
print("Chunks:", len(docs))

# COMMAND ----------

# Build embeddings & Chroma index (persist at CHROMA_DIR)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb  # <-- new

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# IMPORTANT: use a persistent client + a fixed collection name
COLLECTION = "stock_kb"
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create (or reuse) the collection and upsert docs
db = Chroma.from_documents(
    documents=docs,
    embedding=embed,
    client=client,
    collection_name=COLLECTION,
)

# NO db.persist() in langchain-chroma
result = {
    "status": "ok",
    "kb_docs": len(raw_docs),
    "chunks": len(docs),
    "chroma_dir": CHROMA_DIR,
    "collection": COLLECTION,
}
print(result)
dbutils.notebook.exit(json.dumps(result))
