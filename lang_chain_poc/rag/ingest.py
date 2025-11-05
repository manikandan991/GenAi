# rag/ingest.py
from __future__ import annotations
import sys
import os
import json
import hashlib
import sqlite3
import time
from typing import List, Dict, Tuple
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import tiktoken
import requests

from rag.loader import load_documents
from rag.settings import RAGSettings

# =========================
# Config / constants
# =========================

# Choose one key below in your .env / config as RAGSettings.embedding_model
# and make sure Pinecone index dimension matches the value here.
MODEL_DIMS = {
    "hf:bge-small-en-v1.5": 384,   # BAAI/bge-small-en-v1.5
    "hf:bge-base-en-v1.5":  768,   # BAAI/bge-base-en-v1.5
    "hf:all-MiniLM-L6-v2":  384,   # sentence-transformers/all-MiniLM-L6-v2
}

BATCH_SIZE = 128
CACHE_PATH = "/Workspace/Shared/lang_chain_poc/sqlite_temp/embedding_cache.sqlite"
BACKLOG_PATH = "/Workspace/Shared/lang_chain_poc/sqlite_temp/ingest_backlog.jsonl"
MIN_CHUNK_CHARS = 20

# HF Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/{repo_id}"
HF_REPOS = {
    "hf:bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "hf:bge-base-en-v1.5":  "BAAI/bge-base-en-v1.5",
    "hf:all-MiniLM-L6-v2":  "sentence-transformers/all-MiniLM-L6-v2",
}

def _hf_headers() -> Dict[str, str]:
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


# =========================
# Local cache (SQLite)
# =========================
Path(Path(CACHE_PATH).parent).mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(CACHE_PATH)
conn.execute(
    """CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,
        model TEXT,
        vec BLOB
    )"""
)
conn.commit()

def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def cache_get_many(keys: List[str], model: str) -> Dict[str, List[float]]:
    if not keys:
        return {}
    q = "SELECT key, vec FROM cache WHERE key IN ({}) AND model=?".format(
        ",".join("?" * len(keys))
    )
    rows = conn.execute(q, (*keys, model)).fetchall()
    return {k: json.loads(v) for (k, v) in rows}

def cache_put_many(items: List[Tuple[str, str, List[float]]]):
    if not items:
        return
    with conn:
        conn.executemany(
            "REPLACE INTO cache(key, model, vec) VALUES (?,?,?)",
            [(k, m, json.dumps(v)) for (k, m, v) in items],
        )


# =========================
# Tokenizer (optional)
# =========================
def tokenizer():
    # Good enough for estimates; not strictly required.
    return tiktoken.get_encoding("cl100k_base")


# =========================
# HF output normalization
# =========================
def _mean_pool_token_matrix(matrix: List[List[float]]) -> List[float]:
    # matrix: tokens x dim  ->  return: dim
    if not matrix:
        return []
    dim = len(matrix[0])
    sums = [0.0] * dim
    for token_vec in matrix:
        for j, v in enumerate(token_vec):
            sums[j] += float(v)
    n = max(1, len(matrix))
    return [s / n for s in sums]

def _normalize_hf_output(data):
    """
    Normalize HF output to List[List[float]] where each inner list is a single embedding vector.
    Handles these cases:
      - [float, float, ...]                    -> single vector
      - [[float, ...], [float, ...], ...]      -> batch of vectors
      - [[[float, ...], ...], [[float, ...]]]  -> batch of token matrices -> mean-pooled per item
    """
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected HF output: {type(data)}")

    # Case A: single vector (one input, sentence-level)
    if all(isinstance(x, (int, float)) for x in data):
        return [[float(v) for v in data]]

    # Case B: batch of sentence-level vectors
    if all(isinstance(row, list) and row and isinstance(row[0], (int, float)) for row in data):
        return [[float(v) for v in row] for row in data]

    # Case C: batch of token matrices -> mean pool each
    if all(isinstance(seq, list) and seq and isinstance(seq[0], list) for seq in data):
        pooled = []
        for seq in data:
            pooled.append(_mean_pool_token_matrix(seq))
        return pooled

    raise RuntimeError("Unrecognized HF embedding response structure")


def _embed_batch_hf(texts: List[str], model_key: str) -> List[List[float]]:
    """Call HF Inference API and return List[List[float]] (one vector per input)."""
    repo_id = HF_REPOS[model_key]
    url = HF_API_URL.format(repo_id=repo_id)
    headers = _hf_headers()
    if not headers:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set in environment.")

    while True:
        r = requests.post(url, headers=headers, json={"inputs": texts})
        if r.status_code == 200:
            data = r.json()
            vectors = _normalize_hf_output(data)
            # ensure floats (not Decimal/np types) and no nested lists
            vectors = [[float(x) for x in vec] for vec in vectors]
            return vectors
        elif r.status_code in (429, 503):
            time.sleep(1.2)
            continue
        else:
            raise RuntimeError(f"HF Inference API error {r.status_code}: {r.text}")


def embed_texts_with_cache(texts: List[str], model: str) -> Tuple[List[List[float] | None], List[int]]:
    """Return (vectors, missing_idx). With HF API we don't defer; missing_idx empty unless error."""
    keys = [_hash_key(t) for t in texts]
    cached = cache_get_many(keys, model)
    out: List[List[float] | None] = [None] * len(texts)

    to_idx, to_send = [], []
    for i, (k, t) in enumerate(zip(keys, texts)):
        if k in cached:
            out[i] = cached[k]
        else:
            to_idx.append(i)
            to_send.append(t)

    for start in range(0, len(to_send), BATCH_SIZE):
        batch = to_send[start:start + BATCH_SIZE]
        vecs = _embed_batch_hf(batch, model)
        cache_items = []
        for j, v in enumerate(vecs):
            idx = to_idx[start + j]
            out[idx] = v
            cache_items.append((_hash_key(texts[idx]), model, v))
        cache_put_many(cache_items)

    return out, []


# =========================
# Optional no-op embeddings (only needed if you instantiate LC VectorStore elsewhere)
# =========================
class _NoopEmbeddings(Embeddings):
    def embed_documents(self, _): raise RuntimeError("Not used (we precompute & upsert vectors directly).")
    def embed_query(self, _): raise RuntimeError("Not used (we precompute & upsert vectors directly).")


# =========================
# Utility: backlog writer (kept for parity; HF path usually has no backlog)
# =========================
def write_backlog(chunks: List[Document], missing_idx: List[int], index_name: str, model: str):
    if not missing_idx:
        return
    Path(Path(BACKLOG_PATH).parent).mkdir(parents=True, exist_ok=True)
    with open(BACKLOG_PATH, "a", encoding="utf-8") as f:
        for mi in missing_idx:
            item = {
                "id": _hash_key(chunks[mi].page_content),
                "text": chunks[mi].page_content,
                "metadata": chunks[mi].metadata,
                "index_name": index_name,
                "model": model,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =========================
# Vector normalization helper (used in ingest & backfill)
# =========================
def _to_float_vector(vec, expected_dim: int) -> List[float]:
    # If token-level matrix, mean-pool to sentence vector
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = _mean_pool_token_matrix(vec)
    # Force pure Python floats
    vec = [float(v) for v in vec]
    # Enforce dimension
    if len(vec) != expected_dim:
        raise ValueError(f"wrong dim={len(vec)}, expected={expected_dim}")
    return vec


# =========================
# Main ingest
# =========================
def ingest(folder: str):
    settings = RAGSettings()
    embed_model = settings.embedding_model
    if embed_model not in MODEL_DIMS:
        raise ValueError(
            f"Unknown embedding model '{embed_model}'. "
            f"Choose one of {list(MODEL_DIMS)} in RAGSettings.embedding_model"
        )
    index_dim = MODEL_DIMS[embed_model]

    print("Loading documents…")
    docs = load_documents(folder)
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print("Splitting into chunks…")
    chunks: List[Document] = splitter.split_documents(docs)

    # Filter short/empty chunks & exact duplicate content
    seen = set()
    filtered: List[Document] = []
    for d in chunks:
        txt = " ".join(d.page_content.split())
        if len(txt) < MIN_CHUNK_CHARS:
            continue
        h = _hash_key(txt)
        if h in seen:
            continue
        seen.add(h)
        d.page_content = txt
        filtered.append(d)
    chunks = filtered
    print(f"Generated {len(chunks)} unique chunks")

    if not chunks:
        print("No chunks to process. Done.")
        return

    print("Initializing Pinecone index…")
    pc = Pinecone()
    existing = {i.name for i in pc.list_indexes()}
    if settings.index_name not in existing:
        pc.create_index(
            name=settings.index_name,
            dimension=index_dim,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )

    print(f"Embedding with model='{embed_model}' (dim={index_dim}) in batches of {BATCH_SIZE}…")
    texts = [c.page_content for c in chunks]
    vectors, missing_idx = embed_texts_with_cache(texts, model=embed_model)

    # Prepare Pinecone upsert payload (CONSISTENT SPACES INDENTATION)
    to_upsert: List[Dict] = []
    expected_dim = index_dim
    skipped = 0
    for i, vec in enumerate(vectors):
        if vec is None:
            skipped += 1
            continue
        try:
            vec = _to_float_vector(vec, expected_dim)
        except Exception as e:
            print(f"Skipping vector[{i}]: {e}")
            skipped += 1
            continue

        _id = _hash_key(chunks[i].page_content)
        meta = dict(chunks[i].metadata) if chunks[i].metadata else {}
        meta.update({"source_hash": _id, "model": embed_model})

        to_upsert.append({
            "id": _id,
            "values": vec,
            "metadata": meta
        })
    if skipped:
        print(f"Skipped {skipped} chunk(s) due to invalid/incorrect-dimension vectors.")

    if to_upsert:
        print(f"Upserting {len(to_upsert)} vectors to Pinecone index '{settings.index_name}'…")
        index = pc.Index(settings.index_name)
        index.upsert(vectors=to_upsert)
        print("Upsert complete.")
    else:
        print("No vectors to upsert.")

    if missing_idx:
        write_backlog(chunks, missing_idx, index_name=settings.index_name, model=embed_model)
        print(f"Deferred {len(missing_idx)} chunk(s) to backlog: {BACKLOG_PATH}")

    print("Done.")


# =========================
# Optional: backfill helper (reads BACKLOG_PATH and upserts)
# =========================
def backfill_backlog():
    if not Path(BACKLOG_PATH).exists():
        print("No backlog file found. Nothing to backfill.")
        return

    lines = Path(BACKLOG_PATH).read_text(encoding="utf-8").splitlines()
    if not lines:
        print("Backlog empty. Nothing to backfill.")
        return

    # Group by (index_name, model)
    buckets: Dict[Tuple[str, str], List[Dict]] = {}
    for ln in lines:
        item = json.loads(ln)
        key = (item["index_name"], item["model"])
        buckets.setdefault(key, []).append(item)

    pc = Pinecone()
    for (index_name, model), items in buckets.items():
        print(f"Backfilling {len(items)} items into index '{index_name}' with model '{model}'…")
        texts = [it["text"] for it in items]
        vecs, missing_idx = embed_texts_with_cache(texts, model=model)

        # Ensure index exists with correct dims
        existing = {i.name for i in pc.list_indexes()}
        dim = MODEL_DIMS.get(model)
        if not dim:
            raise ValueError(f"Unknown model '{model}' in backlog.")
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )

        index = pc.Index(index_name)
        to_upsert = []
        expected_dim = dim
        for i, vec in enumerate(vecs):
            if vec is None:
                continue
            try:
                v = _to_float_vector(vec, expected_dim)
            except Exception as e:
                print(f"Skipping backlog vector[{i}]: {e}")
                continue

            _id = items[i]["id"]
            meta = items[i].get("metadata", {})
            meta.update({"source_hash": _id, "model": model})
            to_upsert.append({"id": _id, "values": v, "metadata": meta})

        if to_upsert:
            index.upsert(vectors=to_upsert)
            print(f"Upserted {len(to_upsert)} items.")
        if missing_idx:
            print(f"Still pending due to errors: {len(missing_idx)} item(s).")

    # Clear backlog
    Path(BACKLOG_PATH).unlink(missing_ok=True)
    print("Backfill complete; backlog cleared.")


if __name__ == "__main__":
    # Example:
    #   python rag/ingest.py "/Workspace/Users/.../kb_docs"
    folder = sys.argv[1] if len(sys.argv) > 1 else "/Workspace/Shared/lang_chain_poc/data/samples/kb_docs"
    ingest(folder)
