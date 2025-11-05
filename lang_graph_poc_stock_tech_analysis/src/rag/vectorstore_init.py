import os
from typing import List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_kb_index(kb_dir: str = "./src/rag/kb", chroma_dir: str = "./artifacts/chroma"):
    os.makedirs(chroma_dir, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docs: List[Document] = []
    for fname in os.listdir(kb_dir):
        if fname.endswith(".md"):
            text = open(os.path.join(kb_dir, fname), "r", encoding="utf-8").read()
            for chunk in splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"source": fname}))
    vs = Chroma.from_documents(docs, emb, persist_directory=chroma_dir)
    return vs
