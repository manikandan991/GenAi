from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

def load_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    for p in path.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), autodetect_encoding=True).load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
    return docs
