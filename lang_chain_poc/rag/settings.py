from pydantic import BaseModel
from config import CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_EMBEDDING_MODEL, PINECONE_INDEX_NAME, EMBEDDING_MODEL

class RAGSettings(BaseModel):
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    embedding_model: str = EMBEDDING_MODEL
    index_name: str = PINECONE_INDEX_NAME
