from __future__ import annotations
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import DEFAULT_K

def build_rag_chain(index_name: str) -> RunnablePassthrough:
    retriever = PineconeVectorStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings()
    ).as_retriever(search_kwargs={"k": DEFAULT_K})

    system = ("You are a helpful analyst. Use the provided CONTEXT to answer.\n"
              "If the answer isn't in context, say you don't know.")

    prompt = PromptTemplate.from_template(
        """{system}

CONTEXT:
{context}

QUESTION: {question}

Answer concisely.
"""
    )

    llm = ChatOpenAI()

    setup = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
        "system": lambda _: system,
    })

    chain = setup | prompt | llm
    return chain
