# chains/convo_chain.py
from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory  # simple in-memory history
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI  # or your LLM of choice

# If you aren’t using OpenAI, swap ChatOpenAI for your provider's chat LLM class.

_SESSION_STORE: dict[str, BaseChatMessageHistory] = {}

def _get_history(session_id: str) -> BaseChatMessageHistory:
    # You can replace with Redis/SQL-backed histories later
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

def build_conversational_chain() -> Tuple[BaseChatMessageHistory, RunnableWithMessageHistory]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, concise assistant. Keep answers tight."),
        MessagesPlaceholder(variable_name="history"),   # ⬅️ this is the right way
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # swap to your model/provider if needed
    chain = prompt | llm

    convo = RunnableWithMessageHistory(
        chain,
        get_session_history=_get_history,
        history_messages_key="history",   # must match MessagesPlaceholder name
        input_messages_key="input",       # the user field name you’ll pass
        output_messages_key="ai"          # optional; defaults to the LLM output
    )
    # Return a *handle* to history via _get_history; actual instance comes per session_id.
    return _get_history, convo
