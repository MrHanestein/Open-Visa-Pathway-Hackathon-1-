import os
from openai import OpenAI
import streamlit as st

# ---------- OpenAI client via environment variable ----------
# Will crash clearly if OPENAI_API_KEY is missing
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set. "
        "In PowerShell, run:\n"
        '$env:OPENAI_API_KEY = "YOUR_REAL_KEY_HERE"'
    )

client = OpenAI(api_key=api_key)


def init_model(default_model: str = "gpt-5-nano") -> None:
    """
    Ensure st.session_state has an openai_model key.
    Call this once at app startup.
    """
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = default_model


def stream_chat_completion(messages: list[dict]):
    """
    Wraps client.chat.completions.create with stream=True
    and returns the streaming iterator.

    messages = [{"role": "user"/"assistant"/"system", "content": "..."}]
    """
    return client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        stream=True,
    )


def create_chat_completion(messages: list[dict]):
    """
    Non-streaming version (not used right now, but handy if you need it).
    """
    return client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
    )
# End of llm.py