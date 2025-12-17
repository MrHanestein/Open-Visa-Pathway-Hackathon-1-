import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

# Embeddings model
EMBED_MODEL = "text-embedding-3-small"


# ---------------------------
# Secrets / client helpers
# ---------------------------

def _get_api_key() -> Optional[str]:
    """
    Return API key if available, otherwise None.
    Order:
      1) environment variable OPENAI_API_KEY
      2) Streamlit secrets OPENAI_API_KEY
    """
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k

    # Accessing st.secrets can raise if secrets.toml doesn't exist
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        return None


@st.cache_resource
def _client_for_key(api_key: str) -> OpenAI:
    # Cache the OpenAI client as a shared resource
    return OpenAI(api_key=api_key)


def _get_client() -> Optional[OpenAI]:
    key = _get_api_key()
    if not key:
        return None
    return _client_for_key(key)


def _require_client(why: str = "OpenAI API") -> OpenAI:
    """
    For features that must have a client (chat), stop the app with a clear message.
    """
    client = _get_client()
    if client is None:
        st.error(
            f"{why} is not configured.\n\n"
            "Fix one of these:\n"
            "1) Create `.streamlit/secrets.toml` in the folder you run `streamlit run` from, with:\n"
            '   OPENAI_API_KEY="sk-..."\n'
            "2) Or set environment variable OPENAI_API_KEY.\n"
        )
        st.stop()
    return client


# ---------------------------
# Chat helpers
# ---------------------------

def init_model(default_model: str = "gpt-5-nano") -> None:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = default_model


def stream_chat_completion(messages: list[dict]):
    """
    Streaming Chat Completions.
    """
    client = _require_client("Chat (OpenAI API)")
    return client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        stream=True,
    )


def create_chat_completion(messages: list[dict]):
    client = _require_client("Chat (OpenAI API)")
    return client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
    )


# ---------------------------
# RAG: KB -> chunk -> embed -> retrieve_top_k
# ---------------------------

@dataclass
class KBChunk:
    kb_file: str
    title: str
    source_url: str
    text: str
    embedding: List[float]


def _parse_kb_markdown(md_text: str) -> Tuple[str, List[str], str]:
    """
    Return: (title, links, body).
    - title: first "# " heading if present
    - links: any http(s) tokens found
    - body: full text (we chunk it later)
    """
    lines = md_text.splitlines()

    title = "KB"
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip()
            break

    links: List[str] = []
    for ln in lines:
        if "http://" in ln or "https://" in ln:
            for token in ln.split():
                if token.startswith("http://") or token.startswith("https://"):
                    links.append(token.strip())

    body = md_text.strip()
    return title, links, body


def _chunk_text(text: str, max_chars: int = 900) -> List[str]:
    """
    Simple paragraph chunking (MVP).
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buff = ""

    for p in paras:
        if len(buff) + len(p) + 2 <= max_chars:
            buff = (buff + "\n\n" + p).strip()
        else:
            if buff:
                chunks.append(buff)
            buff = p

    if buff:
        chunks.append(buff)
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeddings API: pass a string or list of strings.

    If embeddings fail (missing key, rate limit, model issue), we:
    - show a Streamlit error (visible in UI)
    - return [] so callers can safely bail out without crashing
    """
    client = _get_client()
    if client is None:
        st.warning("No OPENAI_API_KEY found — embeddings (RAG) are disabled.")
        return []

    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        st.error(f"Embedding error (RAG disabled for this request): {e}")
        return []


def _kb_fingerprint(kb_dir: str) -> Tuple[Tuple[str, float], ...]:
    """
    Used only to invalidate cache when kb files change.
    """
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    return tuple((os.path.basename(p), os.path.getmtime(p)) for p in md_files)


@st.cache_data(show_spinner="Indexing knowledge base…")
def _build_kb_index_cached(
    kb_dir: str,
    fingerprint: Tuple[Tuple[str, float], ...],
) -> List[KBChunk]:
    """
    Build KB index (cached). fingerprint param forces recompute when files change.

    Uses st.cache_data because we are caching "data" (a list of pickleable dataclasses).
    """
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    if not md_files:
        return []

    meta: List[Tuple[str, str, str, str]] = []
    texts: List[str] = []

    for fp in md_files:
        md = open(fp, "r", encoding="utf-8").read()
        title, links, body = _parse_kb_markdown(md)
        primary_url = links[0] if links else ""

        for ch in _chunk_text(body):
            meta.append((os.path.basename(fp), title, primary_url, ch))
            texts.append(ch)

    embs = embed_texts(texts)

    # If embeddings failed, don't crash: return empty index for this run
    if not embs or len(embs) != len(texts):
        st.warning("Knowledge base indexing skipped (embedding failure).")
        return []

    kb_index: List[KBChunk] = []
    for (kb_file, title, url, ch), emb in zip(meta, embs):
        kb_index.append(
            KBChunk(
                kb_file=kb_file,
                title=title,
                source_url=url,
                text=ch,
                embedding=emb,
            )
        )

    return kb_index


def build_kb_index(kb_dir: str = "kb") -> List[KBChunk]:
    fp = _kb_fingerprint(kb_dir)
    return _build_kb_index_cached(kb_dir, fp)


def _cosine_sims(query_emb: List[float], kb_index: List[KBChunk]) -> List[float]:
    q = np.array(query_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q) + 1e-8

    sims: List[float] = []
    for item in kb_index:
        v = np.array(item.embedding, dtype=np.float32)
        sim = float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) + 1e-8)))
        sims.append(sim)
    return sims


def retrieve_top_k(query: str, k: int = 3, kb_dir: str = "kb") -> List[KBChunk]:
    """
    Cosine similarity over embeddings; returns top-k chunks.
    """
    kb_index = build_kb_index(kb_dir=kb_dir)
    if not kb_index:
        return []

    q_embs = embed_texts([query])
    if not q_embs:
        return []
    q_emb = q_embs[0]

    sims = _cosine_sims(q_emb, kb_index)
    top_idx = np.argsort(sims)[::-1][:k]
    return [kb_index[i] for i in top_idx]


def retrieve_top_k_with_scores(
    query: str,
    k: int = 3,
    kb_dir: str = "kb",
) -> List[Tuple[KBChunk, float]]:
    """
    Same as retrieve_top_k, but returns (KBChunk, similarity_score).
    """
    kb_index = build_kb_index(kb_dir=kb_dir)
    if not kb_index:
        return []

    q_embs = embed_texts([query])
    if not q_embs:
        return []
    q_emb = q_embs[0]

    sims = _cosine_sims(q_emb, kb_index)
    top_idx = np.argsort(sims)[::-1][:k]
    return [(kb_index[i], float(sims[i])) for i in top_idx]
