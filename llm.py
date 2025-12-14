import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI

# Embeddings model (OpenAI docs show text-embedding-3-small usage) :contentReference[oaicite:2]{index=2}
EMBED_MODEL = "text-embedding-3-small"


def _get_api_key() -> str:
    # Streamlit best practice: use st.secrets or env var
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise RuntimeError("Missing OPENAI_API_KEY (set Streamlit secret or env var).")
    return k


@st.cache_resource
def _get_client() -> OpenAI:
    # Cache the OpenAI client as a shared resource :contentReference[oaicite:3]{index=3}
    return OpenAI(api_key=_get_api_key())


def init_model(default_model: str = "gpt-5-nano") -> None:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = default_model


def stream_chat_completion(messages: list[dict]):
    """
    Streaming Chat Completions. :contentReference[oaicite:4]{index=4}
    """
    client = _get_client()
    return client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        stream=True,
    )


def create_chat_completion(messages: list[dict]):
    client = _get_client()
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
    Embeddings API: pass a string or list of strings. :contentReference[oaicite:5]{index=5}
    """
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _kb_fingerprint(kb_dir: str) -> Tuple[Tuple[str, float], ...]:
    """
    Used only to invalidate cache when kb files change.
    """
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    return tuple((os.path.basename(p), os.path.getmtime(p)) for p in md_files)


@st.cache_data(show_spinner="Indexing knowledge base…")
def _build_kb_index_cached(kb_dir: str, fingerprint: Tuple[Tuple[str, float], ...]) -> List[KBChunk]:
    """
    Build KB index (cached). fingerprint param forces recompute when files change. :contentReference[oaicite:6]{index=6}
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


def retrieve_top_k(query: str, k: int = 3, kb_dir: str = "kb") -> List[KBChunk]:
    """
    Cosine similarity over embeddings; returns top-k chunks.
    """
    kb_index = build_kb_index(kb_dir=kb_dir)
    if not kb_index:
        return []

    q_emb = embed_texts([query])[0]
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q) + 1e-8

    sims = []
    for item in kb_index:
        v = np.array(item.embedding, dtype=np.float32)
        sim = float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) + 1e-8)))
        sims.append(sim)

    top_idx = np.argsort(sims)[::-1][:k]
    return [kb_index[i] for i in top_idx]
