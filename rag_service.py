#!/usr/bin/env python3
"""
PRWhisperer — FastAPI RAG: hybrid Chroma + BM25, optional vLLM via direct or PrefixPilot.

Environment:
  CHROMA_DIR         — Chroma dir (e.g. ./chroma_prwhisperer)
  EMBED_MODEL, EMBED_DEVICE, COLLECTION, BM25_PKL
  VLLM_BASE_URL      — e.g. http://127.0.0.1:8000/v1
  VLLM_MODEL
  PREFIX_PILOT_URL   — e.g. http://127.0.0.1:8001  (triage_routed)
  RAG_PORT, RAG_HOST
  PREFIX_DEBUG       — if set, log md5 of OpenAI prefix messages to stderr
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

_vectorstore: Chroma | None = None
_bm25: BM25Okapi | None = None
_bm25_docs: list[dict[str, Any]] = []
_tokenize_cache: list[list[str]] = []
_embedder: Any = None
_SYSTEM_TRIAGE = """You are PRWhisperer, a triage assistant for a GitHub repository.
Use ONLY the provided context snippets. If the answer is not in the context, say you do not have enough information.
Cite PR numbers and URLs from metadata when present.
Be concise: summary, risk, suggested labels/actions."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _rrf_fuse(
    results_lists: list[list[dict[str, Any]]], k: int = 60, top_n: int = 8
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}
    for rlist in results_lists:
        for rank, item in enumerate(rlist, start=1):
            key = item["id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payload[key] = item
    ordered = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [payload[k] for k, _ in ordered]


def _meta_int(m: dict[str, Any], key: str, default: int = 0) -> int:
    v = m.get(key, default)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _sort_citations(cites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(c: dict[str, Any]) -> tuple:
        m: dict[str, Any] = c.get("metadata") or {}
        cid = m.get("chunk_id")
        if cid is not None:
            return (0, _meta_int(m, "chunk_id", 0), _meta_int(m, "pr_number", 0), _meta_int(m, "chunk_index", 0))
        return (1, _meta_int(m, "pr_number", 0), _meta_int(m, "chunk_index", 0), str(c.get("id", "")))

    return sorted(cites, key=key)


def _load_bm25_payload(chroma_dir: Path) -> dict[str, Any]:
    env_p = os.environ.get("BM25_PKL", "").strip()
    candidates: list[Path] = []
    if env_p:
        candidates.append(Path(env_p))
    candidates.append(chroma_dir.parent / "bm25.pkl")
    candidates.append(chroma_dir / "bm25_corpus.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".pkl" or path.name.endswith(".pkl"):
                data = pickle.loads(path.read_bytes())
                if not isinstance(data, dict) or "chunks" not in data:
                    continue
                return data
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "chunks" in data:
                return data
        except (OSError, json.JSONDecodeError, pickle.UnpicklingError) as e:
            print(f"Skip BM25 file {path}: {e!s}", file=sys.stderr)
            continue
    print("No BM25 data found. Run ingest.py first.", file=sys.stderr)
    sys.exit(1)


def _load_sidecar(chroma_dir: Path) -> None:
    global _bm25, _bm25_docs, _tokenize_cache
    data = _load_bm25_payload(chroma_dir)
    _bm25_docs = data["chunks"]
    _tokenize_cache = [_tokenize(c["page_content"]) for c in _bm25_docs]
    _bm25 = BM25Okapi(_tokenize_cache)
    print(f"BM25 index: {len(_bm25_docs)} chunks (github_repo={data.get('github_repo', '?')!r})")


def _startup() -> None:
    global _vectorstore, _embedder
    chroma_dir = os.environ.get("CHROMA_DIR", "./data/chroma")
    path = Path(chroma_dir)
    if not path.is_dir():
        print(f"CHROMA_DIR not a directory: {chroma_dir}", file=sys.stderr)
        sys.exit(1)
    _load_sidecar(path)
    embed_model = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dev = os.environ.get("EMBED_DEVICE", "cpu")
    _embedder = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True},
    )
    coll = os.environ.get("COLLECTION", "pr_triage")
    _vectorstore = Chroma(
        collection_name=coll,
        persist_directory=str(path),
        embedding_function=_embedder,
    )
    try:
        n = _vectorstore._collection.count()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        n = -1
    print(f"Chroma loaded: collection={coll} count~ {n} path={path}")


@asynccontextmanager
async def lifespan(fastapi_instance: FastAPI):
    _startup()
    to = httpx.Timeout(10.0, read=1200.0, write=60.0, pool=5.0)
    async with httpx.AsyncClient(timeout=to) as client:
        fastapi_instance.state.http = client
        yield


app = FastAPI(title="PRWhisperer RAG", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(6, ge=1, le=30)


class TriageRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(6, ge=1, le=30)
    max_tokens: int = Field(1024, ge=1, le=32768)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    system_prompt: str | None = None


def _bm25_top(query: str, k: int) -> list[dict[str, Any]]:
    if not _bm25 or not _bm25_docs:
        return []
    tq = _tokenize(query)
    scores = _bm25.get_scores(tq)
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    out: list[dict[str, Any]] = []
    for i in idx:
        c = _bm25_docs[i]
        out.append(
            {
                "id": f"bm25:{i}",
                "score": float(scores[i]),
                "page_content": c["page_content"],
                "metadata": c.get("metadata", {}),
                "source": "bm25",
            }
        )
    return out


def _chroma_top(query: str, k: int) -> list[dict[str, Any]]:
    if not _vectorstore:
        return []
    try:
        pairs = _vectorstore.similarity_search_with_score(query, k=k)
    except (NotImplementedError, AttributeError):
        docs = _vectorstore.similarity_search(query, k=k)
        pairs = [(d, 0.0) for d in docs]
    out: list[dict[str, Any]] = []
    for d, s in pairs:
        uid = f"vec:{d.metadata.get('pr_number', '')}:{d.metadata.get('chunk_index', '')}"
        out.append(
            {
                "id": uid,
                "score": float(s),
                "page_content": d.page_content,
                "metadata": dict(d.metadata),
                "source": "chroma",
            }
        )
    return out


def _format_context(cites: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for j, c in enumerate(cites, 1):
        m: dict[str, Any] = c.get("metadata", {}) or {}
        prn = m.get("pr_number", "")
        title = m.get("title", "")
        url = m.get("url", "")
        head = f"[{j}] PR #{prn} {title} {url}"
        blocks.append(head.strip() + "\n" + c.get("page_content", ""))
    return "\n\n---\n\n".join(blocks)


def _chat_completions_url_from_base(base: str) -> str:
    b = (base or "").rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


def _log_prefix_hash(messages: list[dict[str, str]]) -> None:
    if not os.environ.get("PREFIX_DEBUG", "").strip():
        return
    pre: list[dict[str, str]] = (
        messages[:-1] if messages and messages[-1].get("role") == "user" else list(messages)
    )
    s = json.dumps(pre, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    print(
        f"[rag_service] PREFIX_DEBUG md5={h} len={len(s)} prefix[0:200]={s[:200]!r}",
        file=sys.stderr,
    )


async def _openai_completions_httpx(
    client: httpx.AsyncClient,
    chat_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    r = await client.post(
        chat_url, json=payload, headers={"Content-Type": "application/json"}
    )
    t1 = time.perf_counter()
    upstream_ms = (t1 - t0) * 1000.0
    r.raise_for_status()
    body = r.json()
    try:
        text: str = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Bad upstream response: {body!r}") from e
    return text, upstream_ms


def _triage_vllm_base() -> str:
    return (os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1") or "").rstrip("/")


def _triage_prefix_pilot_base() -> str:
    return (os.environ.get("PREFIX_PILOT_URL", "http://127.0.0.1:8001") or "").rstrip("/")


async def _run_triage_core(
    client: httpx.AsyncClient, req: TriageRequest, use_prefix_pilot: bool
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cr = _chroma_top(req.query, req.top_k)
    br = _bm25_top(req.query, req.top_k)
    fused = _rrf_fuse([br, cr], top_n=req.top_k)
    sorted_cites = _sort_citations(fused)
    t1 = time.perf_counter()
    context = _format_context(sorted_cites)
    sys_p = req.system_prompt or _SYSTEM_TRIAGE
    user_block = f"""Repository question:\n{req.query}

Context (PR snippets):
{context}
"""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_block},
    ]
    t2 = time.perf_counter()
    _log_prefix_hash(messages)
    if use_prefix_pilot:
        base = _triage_prefix_pilot_base()
    else:
        base = _triage_vllm_base()
    chat_url = _chat_completions_url_from_base(base)
    model = os.environ.get("VLLM_MODEL", "")
    if not model:
        t_end = time.perf_counter()
        return {
            "query": req.query,
            "citations": sorted_cites,
            "answer": None,
            "note": "Set VLLM_MODEL to enable generation; citations still returned.",
            "latency_ms": {
                "retrieval": (t1 - t0) * 1000.0,
                "prompt_assembly": (t2 - t1) * 1000.0,
                "upstream": 0.0,
                "total": (t_end - t0) * 1000.0,
            },
        }
    try:
        text, upstream_ms = await _openai_completions_httpx(
            client, chat_url, model, messages, req.max_tokens, req.temperature
        )
    except httpx.HTTPStatusError as e:
        snippet = (e.response.text or "")[:500]
        raise HTTPException(502, detail=f"Upstream HTTP: {e!s} body={snippet!r}") from e
    t3 = time.perf_counter()
    out: dict[str, Any] = {
        "query": req.query,
        "citations": sorted_cites,
        "answer": text,
        "latency_ms": {
            "retrieval": (t1 - t0) * 1000.0,
            "prompt_assembly": (t2 - t1) * 1000.0,
            "upstream": upstream_ms,
            "total": (t3 - t0) * 1000.0,
        },
    }
    if use_prefix_pilot:
        out["routed_via"] = "prefix_pilot"
    return out


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chroma": _vectorstore is not None,
        "bm25": _bm25 is not None and len(_bm25_docs) > 0,
        "vllm_base": _triage_vllm_base(),
        "prefix_pilot": _triage_prefix_pilot_base(),
    }


@app.post("/v1/retrieve")
def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    cr = _chroma_top(req.query, req.top_k)
    br = _bm25_top(req.query, req.top_k)
    fused = _rrf_fuse([br, cr], top_n=req.top_k)
    return {"query": req.query, "citations": fused, "chroma": cr, "bm25": br}


@app.post("/v1/triage")
async def triage(req: TriageRequest, request: Request) -> dict[str, Any]:
    return await _run_triage_core(request.app.state.http, req, use_prefix_pilot=False)


@app.post("/v1/triage_routed")
async def triage_routed(req: TriageRequest, request: Request) -> dict[str, Any]:
    return await _run_triage_core(request.app.state.http, req, use_prefix_pilot=True)


def main() -> None:
    import uvicorn

    host = os.environ.get("RAG_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_PORT", "8080"))
    uvicorn.run("rag_service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
