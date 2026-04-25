#!/usr/bin/env python3
"""
PRWhisperer — PrefixPilot: OpenAI-compatible proxy with prefix-hash bucketing for vLLM.

Buckets requests by MD5 of messages prefix (all except the last user message);
flushes on MAX_BUCKET or FLUSH_MS delay. Forwards each request as a separate POST
to vLLM so the engine can reuse the shared prefix in cache.

Env:
  VLLM_URL  — default http://localhost:8000
  FLUSH_MS  — default 50 (ms); raise to 100 for higher avg bucket size
  MAX_BUCKET — default 8
  PREFIX_LOG — set to 1 to log first 200 chars of prefix JSON on every flush
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# --- tunable (Task 5: increase FLUSH_MS to improve avg_bucket_size) ---
VLLM_URL: str = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
FLUSH_MS: int = int(os.environ.get("FLUSH_MS", "50"))
MAX_BUCKET: int = int(os.environ.get("MAX_BUCKET", "8"))


def _vllm_chat_completions_url() -> str:
    b = (os.environ.get("VLLM_URL", "http://localhost:8000") or "").rstrip("/")
    return f"{b}/v1/chat/completions"

log = logging.getLogger("prefix_pilot")
if os.environ.get("PREFIX_LOG", "").strip() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
else:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

_bucket_lock = asyncio.Lock()
_buckets: dict[str, list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]]] = {}
_timer_tasks: dict[str, asyncio.Task[None]] = {}

_metrics_lock = asyncio.Lock()
_queries_served: int = 0
_total_flushes: int = 0
_sum_bucket_size_at_flush: int = 0
_sum_cache_proxy_numer: int = 0


def _build_prefix_string(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "[]"
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        prefix_messages = messages[:-1]
    else:
        prefix_messages = messages
    return json.dumps(prefix_messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _prefix_hash(body: dict[str, Any]) -> str:
    s = _build_prefix_string(body)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(connect=10.0, read=1200.0, write=60.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.http = client
        yield


app = FastAPI(
    title="PRWhisperer PrefixPilot",
    version="1.0.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str = "ok"


class MetricsResponse(BaseModel):
    queries_served: int
    total_flushes: int
    avg_bucket_size_at_flush: float
    cache_hit_rate_proxy: float
    open_buckets: int = Field(
        description="Number of non-empty prefix buckets currently waiting for flush"
    )


async def _post_to_vllm(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> dict[str, Any]:
    r = await client.post(
        _vllm_chat_completions_url(), json=body, headers={"Content-Type": "application/json"}
    )
    r.raise_for_status()
    return r.json()


async def _execute_flush(
    client: httpx.AsyncClient, items: list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]], reason: str
) -> None:
    global _queries_served, _total_flushes, _sum_bucket_size_at_flush, _sum_cache_proxy_numer
    if not items:
        return
    size = len(items)
    if os.environ.get("PREFIX_LOG", "").strip() in ("1", "true", "yes") and items:
        pprev = _build_prefix_string(items[0][0])[:200]
        log.info("flush reason=%s size=%d prefix[0:200]=%r", reason, size, pprev)

    async def one(it: tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]) -> None:
        b, fut = it
        if fut.done():
            return
        try:
            res = await _post_to_vllm(client, b)
            if not fut.done():
                fut.set_result(res)
        except Exception as e:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(e)

    await asyncio.gather(*[one(x) for x in items])

    async with _metrics_lock:
        _queries_served += size
        _total_flushes += 1
        _sum_bucket_size_at_flush += size
        if size > 0:
            _sum_cache_proxy_numer += size - 1


async def _timer_flush(h: str) -> None:
    try:
        await asyncio.sleep(FLUSH_MS / 1000.0)
    except asyncio.CancelledError:
        return
    to_flush: list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]] | None = None
    async with _bucket_lock:
        _timer_tasks.pop(h, None)
        if h in _buckets and _buckets[h]:
            to_flush = _buckets.pop(h, [])
    if to_flush and app.state.http:  # type: ignore[union-attr]
        client = app.state.http
        await _execute_flush(client, to_flush, "time")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    async with _metrics_lock:
        q = _queries_served
        tf = _total_flushes
        sbs = _sum_bucket_size_at_flush
        scn = _sum_cache_proxy_numer
    async with _bucket_lock:
        ob = len(_buckets)
    avg = (sbs / tf) if tf else 0.0
    chit = (scn / q) if q else 0.0
    return MetricsResponse(
        queries_served=q,
        total_flushes=tf,
        avg_bucket_size_at_flush=avg,
        cache_hit_rate_proxy=chit,
        open_buckets=ob,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    try:
        body: dict[str, Any] = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(400, detail=f"Invalid JSON: {e!s}") from e
    if not body.get("messages"):
        raise HTTPException(400, detail="Missing messages")
    bcopy = copy.deepcopy(body)
    h = _prefix_hash(bcopy)
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[dict[str, Any]] = loop.create_future()

    to_run: list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]] | None = None
    client: httpx.AsyncClient = request.app.state.http

    async with _bucket_lock:
        if h not in _buckets:
            _buckets[h] = []
        _buckets[h].append((bcopy, fut))
        n = len(_buckets[h])
        if n >= MAX_BUCKET:
            to_run = _buckets.pop(h, [])
            tsk = _timer_tasks.pop(h, None)
        else:
            tsk = None
            if h not in _timer_tasks:
                _timer_tasks[h] = asyncio.create_task(_timer_flush(h))
            to_run = None

    if tsk and not tsk.done():
        tsk.cancel()
        try:
            await tsk
        except asyncio.CancelledError:
            pass
    if to_run:
        await _execute_flush(client, to_run, "size")

    try:
        out = await fut
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, detail=f"vLLM: {e!s}") from e
    except httpx.RequestError as e:
        raise HTTPException(502, detail=f"Upstream: {e!s}") from e
    except Exception as e:  # noqa: BLE001
        if isinstance(e, (httpx.HTTPError, OSError, ValueError)):
            raise HTTPException(502, detail=str(e)) from e
        raise
    return JSONResponse(out)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PREFIX_PILOT_PORT", "8001"))
    host = os.environ.get("PREFIX_PILOT_HOST", "0.0.0.0")
    print(f"PrefixPilot VLLM_URL={VLLM_URL!r} FLUSH_MS={FLUSH_MS} MAX_BUCKET={MAX_BUCKET}", file=sys.stderr)
    uvicorn.run("prefix_pilot:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
