#!/usr/bin/env python3
"""
PRWhisperer — load test: run all naive /v1/triage first, wait, then all /v1/triage_routed.
Avoids GPU contention between direct vLLM and PrefixPilot→vLLM in the same window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpx

BASE_QUERIES: list[str] = [
    "What does vLLM prefix caching do and when does it help?",
    "What is --max-num-seqs and how does it affect batching with prefix cache?",
    "How does AWQ weight quantization work with vLLM?",
    "Explain tensor parallelism in distributed LLM serving.",
    "What causes OOM on long contexts in vLLM and how to fix it?",
    "How are chat templates applied to the tokenizer in HuggingFace models?",
    "What is continuous batching in vLLM compared to static batching?",
    "What does the OpenAI compatible API serve at /v1/chat/completions?",
    "What are common causes of high GPU memory usage with a loaded model?",
    "How can I reduce time-to-first-token in a RAG system over vLLM?",
]

TEMPLATES: list[str] = [
    "how do I {q}",
    "why does {q}",
    "what's wrong when {q}",
    "explain: {q}",
]


def _build_50(paraphrase_ratio: float, seed: int) -> list[str]:
    random.seed(seed)
    out: list[str] = []
    for _ in range(50):
        if random.random() < paraphrase_ratio:
            base = random.choice(BASE_QUERIES)
            out.append(random.choice(TEMPLATES).format(q=base))
        else:
            out.append(random.choice(BASE_QUERIES))
    return out


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def _stats_for_results(
    results: list[dict[str, Any]], phase_t0: float, phase_t1: float
) -> tuple[float, float, float, int, list[float]]:
    """p50, p95, throughput (ok/s / wall of phase), n_ok, latencies ms."""
    ok = [r for r in results if r.get("ok") and r.get("ttft_ms") is not None]
    lats = [float(x["ttft_ms"]) for x in ok]
    p50 = _pctl(lats, 50) if lats else 0.0
    p95 = _pctl(lats, 95) if lats else 0.0
    span = max(1e-9, phase_t1 - phase_t0)
    tput = len(ok) / span
    return p50, p95, tput, len(ok), lats


async def _one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    path: str,
    base_url: str,
    q: str,
    top_k: int,
    max_tokens: int,
    temp: float,
) -> dict[str, Any]:
    async with sem:
        t_send = time.perf_counter()
        try:
            r = await client.post(
                f"{base_url}{path}",
                json={
                    "query": q,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                },
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:  # noqa: BLE001
            return {
                "endpoint": path,
                "ok": False,
                "error": str(e),
                "ttft_ms": None,
                "finished_at": time.perf_counter(),
            }
        t_done = time.perf_counter()
        try:
            data = r.json()
        except json.JSONDecodeError:
            return {
                "endpoint": path,
                "ok": False,
                "error": "bad json",
                "ttft_ms": None,
                "finished_at": t_done,
            }
        if r.status_code != 200:
            return {
                "endpoint": path,
                "ok": False,
                "error": str(data)[:200] if isinstance(data, dict) else str(r.status_code),
                "ttft_ms": None,
                "finished_at": t_done,
            }
        lat = data.get("latency_ms") or {}
        total_ms = float(lat.get("total", (t_done - t_send) * 1000.0))
        return {
            "endpoint": path,
            "ok": True,
            "ttft_ms": total_ms,
            "finished_at": t_done,
        }


async def _run_phase(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    path: str,
    base_url: str,
    queries: list[str],
    top_k: int,
    max_tokens: int,
    temp: float,
    per_phase_timeout: float,
) -> tuple[list[dict[str, Any]], float, float]:
    """Returns (results, t_start, t_end) wall times for the whole phase."""
    t_start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _one(client, sem, path, base_url, q, top_k, max_tokens, temp)
        )
        for q in queries
    ]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=per_phase_timeout,
        )
    except TimeoutError:
        for t_ in tasks:
            t_.cancel()
        raise
    t_end = time.perf_counter()
    return (list(results), t_start, t_end)


async def _run(
    base_url: str,
    prefix_metrics_url: str,
    queries: list[str],
    concurrency: int,
    per_phase_timeout: float,
    pr_ratio: float,
    phase_gap_s: float,
    top_k: int,
    max_tokens: int,
    temp: float,
) -> None:
    to = httpx.Timeout(10.0, read=1200.0)
    async with httpx.AsyncClient(timeout=to) as client:
        sem = asyncio.Semaphore(concurrency)
        t_wall0 = time.perf_counter()
        try:
            naive_res, t_n0, t_n1 = await _run_phase(
                client,
                sem,
                "/v1/triage",
                base_url,
                queries,
                top_k,
                max_tokens,
                temp,
                per_phase_timeout,
            )
        except TimeoutError:
            print(json.dumps({"error": "loadtest timeout (naive phase)"}, indent=2))
            return
        print(f"--- Naive phase done: {len(queries)} requests, wall {t_n1 - t_n0:.2f}s", flush=True)

        await asyncio.sleep(phase_gap_s)
        t_gap = time.perf_counter()

        try:
            routed_res, t_r0, t_r1 = await _run_phase(
                client,
                sem,
                "/v1/triage_routed",
                base_url,
                queries,
                top_k,
                max_tokens,
                temp,
                per_phase_timeout,
            )
        except TimeoutError:
            print(json.dumps({"error": "loadtest timeout (routed phase)"}, indent=2))
            return
        print(f"--- Routed phase done: {len(queries)} requests, wall {t_r1 - t_r0:.2f}s", flush=True)

        cache_hr = 0.0
        try:
            mr = await client.get(f"{prefix_metrics_url.rstrip('/')}/metrics")
            if mr.status_code == 200:
                cache_hr = float(mr.json().get("cache_hit_rate_proxy", 0.0))
        except httpx.RequestError:
            pass

    t_wall1 = time.perf_counter()
    wall = t_wall1 - t_wall0

    naive_p50, naive_p95, naive_tput, naive_ok, _ = _stats_for_results(naive_res, t_n0, t_n1)
    routed_p50, routed_p95, routed_tput, routed_ok, _ = _stats_for_results(routed_res, t_r0, t_r1)
    mult = (routed_tput / naive_tput) if naive_tput > 0 else 0.0

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "sequential_naive_then_routed",
        "concurrency": concurrency,
        "paraphrase_ratio": pr_ratio,
        "phase_gap_s": phase_gap_s,
        "naive_p50": naive_p50,
        "naive_p95": naive_p95,
        "routed_p50": routed_p50,
        "routed_p95": routed_p95,
        "naive_wall_s": t_n1 - t_n0,
        "routed_wall_s": t_r1 - t_r0,
        "naive_throughput": naive_tput,
        "routed_throughput": routed_tput,
        "multiplier": mult,
        "total_wall_s": wall,
        "cache_hit_rate_proxy": cache_hr,
        "naive_ok": naive_ok,
        "routed_ok": routed_ok,
    }
    with open("loadtest_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print("---")
    print(
        f"naive:  p50={naive_p50:.1f}ms p95={naive_p95:.1f}ms  wall={t_n1 - t_n0:.2f}s  tput~{naive_tput:.2f}q/s  ok={naive_ok}\n"
        f"routed: p50={routed_p50:.1f}ms p95={routed_p95:.1f}ms  wall={t_r1 - t_r0:.2f}s  tput~{routed_tput:.2f}q/s  ok={routed_ok}\n"
        f"throughput mult (routed/naive) = {mult:.2f}x   cache_hit_rate_proxy~{cache_hr:.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run 50 naive triage, pause, then 50 routed (same queries) to avoid GPU cross-contention."
    )
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--paraphrase-ratio", type=float, default=0.35)
    p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Max seconds per phase (naive and routed each); default 120",
    )
    p.add_argument(
        "--phase-gap",
        type=float,
        default=5.0,
        help="Seconds to sleep between naive and routed phases (default 5)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--rag-base",
        default=os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8080").rstrip("/"),
    )
    p.add_argument(
        "--prefix-metrics",
        default=os.environ.get("PREFIX_PILOT_METRICS", "http://127.0.0.1:8001").rstrip("/"),
    )
    args = p.parse_args()
    pr = float(args.paraphrase_ratio)
    queries = _build_50(pr, args.seed)
    asyncio.run(
        _run(
            args.rag_base,
            args.prefix_metrics,
            queries,
            args.concurrency,
            float(args.duration),
            pr,
            float(args.phase_gap),
            args.top_k,
            args.max_tokens,
            args.temperature,
        )
    )


if __name__ == "__main__":
    main()
