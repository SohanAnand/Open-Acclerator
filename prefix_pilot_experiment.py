#!/usr/bin/env python3
"""
PRWhisperer — prefix-cache-aware routing pilot for vLLM.

Sends the same set of /v1/chat/completions jobs in two orderings:
  1) shuffled (cache-unfriendly)
  2) grouped by shared RAG "context" prefix (long identical prefix before the question)

This helps validate that your deployment benefits from vLLM automatic prefix caching when
RAG context blocks repeat across user turns (e.g. same cluster of PRs).

Environment (same as rag_service for LLM):
  VLLM_BASE_URL  — default http://127.0.0.1:8000/v1
  VLLM_MODEL     — required for requests
  PILOT_N        — optional, default 12 tasks
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Task:
    context_id: str
    rag_context: str
    question: str

    def messages(self, system: str) -> list[dict[str, str]]:
        # Same shape as rag_service: system, then user with context+question
        user = f"""Repository question:
{self.question}

Context (PR snippets):
{self.rag_context}
"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def _vllm_chat_url() -> str:
    b = (os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1") or "").rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return f"{b}/chat/completions"


def _one_completion(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> dict[str, Any]:
    url = _vllm_chat_url()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    return {"latency_s": t1 - t0, "raw": body}


def _lorem_block(n_words: int = 200) -> str:
    w = "lorem ipsum dolor sit amet consectetur adipiscing elit".split()
    return " ".join(w[i % len(w)] for i in range(n_words))


def _build_synthetic_tasks(n: int) -> list[Task]:
    """Reuses a few long synthetic 'RAG' contexts so prefix routing has something to group."""
    blocks = {
        "A": _lorem_block(120) + "\n" + "Focus: performance regressions, latency.",
        "B": _lorem_block(120) + "\n" + "Focus: security, auth, cookies.",
        "C": _lorem_block(120) + "\n" + "Focus: API contracts, breaking changes.",
    }
    questions = [
        "What PRs look risky to merge this week?",
        "Summarize test gaps from prior reviews.",
        "Which labels would you apply and why?",
        "Any dependency upgrade concerns?",
    ]
    tasks: list[Task] = []
    for i in range(n):
        k = list(blocks.keys())[i % 3]
        q = questions[i % len(questions)]
        tasks.append(
            Task(
                context_id=k,
                rag_context=blocks[k],
                question=f"{q} (task {i + 1})",
            )
        )
    return tasks


def _run_seq(name: str, order: list[Task], system: str, model: str) -> dict[str, Any]:
    latencies: list[float] = []
    t0 = time.perf_counter()
    for t in order:
        r = _one_completion(model, t.messages(system))
        latencies.append(r["latency_s"])
    t1 = time.perf_counter()
    return {
        "name": name,
        "wall_s": t1 - t0,
        "per_request_s": latencies,
        "mean_s": sum(latencies) / max(1, len(latencies)),
    }


def main() -> None:
    model = os.environ.get("VLLM_MODEL", "")
    if not model:
        print("Set VLLM_MODEL to a model served by vLLM.", file=sys.stderr)
        sys.exit(1)
    n = int(os.environ.get("PILOT_N", "12"))
    system = os.environ.get(
        "PILOT_SYSTEM",
        "You are a brief assistant. One short paragraph. Use context only if relevant.",
    )
    random.seed(0)
    tasks = _build_synthetic_tasks(n)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    # Group by context_id: same long RAG block → high prefix sharing when consecutive
    grouped = sorted(tasks, key=lambda x: (x.context_id, x.question))

    print(f"Model={model!r} url={_vllm_chat_url()!r} n={n}")
    print("--- run A: shuffled (random) order")
    a = _run_seq("shuffled", shuffled, system, model)
    print(json.dumps(a, indent=2)[: 4000])
    time.sleep(2)
    print("--- run B: prefix-grouped (by shared RAG context) order")
    b = _run_seq("grouped", grouped, system, model)
    print(json.dumps(b, indent=2)[: 4000])
    ar, br = a["wall_s"], b["wall_s"]
    if ar > 0 and br > 0:
        pct = 100.0 * (1.0 - (min(ar, br) / max(ar, br)))
        faster = "grouped" if br < ar else "shuffled"
        print(
            f"\nWall time: shuffled={ar:.2f}s grouped={br:.2f}s "
            f"(~{pct:.1f}% relative gap; faster run: {faster})"
        )
    print(
        "\nNote: vLLM prefix caching, batching, and load vary; "
        "run on an idle server and match max_model_len to your RAG prompt sizes."
    )


if __name__ == "__main__":
    main()
