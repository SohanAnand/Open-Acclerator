# [Draft] PR: Example maintainer-triage RAG for the vLLM community

**Title:** *Docs or examples: RAG triage on GitHub PRs with hybrid retrieval + vLLM*

## Motivation

Open-source maintainers triage a large volume of pull requests. A **retrieval-first** assist (no training) can ground answers in **actual PR text** (description, review discussion) and refuse when context is missing. vLLM’s OpenAI-compatible server makes it easy to deploy the generator. This project (**PRWhisperer**) is a small, self-contained example: `ingest` from GitHub → **Chroma + BM25** (file-backed) → **FastAPI** → optional vLLM.

## What’s included (conceptual)

- **ingest** — Pull PRs, chunk, embed, persist to local Chroma + BM25 sidecar (no paid vector DB).
- **RAG service** — Hybrid (dense + BM25) with RRF fusion, `/v1/triage` for cited answers, `/v1/retrieve` for raw citations.
- **Streamlit** — Simple UI for a live demo; load-test harness compares naive vs a prefix-batching path.

**Disclaimer:** Citation quality depends on your repo and questions; it is a **triage assist**, not a replacement for review.

## Deployment sketch

- One GPU box: **vLLM** on :8000, **RAG** on :8080, optional **PrefixPilot** on :8001, **Streamlit** on :8501.
- **Env:** `CHROMA_DIR`, `VLLM_MODEL`, `VLLM_BASE_URL`, `GITHUB_TOKEN` (for private or rate limits).

## Demo numbers (illustrative, hardware-dependent)

On a 7B-class model, **~50 concurrent** question pairs, **~35% paraphrase** mix, **~2k-token-class** RAG user prompts, we observed (your numbers may differ; disclose hardware and vLLM flags):

- Lower **p50 latency** and higher **effective throughput** on the **routed** path when the proxy groups shared-prefix work.
- **PrefixPilot “cache hit (proxy)”** is a **proxy metric** (bucket reuse), not vLLM’s internal counter.

Always report **workload, model, and flags** (e.g. `--enable-prefix-caching`, `--max-num-seqs`).

## Close

We’re happy to **trim this to a minimal doc + script** in `examples/` or link out to the standalone repo, whichever fits the vLLM docs style.
