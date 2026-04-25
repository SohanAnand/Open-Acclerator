# PRWhisperer

**PRWhisperer** is a vLLM-backed **GitHub PR triage RAG**: hybrid **Chroma + BM25** retrieval, optional **PrefixPilot** proxy to improve prefix-cache locality under load, and a **Streamlit** dashboard to compare “naive” (direct vLLM) vs “routed” (via PrefixPilot) on the **same** prompts. The index is **file-backed** (no per-vector cloud bill).

*Illustrative numbers below depend on your GPU, model, and vLLM flags; re-run `loadtest.py` to reproduce.*

**Three sentences:** (1) Ingest pull requests into a local vector + lexical index. (2) Answer questions with **cited** chunks and optional refusals when context is missing. (3) Under concurrent load, batch **shared-prefix** OpenAI calls through **PrefixPilot** on **:8001** so vLLM can reuse the KV cache more effectively.

## Architecture (ASCII)

```
  Streamlit :8501
       │  HTTP
       ▼
  rag_service :8080  ──POST /v1/triage  (naive)──────────────┐
       │  same retrieval                                      │
       │  POST /v1/triage_routed ──►  PrefixPilot :8001       │
       │                                 │  batched           │
       │                                 ▼  per-request POSTs │
       └──────────────────────────►  vLLM :8000  (Qwen, etc.) ◄┘
```

## Port table

| Service       | Port | Notes |
|---------------|------|--------|
| vLLM (OpenAI) | 8000 | Do not hard-restart during tuning if users asked not to. |
| PrefixPilot   | 8001 | `/v1/chat/completions`, `/health`, `/metrics` |
| RAG (FastAPI) | 8080 | `/v1/retrieve`, `/v1/triage`, `/v1/triage_routed` |
| Streamlit     | 8501 | `streamlit run app.py` |

## Run order (5 commands)

1. **vLLM** (user-managed): e.g. `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000` with your flags (e.g. `--enable-prefix-caching`).

2. **Ingest (once per corpus):**  
   `export CHROMA_DIR=./chroma_prwhisperer GITHUB_REPO=org/repo`  
   `python ingest.py --chroma-dir "$CHROMA_DIR" --max-prs 200`

3. **PrefixPilot:**  
   `python -m uvicorn prefix_pilot:app --host 0.0.0.0 --port 8001`

4. **RAG:**  
   `export VLLM_MODEL=... VLLM_BASE_URL=http://127.0.0.1:8000/v1 PREFIX_PILOT_URL=http://127.0.0.1:8001 CHROMA_DIR=...`  
   `python -m uvicorn rag_service:app --host 0.0.0.0 --port 8080`

5. **UI:**  
   `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`

Or: `bash scripts/reproduce_demo.sh` (expects vLLM up; may start 8001/8080 and run one load test).

## Workload & disclosure (example)

- **loadtest** runs **50 `/v1/triage` (naive)**, waits **5s**, then **50 `/v1/triage_routed` (routed)** with the **same** query set so the two paths do **not** compete for the same vLLM GPU in one wave. Concurrency is per phase; **~35%** paraphrase mix; RAG user prompts in the **~1–2k token** class depending on `top_k` and chunking.
- **Tuning (Task 5):** set `PREFIX_DEBUG=1` in `rag_service` to log prefix `md5`; raise `FLUSH_MS` to **100** for PrefixPilot; try `--paraphrase-ratio 0.5` on `loadtest.py` to increase structural overlap. Target **higher** `avg_bucket_size_at_flush` and vLLM’s own **prefix cache** log line.
- **Example illustrative outcome** (fill in after your runs): *naive p50 X ms, routed p50 Y ms, multiplier Z×, proxy cache hit P%.*

## Two-line PrefixPilot adoption

```bash
export PREFIX_PILOT_URL=http://127.0.0.1:8001
# In your RAG app, POST chat/completions to http://127.0.0.1:8001/v1/chat/completions instead of vLLM :8000.
```

## Files

- `ingest.py`, `rag_service.py`, `prefix_pilot.py` (proxy), `prefix_pilot_experiment.py` (CLI A/B), `loadtest.py`, `app.py`
- `demo/` — `demo_questions.md`, `demo_script.md`
- `docs/` — draft upstream blurbs

## Credits

PRWhisperer is a team / hackathon build; vLLM and dependencies are their respective projects.
