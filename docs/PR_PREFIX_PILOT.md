# [Draft] Blog / small PR: PrefixPilot — batch OpenAI requests by RAG prefix for vLLM

## Motivation (cache waste)

RAG services often build **huge, shared context** in the first part of a chat, then only change the last user line. When **many** clients hit the same server, **arrival order** of unrelated prompts can break **prefix cache locality**. A tiny **OpenAI-compatible proxy** in front of vLLM can **coalesce** requests that share the same **prefix** (all messages before the last user turn) and **forward them back-to-back** as individual `/v1/chat/completions` calls so the engine can reuse KV for that prefix.

**This is a deployment pattern, not a model change.** Numbers depend on your hardware, vLLM version, and flags (`--enable-prefix-caching`, batch limits, etc.).

## 90-line implementation idea

- Compute `md5` of a **stable JSON** dump of the **prefix** messages (omit the last `user` turn).
- Append each incoming request to `bucket[hash]`.
- **Flush** when the bucket has **8** items or after **~50ms** (tunable) so small bursts still batch.
- On flush, `asyncio.gather` of **independent** `httpx` POSTs to vLLM (one per user message).
- Expose `GET /metrics` with: `cache_hit_rate_proxy = sum(flush_size - 1) / queries_served` and `avg_bucket_size_at_flush` as rough batching health checks.

**Two-line integration (conceptual):** Point your RAG service’s `chat_completions` base URL to `http://prefix-pilot:8001` instead of vLLM’s `:8000`, keep the same request JSON.

## Integration snippet

```python
# Before: VLLM_BASE_URL=http://localhost:8000/v1
# After:  PREFIX_PILOT_URL=http://localhost:8001  (RAG app posts chat completions to PrefixPilot)
os.environ["PREFIX_PILOT_URL"] = "http://127.0.0.1:8001"
```

## Demo numbers (illustrative; disclose everything)

- **Work disclosure:** 50 concurrent “query pairs” (naive vs routed) over your stack, paraphrase ratio ~0.35, Qwen2.5-7B-class model, local GPU.
- **Caveat:** The proxy metric is **illustrative**; compare with vLLM’s own prefix-cache hit logs for ground truth. Use results to **tune** `FLUSH_MS` and your RAG’s **citation sort order** for stable prompt prefixes.

## Close

If useful, this can live as a **gist + diagram** in vLLM discussions or a short **“deployment patterns”** note—no need to merge heavy code if the project prefers a link-out.
