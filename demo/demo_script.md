# 3-minute demo script (PRWhisperer)

| Time | What you say & do |
|------|------------------|
| **0:00–0:45** | *Problem:* Maintainers drown in PR noise; vLLM helps answer questions, but **redundant RAG prompts waste prefix cache and GPU** when many similar questions hit the same context. *Solution:* We batch requests that share a long prefix and route them through a small proxy in front of vLLM. |
| **0:45–1:30** | *Live:* Open the Streamlit app, type a **in-corpus** question from `demo_questions.md`, click **Submit to Both**. Narrate: same citations left and right, same prompt shape; the right path goes through **PrefixPilot:8001** (batched) before vLLM. |
| **1:30–2:30** | *Live:* **Run Load Test** (concurrency 50, paraphrase 0.35). Point to **p50** naive vs routed, **multiplier**, and **cache hit (proxy)**. Say workload once: 50 parallel pairs, illustrative paraphrase mix. |
| **2:30–3:00** | *Close:* Repo is public; we offer two upstream PRs: (1) triage RAG example for the vLLM ecosystem, (2) PrefixPilot pattern as a drop-in batching layer. **Optional in 20s:** one **OOC** question → refusal. |

*Reading aloud, target &lt; 3:00. Practice once with a clock.*
