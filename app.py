#!/usr/bin/env python3
"""
PRWhisperer — two-column Streamlit: /v1/triage (naive) vs /v1/triage_routed (PrefixPilot).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import streamlit as st

_DEFAULT_RAG = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
_DEFAULT_PREFIX_METRICS = os.environ.get("PREFIX_PILOT_METRICS", "http://127.0.0.1:8001").rstrip("/")
_JSONL = Path(__file__).resolve().parent / "loadtest_results.jsonl"


@st.cache_resource
def _httpx_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(10.0, read=1200.0), follow_redirects=True)


def _post(path: str, body: dict) -> dict:
    c = _httpx_client()
    base = st.session_state.get("rag", _DEFAULT_RAG).rstrip("/")
    r = c.post(f"{base}{path}", json=body, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict:
    c = _httpx_client()
    base = st.session_state.get("rag", _DEFAULT_RAG).rstrip("/")
    r = c.get(f"{base}{path}")
    r.raise_for_status()
    return r.json()


def _get_prefix_metrics() -> dict:
    c = _httpx_client()
    base = st.session_state.get("prefix_m", _DEFAULT_PREFIX_METRICS).rstrip("/")
    r = c.get(f"{base}/metrics")
    if r.status_code != 200:
        return {}
    return r.json()


def _read_jsonl_chart() -> tuple[list[float], list[float], list[str]]:
    naive_s: list[float] = []
    routed_s: list[float] = []
    ts: list[str] = []
    if not _JSONL.is_file():
        return naive_s, routed_s, ts
    for line in _JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            naive_s.append(float(o.get("naive_p50", 0)))
            routed_s.append(float(o.get("routed_p50", 0)))
            ts.append(str(o.get("timestamp", ""))[:19])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return naive_s, routed_s, ts


st.set_page_config(page_title="PRWhisperer", layout="wide")
if "rag" not in st.session_state:
    st.session_state["rag"] = _DEFAULT_RAG
if "prefix_m" not in st.session_state:
    st.session_state["prefix_m"] = _DEFAULT_PREFIX_METRICS
if "query" not in st.session_state:
    st.session_state["query"] = "What does vLLM prefix caching do?"

st.title("PRWhisperer⚡ Triage RAG")
st.caption("Left: direct vLLM. Right: PrefixPilot (batched) → vLLM.")

with st.sidebar:
    st.subheader("Endpoints")
    st.text_input("RAG base URL", key="rag")
    st.text_input("PrefixPilot metrics", key="prefix_m")
    try:
        h = _get("/health")
        st.success(
            f"RAG: **ok** vLLM `{h.get('vllm_base', '')}` · Prefix `{h.get('prefix_pilot', '')}`"
        )
    except httpx.RequestError as e:
        st.error(f"RAG not reachable: {e!s}")
    st.subheader("Model params")
    top_k = st.slider("top_k", 3, 20, 8)
    max_tokens = st.number_input("max_tokens", 64, 2048, 256, 32)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    st.subheader("Load test")
    lt_c = st.slider("concurrency", 1, 50, 50)
    lt_p = st.slider("paraphrase ratio", 0.0, 1.0, 0.35, 0.05)

q = st.text_area("Question", key="query", height=100)
if st.button("Submit to Both", type="primary", use_container_width=True):
    body = {
        "query": (q or "").strip(),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if not body["query"]:
        st.warning("Enter a non-empty question.")
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Naive (direct vLLM)")
            try:
                rna = _post("/v1/triage", body)
            except httpx.RequestError as e:
                st.exception(e)
            else:
                st.write(rna.get("answer", "") or rna.get("note", "—"))
                st.json(rna.get("latency_ms", {}))
                for i, cit in enumerate(rna.get("citations", []), 1):
                    m = (cit.get("metadata") or {})
                    st.expander(
                        f"[{i}] PR #{m.get('pr_number', '?')}"
                    ).markdown(cit.get("page_content", "")[:12000] or "—")
        with c2:
            st.subheader("PRWhisperer⚡ Routed (PrefixPilot)")
            try:
                rro = _post("/v1/triage_routed", body)
            except httpx.RequestError as e:
                st.exception(e)
            else:
                st.write(rro.get("answer", "") or rro.get("note", "—"))
                st.json(rro.get("latency_ms", {}))
                if "routed_via" in rro:
                    st.caption(f"routed via: {rro['routed_via']}")
                for i, cit in enumerate(rro.get("citations", []), 1):
                    m = (cit.get("metadata") or {})
                    st.expander(
                        f"[{i}] PR #{m.get('pr_number', '?')}"
                    ).markdown(cit.get("page_content", "")[:12000] or "—")

st.subheader("Last load test (Run Load Test to refresh)")
if "last_loadtest_log" not in st.session_state:
    st.session_state["last_loadtest_log"] = ""
if st.button("Run Load Test", use_container_width=True):
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "loadtest.py"),
        "--concurrency",
        str(lt_c),
        "--paraphrase-ratio",
        str(lt_p),
        "--duration",
        "120",
        "--phase-gap",
        "5",
        "--rag-base",
        st.session_state.get("rag", _DEFAULT_RAG).rstrip("/"),
        "--prefix-metrics",
        st.session_state.get("prefix_m", _DEFAULT_PREFIX_METRICS).rstrip("/"),
    ]
    with st.spinner("Running loadtest: naive 50, pause 5s, routed 50 (may take a few minutes)…"):
        p = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
        )
    st.session_state["last_loadtest_log"] = (p.stdout or p.stderr or "(no output)").strip()
    st.session_state["last_loadtest_exit"] = p.returncode
    if p.returncode != 0:
        st.error(f"loadtest exit {p.returncode}")
if st.session_state.get("last_loadtest_log"):
    st.code(st.session_state["last_loadtest_log"][: 50000], language="text")

met = _get_prefix_metrics()
naive_s, routed_s, tss = _read_jsonl_chart()
last_n = naive_s[-1] if naive_s else None
last_r = routed_s[-1] if routed_s else None
last_mult = None
if _JSONL.is_file():
    for line in reversed(_JSONL.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                o = json.loads(line)
                last_mult = o.get("multiplier")
            except json.JSONDecodeError:
                pass
            break

rowm = st.columns(4)
rowm[0].metric("Naive p50 (ms)", f"{last_n:.0f}" if last_n is not None else "—")
rowm[1].metric("Routed p50 (ms)", f"{last_r:.0f}" if last_r is not None else "—")
rowm[2].metric("Throughput mult.", f"{last_mult:.2f}x" if last_mult is not None else "—")
rowm[3].metric("Cache hit (proxy)", f"{met.get('cache_hit_rate_proxy', 0.0) * 100:.1f}%")

if len(naive_s) >= 2 and len(routed_s) >= 2:
    chart = {
        "naive_p50": naive_s,
        "routed_p50": routed_s,
    }
    st.line_chart(chart, height=220)
else:
    st.caption("Run at least two load tests to show trend lines in the chart above.")

st.divider()
st.caption("Pipeline: vLLM:8000 ← PrefixPilot:8001 ← rag:8080 ← this UI. CLI experiment: `prefix_pilot_experiment.py`.")
