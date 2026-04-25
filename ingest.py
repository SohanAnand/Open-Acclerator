#!/usr/bin/env python3
"""
PRWhisperer — ingest GitHub pull requests into Chroma (vectors) and a parallel BM25 index.

Run first in the pipeline. Requires network access to GitHub and HuggingFace (embeddings).

Environment:
  GITHUB_TOKEN   — required for private repos and higher rate limits
  GITHUB_REPO    — default "owner/name" if --repo omitted
  CHROMA_DIR     — default ./data/chroma (Chroma persist directory)
  EMBED_MODEL    — default sentence-transformers/all-MiniLM-L6-v2
  BM25_PKL       — optional; default {parent of CHROMA_DIR}/bm25.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

# --- deps: langchain-chroma, chromadb, PyGithub, sentence-transformers, langchain-core
from github import Auth, Github
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def _chunk_text_sliding(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Non-overlapping step = chunk_size - overlap; simple and deterministic."""
    if not text.strip():
        return []
    step = max(1, chunk_size - max(0, min(chunk_overlap, chunk_size - 1)))
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + chunk_size]
        if piece.strip():
            chunks.append(piece)
        i += step
    return chunks


def split_docs(docs: Iterable[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    result: list[Document] = []
    for d in docs:
        chunks = _chunk_text_sliding(
            d.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        n = max(1, len(chunks))
        for i, c in enumerate(chunks):
            meta = dict(d.metadata)
            meta["chunk_index"] = i
            meta["chunk_total"] = n
            result.append(Document(page_content=c, metadata=meta))
    return result


def build_documents_from_prs(
    g: Github, repo_name: str, max_prs: int, include_comments: int
) -> list[Document]:
    repo = g.get_repo(repo_name)
    pulls = repo.get_pulls(state="all", sort="updated", direction="desc")
    out: list[Document] = []
    count = 0
    for pr in pulls:
        if count >= max_prs:
            break
        count += 1
        try:
            body = pr.body or ""
            title = pr.title or ""
            labels = ",".join(l.name for l in pr.labels)
            try:
                try:
                    issue = pr.as_issue() if hasattr(pr, "as_issue") else None
                except Exception:
                    issue = None
                if issue is None:
                    issue = repo.get_issue(pr.number)
                ic = list(issue.get_comments()[:include_comments])
                comments_text = "\n---\n".join(c.body or "" for c in ic)
            except Exception:
                comments_text = ""
            try:
                review_bodies = "\n".join(
                    (rv.body or "")
                    for rv in list(pr.get_reviews()[: 3 * include_comments])
                    if rv.body
                )
            except Exception:
                review_bodies = ""
            merged = pr.merged
            text = f"""# PR #{pr.number}: {title}
State: {pr.state} merged={merged} labels={labels}
Author: {getattr(pr.user, "login", "")}

{body}

### Issue comments
{comments_text}

### Reviews (truncated)
{review_bodies}
"""
            meta = {
                "pr_number": pr.number,
                "title": title[:500],
                "state": pr.state,
                "merged": bool(merged),
                "labels": labels[:2000],
                "url": pr.html_url,
                "github_repo": repo_name,
            }
            out.append(Document(page_content=text, metadata=meta))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip PR fetch error: {e}", file=sys.stderr)
    return out


def persist_ingest_sidecar(
    chroma_dir: Path, repo: str, split_docs: list[Document], bm25_pkl: Path
) -> None:
    """Writes data/bm25.pkl (and optional bm25_corpus.json) with Chroma under e.g. data/chroma/."""
    ser = {
        "version": 1,
        "github_repo": repo,
        "chunks": [
            {
                "page_content": d.page_content,
                "metadata": d.metadata,
            }
            for d in split_docs
        ],
    }
    chroma_dir.mkdir(parents=True, exist_ok=True)
    bm25_pkl.parent.mkdir(parents=True, exist_ok=True)
    bm25_pkl.write_bytes(pickle.dumps(ser, protocol=pickle.HIGHEST_PROTOCOL))
    (bm25_pkl.parent / "bm25_corpus.json").write_text(
        json.dumps(ser, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {bm25_pkl} (+ bm25_corpus.json)")


def run_ingest(
    repo: str,
    token: str | None,
    chroma_dir: str,
    embed_model: str,
    max_prs: int,
    include_comments: int,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
    reset: bool,
    bm25_pkl: Path,
) -> None:
    if not token:
        print("Warning: GITHUB_TOKEN empty — public data only, stricter rate limits.", file=sys.stderr)
    auth = Auth.Token(token) if token else None
    g = Github(auth=auth) if auth else Github()
    print(f"Fetching PRs from {repo} (max {max_prs})...")
    raw = build_documents_from_prs(g, repo, max_prs=max_prs, include_comments=include_comments)
    if not raw:
        print("No PRs retrieved. Check repo name and token.", file=sys.stderr)
        sys.exit(1)
    print(f"Splitting {len(raw)} PR documents...")
    parts = split_docs(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for i, d in enumerate(parts):
        d.metadata["chunk_id"] = i
    print(f"Chunks: {len(parts)}")

    out_path = Path(chroma_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    persist_ingest_sidecar(out_path, repo, parts, bm25_pkl)

    print(f"Embedding with {embed_model} ...")
    dev = os.environ.get("EMBED_DEVICE", "cpu")
    emb = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True},
    )

    if reset and out_path.exists():
        import shutil

        for p in out_path.iterdir():
            if p.name in ("chroma.sqlite3",) or p.suffix in (".bin", ".sqlite3", ".index"):
                try:
                    p.unlink() if p.is_file() else shutil.rmtree(p)
                except OSError:
                    pass
    # Chroma 0.4+ persistence: pass persist_directory
    _ = Chroma.from_documents(
        documents=parts,
        embedding=emb,
        collection_name=collection_name,
        persist_directory=str(out_path),
    )
    # Explicit persist for older client patterns
    try:
        _.persist()  # type: ignore[attr-defined]
    except Exception:
        pass
    print(f"Chroma collection '{collection_name}' written under {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest GitHub PRs into Chroma + BM25 sidecar")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPO", "microsoft/vscode"), help="owner/repo")
    p.add_argument(
        "--chroma-dir",
        default=os.environ.get("CHROMA_DIR", "./data/chroma"),
        help="Chroma persist directory",
    )
    p.add_argument(
        "--bm25-pkl",
        default=os.environ.get("BM25_PKL", ""),
        help="BM25 payload path (default: <parent of chroma-dir>/bm25.pkl)",
    )
    p.add_argument(
        "--embed-model",
        default=os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    p.add_argument("--max-prs", type=int, default=200)
    p.add_argument("--include-comments", type=int, default=5, help="max issue comments + reviews to pull per PR")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--collection", default="pr_triage", dest="collection_name")
    p.add_argument("--reset", action="store_true", help="clear chroma files before write (best effort)")
    args = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    cp = Path(args.chroma_dir)
    bp = Path(args.bm25_pkl) if args.bm25_pkl else (cp.parent / "bm25.pkl")
    run_ingest(
        repo=args.repo,
        token=token or None,
        chroma_dir=args.chroma_dir,
        embed_model=args.embed_model,
        max_prs=args.max_prs,
        include_comments=args.include_comments,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name,
        reset=args.reset,
        bm25_pkl=bp,
    )


if __name__ == "__main__":
    main()
