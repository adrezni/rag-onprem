"""
Stage-by-stage performance benchmark for the RAG pipeline.

Times each stage of a query so we can see where the wall-clock goes:
embedding-model load, Chroma open, retrieval, and LLM generation.

Run:  .venv/bin/python benchmark.py
"""
import time

QUESTION = "What is the penalty for a lost ball?"


def stage(label, fn):
    """Run fn(), print how long it took, and return its result."""
    t = time.time()
    result = fn()
    print(f"{label:<28} {time.time() - t:6.2f}s")
    return result


def main():
    from rag_engine import get_embeddings, load_vector_store, get_rag_chain

    print(f"Question: {QUESTION}\n")

    # ── One-time setup stages (cached after the first query in the app) ──
    stage("embed model load", get_embeddings)
    vs = stage("open chroma", load_vector_store)

    # ── Per-query stages ──
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = stage("retrieval (k=3)", lambda: retriever.invoke(QUESTION))
    ctx_chars = sum(len(d.page_content) for d in docs)
    print(f"{'  context fed to LLM':<28} {ctx_chars} chars (~{ctx_chars // 4} tokens)")

    chain = get_rag_chain()
    out = stage("LLM generation", lambda: chain.invoke({"query": QUESTION}))
    ans = out["result"]
    print(f"{'  answer length':<28} {len(ans)} chars (~{len(ans) // 4} tokens)")


if __name__ == "__main__":
    main()
