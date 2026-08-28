"""Audit script: bypass upstream hybrid_search/reranker, test generate_answer directly.

This only exercises generate_answer.py — no modifications to other components.
"""
import json
import os
import sys
import time

# Ensure RAG_TRACE is on
os.environ["RAG_TRACE"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from generate_answer import (
    generate_answer,
    TemplateProvider,
    OllamaProvider,
    MAX_SOURCES,
    GROUNDING_COVERAGE,
    MIN_CLAIMS_FOR_ANSWER,
    MAX_CLAIMS,
    REFUSAL_RESPONSE,
    build_prompt,
    parse_claim_json,
    _claim_grounded,
    _tokenize,
)
from rerank_results import RerankedResult

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "chunks.json")


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def get_rag_chunks(chunks, limit=5):
    """Pick chunks whose text actually discusses 'retrieval-augmented generation'."""
    matched = [c for c in chunks if "retrieval-augment" in c.get("text", "").lower()]
    if not matched:
        # Fallback: any RAG theme
        matched = [c for c in chunks if c.get("theme") == "rag"]
    return matched[:limit]


def build_mock_results(rag_chunks):
    results = []
    for i, chunk in enumerate(rag_chunks[:MAX_SOURCES]):
        results.append(RerankedResult(
            final_rank=i + 1,
            chunk_id=str(chunk["chunk_id"]),
            document_id=str(chunk["document_id"]),
            theme=str(chunk.get("theme", "")),
            page_start=int(chunk.get("page_start", 1)),
            page_end=int(chunk.get("page_end", 1)),
            hybrid_rank=i + 1,
            hybrid_score=0.5,
            reranker_score=1.0,
            channels=["dense", "bm25"],
            text_preview="",
        ))
    return results


def build_chunk_index(chunks):
    index = {}
    for chunk in chunks:
        index[str(chunk["chunk_id"])] = chunk
    return index


def main():
    query = "What is retrieval augmented generation?"
    chunks = load_chunks()
    rag_chunks = get_rag_chunks(chunks)
    chunk_index = build_chunk_index(chunks)
    mock_results = build_mock_results(rag_chunks)

    print("=" * 80)
    print(f"QUERY: {query}")
    print(f"Sources selected: {len(mock_results)}")
    print(f"GROUNDING_COVERAGE = {GROUNDING_COVERAGE}")
    print(f"MIN_CLAIMS_FOR_ANSWER = {MIN_CLAIMS_FOR_ANSWER}")
    print(f"MAX_CLAIMS = {MAX_CLAIMS}")
    print("=" * 80)

    # --- Template mode ---
    print("\n>>> TEMPLATE MODE")
    template_provider = TemplateProvider()
    t0 = time.perf_counter()
    resp_t = generate_answer(query, mock_results, chunk_index,
                             provider=template_provider, temperature=0.2)
    t1 = time.perf_counter()
    print(f"TEMPLATE answer: {resp_t.answer}")
    print(f"TEMPLATE claims: {[(c.text[:60], c.citations) for c in resp_t.claims]}")
    print(f"TEMPLATE latency: {resp_t.generation_latency} ms (wall: {(t1-t0)*1000:.1f} ms)")

    # --- Ollama mode ---
    print("\n>>> OLLAMA MODE")
    ollama_provider = OllamaProvider(model="qwen2.5:3b")
    print(f"Ollama provider name after init: {ollama_provider.name}")
    print(f"Ollama _template is set: {ollama_provider._template is not None}")

    t0 = time.perf_counter()
    resp_o = generate_answer(query, mock_results, chunk_index,
                             provider=ollama_provider, temperature=0.2)
    t1 = time.perf_counter()
    print(f"\nOLLAMA answer: {resp_o.answer}")
    print(f"OLLAMA claims: {[(c.text[:60], c.citations) for c in resp_o.claims]}")
    print(f"OLLAMA latency: {resp_o.generation_latency} ms (wall: {(t1-t0)*1000:.1f} ms)")

    # --- Comparison ---
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print(f"  Template answer == Ollama answer: {resp_t.answer == resp_o.answer}")
    if resp_t.answer == resp_o.answer:
        print("  >>> IDENTICAL OUTPUT — investigating fallback cause")
    else:
        print("  >>> DIFFÉRENT — Ollama claims were accepted")
    print("=" * 80)


if __name__ == "__main__":
    main()
