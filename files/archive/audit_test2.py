"""Audit script v2: find proper RAG chunks, then test both providers."""
import json
import os
import sys

os.environ["RAG_TRACE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
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
    render_claims_answer,
)
from rerank_results import RerankedResult

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "chunks.json")


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def find_rag_chunks(chunks):
    """Find chunks whose text contains 'retrieval augment' (the RAG concept)."""
    result = []
    for i, c in enumerate(chunks):
        text = c.get("text", "").lower()
        if "retrieval augment" in text and "rag" in text:
            result.append((i, c))
    return result


def build_mock_results(rag_chunks):
    results = []
    for i, (_, chunk) in enumerate(rag_chunks[:MAX_SOURCES]):
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
    chunks = load_chunks()
    chunk_index = build_chunk_index(chunks)

    # Find chunks about RAG
    rag_hits = find_rag_chunks(chunks)
    print(f"Found {len(rag_hits)} chunks mentioning 'retrieval augment' + 'rag'")
    for idx, (i, c) in enumerate(rag_hits[:5]):
        print(f"  [{i}] doc={c['document_id'][:40]} text={c['text'][:120].strip()}")
    print()

    if not rag_hits:
        print("No RAG-specific chunks found. Trying broader search...")
        for i, c in enumerate(chunks):
            text = c.get("text", "").lower()
            if "retrieval augmented" in text:
                print(f"  [{i}] doc={c['document_id'][:40]} text={c['text'][:120].strip()}")
                rag_hits.append((i, c))
                if len(rag_hits) >= 5:
                    break

    mock_results = build_mock_results(rag_hits)
    query = "What is retrieval augmented generation?"

    print(f"\nQUERY: {query}")
    print(f"Providers selected: {len(mock_results)} chunks")
    print(f"GROUNDING_COVERAGE = {GROUNDING_COVERAGE}")
    print(f"MIN_CLAIMS_FOR_ANSWER = {MIN_CLAIMS_FOR_ANSWER}")
    print(f"MAX_CLAIMS = {MAX_CLAIMS}")
    print("=" * 80)

    # --- Template mode ---
    print("\n>>> TEMPLATE MODE")
    tp = TemplateProvider()
    resp_t = generate_answer(query, mock_results, chunk_index,
                             provider=tp, temperature=0.2)
    print(f"\nTEMPLATE answer: {resp_t.answer[:200]}")
    print(f"TEMPLATE claims: {len(resp_t.claims)} claims")
    for c in resp_t.claims:
        print(f"  - text={c.text[:80]} cits={c.citations}")
    print(f"TEMPLATE latency: {resp_t.generation_latency} ms")

    # --- Ollama mode ---
    print("\n>>> OLLAMA MODE")
    op = OllamaProvider(model="qwen2.5:3b")
    print(f"Provider name: {op.name}, _template set: {op._template is not None}")

    resp_o = generate_answer(query, mock_results, chunk_index,
                             provider=op, temperature=0.2)
    print(f"\nOLLAMA answer: {resp_o.answer[:200]}")
    print(f"OLLAMA claims: {len(resp_o.claims)} claims")
    for c in resp_o.claims:
        print(f"  - text={c.text[:80]} cits={c.citations}")
    print(f"OLLAMA latency: {resp_o.generation_latency} ms")

    # --- Comparison ---
    print("\n" + "=" * 80)
    print(f"Template answer == Ollama answer: {resp_t.answer == resp_o.answer}")
    if resp_t.answer == resp_o.answer:
        print(">>> IDENTICAL OUTPUT")
    else:
        print(">>> DIFFERENT OUTPUT")
    print("=" * 80)


if __name__ == "__main__":
    main()
