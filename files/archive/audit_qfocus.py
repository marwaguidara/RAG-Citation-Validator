"""Audit focus-question : pourquoi la réponse est un assemblage de faits
supportés au lieu de répondre à la question.

Pour chaque question :
  1. question utilisateur
  2. prompt exact envoyé au LLM
  3. réponse brute du LLM
  4. claims extraits (+ couverture lexicale par claim vs GROUNDING_COVERAGE)
  5. réponse finale affichée (après gate NLI)
"""
import os
import sys
from pathlib import Path

os.environ["RAG_TRACE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    GROUNDING_COVERAGE, MAX_SOURCES, OllamaProvider, _claim_grounded,
    _tokenize, apply_nli_gate, build_prompt, decide_response,
    load_chunk_text_index, locate_corpus, parse_claim_json,
)

QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG) and what are its main benefits?",
    "Explain the difference between Dense Retrieval, BM25, and Hybrid Search.",
    "Why does Hybrid Search usually outperform Dense Retrieval alone?",
]
POOL_SIZE, TOP_K, MODEL = 20, 5, "qwen2.5:3b"


def coverage(claim, sources_map):
    toks = _tokenize(claim.text)
    out = []
    for i in claim.citations:
        if 1 <= i <= len(sources_map):
            st = _tokenize(str(sources_map[i - 1].get("text", "")))
            out.append((i, round(len(toks & st) / max(len(toks), 1), 3)))
    return out


def main() -> None:
    corpus = locate_corpus("chunks.json")
    chunk_index, _ = load_chunk_text_index(corpus)
    engine = get_engine(EngineConfig(fetch_k=POOL_SIZE))
    reranker = CrossEncoderReranker(device="auto")
    provider = OllamaProvider(model=MODEL)

    for qn, query in enumerate(QUESTIONS, start=1):
        print(f"\n{'#'*78}\n# QUESTION {qn} : {query}\n{'#'*78}")
        hybrid = engine.search(query=query, top_k=POOL_SIZE)
        reranked = rerank_results(
            query=query, hybrid_results=hybrid.results, chunk_index=chunk_index,
            reranker=reranker, top_k=TOP_K, pool_size=POOL_SIZE,
        )
        sources_map = []
        for r in reranked.results[:MAX_SOURCES]:
            chunk = chunk_index.get(str(r.chunk_id))
            if chunk is None:
                continue
            sources_map.append({
                "document_id": str(r.document_id),
                "page_start": int(r.page_start),
                "page_end": int(r.page_end),
                "text": str(chunk.get("text", "")),
            })
        print("\n--- 1) QUESTION UTILISATEUR ---")
        print(query)
        print("\n--- 2) PROMPT EXACT ---")
        print(build_prompt(query, sources_map))

        prompt = build_prompt(query, sources_map)
        raw = provider.generate(prompt, temperature=0.2)
        print("\n--- 3) REPONSE BRUTE DU LLM ---")
        print(repr(raw))

        candidates = OllamaProvider._sanitize_claim_citations(
            parse_claim_json(raw), len(sources_map))
        print("\n--- 4) CLAIMS EXTRAITS (+ couverture vs seuil "
              f"{GROUNDING_COVERAGE}) ---")
        for i, c in enumerate(candidates):
            ok = _claim_grounded(c, sources_map)
            print(f"  [{i}] cits={c.citations} covs={coverage(c, sources_map)} "
                  f"-> {'OK' if ok else 'REJECT'}\n      text={c.text[:110]!r}")

        claims = provider.generate_claims(query, sources_map, temperature=0.2)
        answer = decide_response(claims)
        claims, answer = apply_nli_gate(query, claims, answer, sources_map)
        print("\n--- 5) REPONSE FINALE AFFICHEE ---")
        print(answer)
        print(f"  (claims retenus={len(claims)}, "
              f"citations={[c.citations for c in claims]})")


if __name__ == "__main__":
    main()
