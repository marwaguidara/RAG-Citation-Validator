"""Test déterministe du chemin de refus (SANS LLM) : étape 3bis.

Scénarios :
  A) réponse LLM hallucinée (aucun support)  -> repli extractif -> réponse affichée
  B) pool sans contenu pertinent (textes factices) -> extractif non supporté -> REFUS
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, REFUSAL_RESPONSE, apply_nli_gate,
    load_chunk_text_index, locate_corpus,
)

QUERY = "What is Retrieval-Augmented Generation?"
HALLUCINATION = (
    "Retrieval-Augmented Generation was invented in 1997 by NASA to render "
    "photorealistic dragons, and it requires quantum computers to operate [1]."
)


def gate_pipeline(query: str, answer: str, sources_map: list[dict]) -> str:
    """Appelle la FONCTION RÉELLE apply_nli_gate de generate_answer.py."""
    claims, out = apply_nli_gate(query, [type("C", (), {"text": answer,
                                                       "citations": [1]})()],
                                 answer, sources_map)
    print(f"    [gate] {len(claims)} claims retenus")
    return out


def main() -> None:
    corpus = locate_corpus("chunks.json")
    chunk_index, _ = load_chunk_text_index(corpus)
    engine = get_engine(EngineConfig(fetch_k=20))
    hybrid = engine.search(query=QUERY, top_k=20)
    reranked = rerank_results(
        query=QUERY, hybrid_results=hybrid.results, chunk_index=chunk_index,
        reranker=CrossEncoderReranker(device="auto"), top_k=5, pool_size=20,
    )
    sources_map = []
    for r in reranked.results[:MAX_SOURCES]:
        chunk = chunk_index.get(str(r.chunk_id))
        if chunk is not None:
            sources_map.append({"text": str(chunk.get("text", ""))})

    print("=== SCENARIO A : hallucination LLM, pool réel ===")
    print(f"  entrée : {HALLUCINATION!r}")
    out_a = gate_pipeline(QUERY, HALLUCINATION, sources_map)
    print(f"  sortie : {out_a[:160]!r}")
    print(f"  == REFUSAL ? {out_a == REFUSAL_RESPONSE}")

    print("\n=== SCENARIO B : pool sans contenu pertinent (textes factices) ===")
    fake_pool = [{"text": (
        "The recipe for sourdough bread requires flour, water and salt. "
        "Knead the dough for ten minutes and let it rise overnight [1]."
    )}] * 3
    out_b = gate_pipeline(QUERY, HALLUCINATION, fake_pool)
    print(f"  sortie : {out_b[:120]!r}")
    print(f"  == REFUSAL ? {out_b == REFUSAL_RESPONSE}")


if __name__ == "__main__":
    main()
