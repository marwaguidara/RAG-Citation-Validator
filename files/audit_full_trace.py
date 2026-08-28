"""AUDIT TRACE COMPLET — question : "What is Retrieval-Augmented Generation?"

Dump intégral :
  1. Top-20 hybrides (hybrid_search, RRF)
  2. Top-5 rerankés (BGE cross-encoder)
  3. Texte COMPLET des 5 chunks injectés dans le prompt
  4. Prompt final envoyé à qwen2.5:3b
  5. Réponse brute du LLM
  6. Localisation de "real-world images ... additional references" dans le contexte

Aucune modification des composants retrieval/reranker : simple réutilisation.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, OllamaProvider, build_prompt, load_chunk_text_index,
    locate_corpus,
)

QUERY = "What is Retrieval-Augmented Generation?"
POOL_SIZE = 20
TOP_K = 5
MODEL = "qwen2.5:3b"
MARKERS = [
    "real-world images",
    "real world images",
    "additional references",
    "real-world",
]


def main() -> None:
    out = []

    def say(line: str = "") -> None:
        out.append(line)
        print(line)

    corpus_path = locate_corpus("chunks.json") or (
        Path(__file__).resolve().parent / "corpus" / "chunks.json"
    )
    chunk_index, _meta = load_chunk_text_index(corpus_path)
    say(f"Corpus: {corpus_path}  chunks={len(chunk_index)}")

    # ------------------------------------------------------------------ 1)
    engine = get_engine(EngineConfig(fetch_k=POOL_SIZE))
    t0 = time.perf_counter()
    hybrid = engine.search(query=QUERY, top_k=POOL_SIZE)
    say(f"\n{'='*78}\n1) TOP-{POOL_SIZE} HYBRIDES (RRF)  "
        f"[{ (time.perf_counter()-t0)*1000:.0f} ms]\n{'='*78}")
    for rank, r in enumerate(hybrid.results, start=1):
        say(f"  {rank:>2}. chunk={r.chunk_id} doc={r.document_id} "
            f"p.{r.page_start}-{r.page_end} "
            f"hybrid_score={getattr(r, 'hybrid_score', float('nan')):.6f}")

    # ------------------------------------------------------------------ 2)
    reranker = CrossEncoderReranker(device="auto")
    t0 = time.perf_counter()
    reranked = rerank_results(
        query=QUERY,
        hybrid_results=hybrid.results,
        chunk_index=chunk_index,
        reranker=reranker,
        top_k=TOP_K,
        pool_size=POOL_SIZE,
    )
    say(f"\n{'='*78}\n2) TOP-{TOP_K} RERANKES (BGE)  "
        f"[{ (time.perf_counter()-t0)*1000:.0f} ms]\n{'='*78}")
    for rank, r in enumerate(reranked.results, start=1):
        say(f"  {rank:>2}. chunk={r.chunk_id} doc={r.document_id} "
            f"p.{r.page_start}-{r.page_end} "
            f"reranker_score={getattr(r, 'reranker_score', float('nan')):.4f}")

    # ------------------------------------------------------------------ 3)
    sources_map = []
    for result in reranked.results[:MAX_SOURCES]:
        chunk = chunk_index.get(str(result.chunk_id))
        if chunk is None:
            continue
        sources_map.append({
            "document_id": str(result.document_id),
            "page_start": int(result.page_start),
            "page_end": int(result.page_end),
            "text": str(chunk.get("text", "")),
        })

    say(f"\n{'='*78}\n3) TEXTE COMPLET DES {len(sources_map)} CHUNKS INJECTES\n{'='*78}")
    rerank_order = [str(r.chunk_id) for r in reranked.results]
    for i, s in enumerate(sources_map, start=1):
        say(f"\n----- SOURCE [{i}] doc={s['document_id']} "
            f"p.{s['page_start']}-{s['page_end']} -----")
        say(s["text"])

    # ------------------------------------------------------------- 6) marqueur
    say(f"\n{'='*78}\n6) RECHERCHE DU MARQUEUR FAUTIF DANS LE CONTEXTE\n{'='*78}")
    found = False
    for i, s in enumerate(sources_map, start=1):
        low = s["text"].lower()
        for marker in MARKERS:
            pos = low.find(marker)
            if pos != -1:
                found = True
                ctx = s["text"][max(0, pos - 250): pos + 300]
                say(f"  TROUVE dans SOURCE [{i}] (doc={s['document_id']} "
                    f"p.{s['page_start']}-{s['page_end']}): "
                    f"marker={marker!r} @ {pos}\n"
                    f"  ...{ctx}...")
    if not found:
        say("  Aucun marqueur trouvé dans les 5 chunks => hallucination pure du LLM.")

    # ------------------------------------------------------------------ 4)
    prompt = build_prompt(QUERY, sources_map)
    say(f"\n{'='*78}\n4) PROMPT FINAL ENVOYE A {MODEL}\n{'='*78}")
    say(prompt)

    # ------------------------------------------------------------------ 5)
    provider = OllamaProvider(model=MODEL)
    t0 = time.perf_counter()
    raw = provider.generate(prompt, temperature=0.2)
    say(f"\n{'='*78}\n5) REPONSE BRUTE DU LLM  "
        f"[{ (time.perf_counter()-t0)*1000:.0f} ms]\n{'='*78}")
    say(repr(raw))

    Path(__file__).with_name("audit_full_trace_output.txt").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("\n[saved] files/audit_full_trace_output.txt")


if __name__ == "__main__":
    main()
