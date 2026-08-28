"""Audit avant/après patch : duplication citations + table MNLI + refus.

Rejoue le pipeline exact de app.py.run_pipeline pour la question cible,
affiche la réponse générée et la table verify_citations complète.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from citation_verifier import MNLICitationVerifier, verify_citations  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, OllamaProvider, decide_response, load_chunk_text_index,
    locate_corpus,
)

QUERY = "What is Retrieval-Augmented Generation?"
POOL_SIZE, TOP_K, MODEL = 20, 5, "qwen2.5:3b"


def main() -> None:
    corpus = locate_corpus("chunks.json")
    chunk_index, _ = load_chunk_text_index(corpus)
    engine = get_engine(EngineConfig(fetch_k=POOL_SIZE))
    hybrid = engine.search(query=QUERY, top_k=POOL_SIZE)
    reranked = rerank_results(
        query=QUERY, hybrid_results=hybrid.results, chunk_index=chunk_index,
        reranker=CrossEncoderReranker(device="auto"),
        top_k=TOP_K, pool_size=POOL_SIZE,
    )

    # sources_map identique à generate_answer() (+chunk_id pour la vérification)
    sources_map = []
    for r in reranked.results[:MAX_SOURCES]:
        chunk = chunk_index.get(str(r.chunk_id))
        if chunk is None:
            continue
        sources_map.append({
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "page_start": int(r.page_start),
            "page_end": int(r.page_end),
            "text": str(chunk.get("text", "")),
        })

    provider = OllamaProvider(model=MODEL)
    claims = provider.generate_claims(QUERY, sources_map, temperature=0.2)
    print("=== CLAIMS (post-grounding) ===")
    for c in claims:
        print(f"  text={c.text!r}\n  citations={c.citations}")

    answer = decide_response(claims)
    print("\n=== REPONSE FINALE (decide_response) ===")
    print(answer)

    # Test unitaire anti-doublon de render_claims_answer (claim avec citations
    # inline émises par le LLM, exécuté avec un provider template) :
    from generate_answer import Claim, render_claims_answer  # noqa: E402
    test_claim = Claim(
        text="RAG is a paradigm that allows LLMs to utilize external knowledge, "
             "as described in [2] and [3].",
        citations=[2, 3],
    )
    rendered = render_claims_answer([test_claim])
    print("\n=== TEST UNITAIRE render_claims_answer (claim inline) ===")
    print(f"  in : {test_claim.text!r} citations={test_claim.citations}")
    print(f"  out: {rendered!r}")
    import re as _re  # noqa: E402
    indices = _re.findall(r"\[(\d+)\]", rendered)
    print(f"  marqueurs={indices} doublons={len(indices) != len(set(indices))}")


    sources = [
        {"chunk_id": s["chunk_id"], "document_id": s["document_id"],
         "page_start": s["page_start"], "page_end": s["page_end"],
         "theme": ""}
        for s in sources_map
    ]
    verifier = MNLICitationVerifier(device="auto")
    verified, segments = verify_citations(
        query=QUERY, answer=answer, sources=sources,
        chunk_index=chunk_index, verifier=verifier,
    )
    print("\n=== SEGMENTS (segment_claims) ===")
    for s in segments:
        print(f"  text={s.claim_text!r} citation_indices={s.citation_indices}")

    print("\n=== TABLE CITATION VERIFICATION ===")
    for v in verified:
        print(f"  claim={v.claim_text[:60]!r} doc={v.document_id} "
              f"p.{v.page_start} support={v.support_score} verdict={v.verdict}")
    scores = [v.support_score for v in verified]
    print(f"\n  nb lignes={len(verified)} scores={scores} "
          f"tous<0.40={all(s < 0.40 for s in scores)}")

    # ---------------------------------------------------------------
    # VOIE DE PRODUCTION : generate_answer() avec la gate NLI (3bis).
    # ---------------------------------------------------------------
    from generate_answer import REFUSAL_RESPONSE, generate_answer  # noqa: E402
    resp = generate_answer(
        query=QUERY,
        reranked_results=list(reranked.results),
        chunk_index=chunk_index,
        provider=provider,
        temperature=0.2,
    )
    print("\n=== generate_answer() (voie de production, gate NLI active) ===")
    print(f"  answer={resp.answer!r}")
    print(f"  == REFUSAL_RESPONSE : {resp.answer == REFUSAL_RESPONSE}")
    print(f"  nb claims renvoyés  : {len(resp.claims)}")
    print(f"  nb citations        : {len(resp.citations)}")



if __name__ == "__main__":
    main()
