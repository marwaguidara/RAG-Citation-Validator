"""Replay Q1 : couverture du claim-réponse vs LES 5 sources (réparation citation ?).

Le pipeline retrieval est déterministe => sources_map identiques au run AFTER.
Le texte du claim est repris de la réponse brute sauvegardée (audit_qfocus_AFTER).
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, _all_citations_unsupported, _claim_grounded, _tokenize,
    load_chunk_text_index, locate_corpus,
)

QUERY = ("What is Retrieval-Augmented Generation (RAG) and what are its "
         "main benefits?")
CLAIM0 = ("Retrieval-Augmented Generation (RAG) is a paradigm that allows "
          "large language models to efficiently utilize external knowledge, "
          "enhancing their performance in tasks like code generation and "
          "image generation.")
CLAIM2 = ("RAG also enables models to retrieve relevant documents to "
          "supplement parametric knowledge during response generation.")


def main() -> None:
    chunk_index, _ = load_chunk_text_index(locate_corpus("chunks.json"))
    engine = get_engine(EngineConfig(fetch_k=20))
    hybrid = engine.search(query=QUERY, top_k=20)
    reranked = rerank_results(
        query=QUERY, hybrid_results=hybrid.results, chunk_index=chunk_index,
        reranker=CrossEncoderReranker(device="auto"), top_k=5, pool_size=20,
    )
    sources_map = []
    for r in reranked.results[:MAX_SOURCES]:
        chunk = chunk_index.get(str(r.chunk_id))
        if chunk is None:
            continue
        sources_map.append(str(chunk.get("text", "")))

    for name, claim_text in (("CLAIM0 (réponse directe)", CLAIM0),
                             ("CLAIM2 (preuve)", CLAIM2)):
        toks = _tokenize(claim_text)
        print(f"\n=== {name} ===")
        print(f"  text={claim_text[:90]!r}")
        best = []
        for i, text in enumerate(sources_map, start=1):
            cov = len(toks & _tokenize(text)) / max(len(toks), 1)
            best.append((round(cov, 3), i))
        best.sort(reverse=True)
        for cov, i in best:
            print(f"  source[{i}] cov={cov}")
        # réparation : citer les 1-2 meilleures sources si >= 0.45
        kept = [i for cov, i in best if cov >= 0.45][:2]
        print(f"  citations actuelles=[2,4]  -> réparation propose: {kept}")
        if kept:
            from citation_verifier import (  # noqa: E402
                MNLICitationVerifier, compute_support_score,
            )
            from generate_answer import Claim, _gate_premise
            repaired = Claim(text=claim_text, citations=kept)
            ok = _claim_grounded(repaired, [{"text": t} for t in sources_map])
            print(f"  grounding apres reparation={ok}")
            verifier = MNLICitationVerifier(device="auto")
            for i in kept:
                premise = _gate_premise(sources_map[i - 1], claim_text)
                probs = verifier.score_pairs([premise], [claim_text])
                c, n, e = probs[0]
                print(f"  gate source[{i}]: support={compute_support_score(c, n, e):.4f} "
                      f"(entail={e:.3f} premise={premise[:70]!r})")


if __name__ == "__main__":
    main()
