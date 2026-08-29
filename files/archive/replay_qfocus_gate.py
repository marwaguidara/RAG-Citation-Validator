"""Replay deterministe : gate NLI sur les claims LLM recoltes (APRES patch).

Pour chacune des 3 questions :
  - reconstruit les memes 5 sources (retrieval deterministe) ;
  - rejoue la reponse brute LLM capturee ;
  - score NLI PAR CITATION avec la meme logique que la gate de production ;
  - affiche le verdict de la gate (accept / refus).
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, OllamaProvider, _gate_premise, _get_nli_verifier,
    _tokenize, compute_support_score, load_chunk_text_index, locate_corpus,
    parse_claim_json,
)
from citation_verifier import SUPPORT_LOW  # noqa: E402

QUESTIONS = [
    ("What is Retrieval-Augmented Generation (RAG) and what are its main benefits?",
     '{"claims":[{"text":"Retrieval-Augmented Generation (RAG) is a paradigm that '
     'enhances the generation process by incorporating real-world images as '
     'additional references, improving the accuracy of code and image generation.",'
     '"citations":[3,4]}, {"text":"One main benefit of RAG is its ability to achieve '
     'two to four times higher execution accuracy compared to other methods, '
     'demonstrating its flexibility and effectiveness.","citations":[3,4]}, '
     '{"text":"Another benefit is its capability to handle external knowledge, '
     'allowing LLMs to incorporate real-world images and diverse knowledge bases, '
     'which is particularly useful for code and image generation tasks.",'
     '"citations":[3,4]}]}'),
    ("Explain the difference between Dense Retrieval, BM25, and Hybrid Search.",
     '{"claims":[{"text":"Dense Retrieval focuses on predicting salient spans like '
     'named entities, while BM25 relies on term-frequency and inverse document '
     'frequency for ranking, and Hybrid Search combines both dense and sparse '
     'retrieval methods.","citations":[3,2]},{"text":"Dense Retrieval benefits from '
     'pre-training to predict salient spans, BM25 excels in datasets with strong '
     'baseline performance, and Hybrid Search offers a balanced approach by '
     'integrating both dense and sparse retrieval.","citations":[3,2]}]}'),
    ("Why does Hybrid Search usually outperform Dense Retrieval alone?",
     '{"claims":[{"text":"Hybrid Search usually outperforms Dense Retrieval alone '
     'because it combines both dense and sparse retrieval methods, improving '
     'overall accuracy and relevance.","citations":[1,2]}]}'),
]


def main() -> None:
    chunk_index, _ = load_chunk_text_index(locate_corpus("chunks.json"))
    engine = get_engine(EngineConfig(fetch_k=20))
    reranker = CrossEncoderReranker(device="auto")
    verifier = _get_nli_verifier()

    for query, raw in QUESTIONS:
        print(f"\n{'='*74}\n{query}\n{'='*74}")
        hybrid = engine.search(query=query, top_k=20)
        reranked = rerank_results(
            query=query, hybrid_results=hybrid.results, chunk_index=chunk_index,
            reranker=reranker, top_k=5, pool_size=20,
        )
        sources_map = []
        for r in reranked.results[:MAX_SOURCES]:
            chunk = chunk_index.get(str(r.chunk_id))
            if chunk is None:
                continue
            sources_map.append({
                "chunk_id": str(r.chunk_id), "document_id": str(r.document_id),
                "page_start": int(r.page_start), "page_end": int(r.page_end),
                "text": str(chunk.get("text", "")),
            })
        print("sources:", [s["document_id"] for s in sources_map])

        claims = parse_claim_json(raw)
        claims = OllamaProvider._sanitize_claim_citations(claims, len(sources_map))
        print(f"claims parses+sanitizes: {len(claims)}")
        any_supported = False
        for ci, claim in enumerate(claims):
            qtok = _tokenize(claim.text)
            # Score NLI contre TOUTES les sources (pas seulement celles citées)
            best_idx, best_support, best_probs = None, -1.0, None
            for idx in range(1, len(sources_map) + 1):
                text = str(sources_map[idx - 1].get("text", ""))
                premise = _gate_premise(text, claim.text)
                probs = verifier.score_pairs([premise], [claim.text])[0]
                support = compute_support_score(*probs)
                if support > best_support:
                    best_idx, best_support, best_probs = idx, support, probs
            print(f"  claim[{ci}] MEILLEURE SOURCE = [{best_idx}] "
                  f"doc={sources_map[best_idx-1]['document_id']} "
                  f"support={best_support:.4f} ent={best_probs[2]:.3f} "
                  f"cites_llm={claim.citations}")
            if best_support >= SUPPORT_LOW:
                any_supported = True
        print(f"--> GATE : {'REPONSE GARDEE' if any_supported else 'REFUS -> repli extractif'}")


if __name__ == "__main__":
    main()
