"""Replay deterministe POST-PATCH "tolérance paraphrase".

Utilise les claims RÉELS capturés dans audit_qfocus_AFTER2 (qf_out.txt) et
rejoue le code de PRODUCTION apply_nli_gate avec le rescue
PARAPHRASE_COVERAGE_RESCUE pour montrer le verdict AVANT/APRÈS patch.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, PARAPHRASE_COVERAGE_RESCUE, Claim, apply_nli_gate,
    decide_response, split_sentences, _tokenize,
    load_chunk_text_index, locate_corpus,
)

QUESTIONS = [
    (
        "What is Retrieval-Augmented Generation (RAG) and what are its main benefits?",
        [
            Claim("Retrieval-Augmented Generation (RAG) is a paradigm that enhances the generation process by incorporating real-world images as additional references, improving the accuracy of code and image generation.", [3, 4]),
            Claim("One main benefit of RAG is its ability to achieve two to four times higher execution accuracy compared to other methods, demonstrating its flexibility and effectiveness.", [3, 4]),
            Claim("Another benefit is its capability to handle external knowledge, allowing LLMs to incorporate real-world images and diverse knowledge bases, which is particularly useful for code and image generation tasks.", [3, 4]),
        ],
    ),
    (
        "Explain the difference between Dense Retrieval, BM25, and Hybrid Search.",
        [
            Claim("Dense Retrieval focuses on predicting salient spans like named entities, while BM25 relies on term-frequency and inverse document frequency for ranking, and Hybrid Search combines both dense and sparse retrieval methods.", [2, 3]),
            Claim("Dense Retrieval benefits from pre-training to predict salient spans, BM25 excels in datasets with strong baseline performance, and Hybrid Search offers a balanced approach by integrating both dense and sparse retrieval.", [2, 3]),
        ],
    ),
    (
        "Why does Hybrid Search usually outperform Dense Retrieval alone?",
        [
            Claim("Hybrid Search usually outperforms Dense Retrieval alone because it combines both dense and sparse retrieval methods, improving overall accuracy and relevance.", [1, 2]),
        ],
    ),
]


def main() -> None:
    chunk_index, _ = load_chunk_text_index(locate_corpus("chunks.json"))
    engine = get_engine(EngineConfig(fetch_k=20))
    reranker = CrossEncoderReranker(device="auto")

    for query, claims in QUESTIONS:
        print(f"\n{'='*76}\nQUESTION : {query}\n{'='*76}")
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

        answer_text = decide_response(claims)
        print(f"\nreponse LLM (avant gate) :\n  {answer_text}")

        # Couverture max + contradiction par claim (ce que fait le rescue)
        for ci, c in enumerate(claims):
            tokens = _tokenize(c.text)
            best = max(
                (len(tokens & _tokenize(str(s.get("text", "")))) / len(tokens)
                 for s in sources_map if _tokenize(str(s.get("text", ""))))
            )
            # G3 : couverture au niveau PHRASE sur les sources CITÉES
            cited_sent_cov = 0.0
            cited_best = []
            cited_whole = 0.0
            cited_whole_ids = []
            for idx in c.citations:
                if not 1 <= idx <= len(sources_map):
                    continue
                chunk = str(sources_map[idx - 1].get("text", ""))
                whole_cov = (len(tokens & _tokenize(chunk)) / len(tokens)
                             if _tokenize(chunk) else 0.0)
                cited_whole_ids.append((idx, round(whole_cov, 3)))
                cited_whole = max(cited_whole, whole_cov)
                sents = [s for s in split_sentences(chunk) if s.strip()]
                if not sents:
                    continue
                scov = max(
                    (len(tokens & _tokenize(s)) / len(tokens) for s in sents)
                )
                cited_best.append((idx, round(scov, 3)))
                cited_sent_cov = max(cited_sent_cov, scov)
            print(f"  claim[{ci}] cov_max_toutes_sources={best:.3f} "
                  f"cov_CHUNK_citees={cited_whole_ids} max={cited_whole:.3f} "
                  f"cov_phrase_citees={cited_best} max={cited_sent_cov:.3f}")

        final_claims, final_answer = apply_nli_gate(query, claims, answer_text, sources_map)
        if final_answer == "The provided sources do not contain sufficient information.":
            print("\n--> GATE : REFUS -> repli extractif (hors sujet)\n")
        else:
            print(f"\n--> GATE : REPONSE CONSERVEE (centre-question)\n"
                  f"    {final_answer}\n"
                  f"    claims={len(final_claims)} "
                  f"citations={[c.citations for c in final_claims]}")


if __name__ == "__main__":
    main()