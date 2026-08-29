"""Test gate union : claim vs source individuelle vs union des prémisses.

Claim fixé = celui du run PROMPT_MODIFIE (synthèse [1]+[2]).
Aucune modification de citation_verifier : simple réutilisation.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from citation_verifier import (  # noqa: E402
    MNLICitationVerifier, SUPPORT_LOW, compute_support_score,
    extract_local_premise, verdict_for,
)
from generate_answer import MAX_SOURCES, load_chunk_text_index, locate_corpus  # noqa: E402

QUERY = "What is Retrieval-Augmented Generation?"
CLAIM = (
    "Retrieval-Augmented Generation (RAG) is a paradigm that allows large "
    "language models (LLMs) to efficiently utilize external knowledge by "
    "combining a retrieval system with an LLM-based generation module."
)
CITED = [1, 2]  # sanitation du run PROMPT_MODIFIE


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
        if chunk is None:
            continue
        sources_map.append({"text": str(chunk.get("text", ""))})

    premises = [extract_local_premise(sources_map[i - 1]["text"], CLAIM)
                for i in CITED]
    verifier = MNLICitationVerifier(device="auto")

    # 1) scores individuels
    probs = verifier.score_pairs(premises, [CLAIM] * len(premises))
    for i, p in zip(CITED, probs):
        s = compute_support_score(p[0], p[1], p[2])
        print(f"  source[{i}] support={s:.4f} verdict={verdict_for(s)}")

    # 2) union des prémisses citées
    union = " ".join(premises)
    probs_u = verifier.score_pairs([union], [CLAIM])
    s_u = compute_support_score(probs_u[0][0], probs_u[0][1], probs_u[0][2])
    print(f"\n  UNION [{','.join(map(str, CITED))}] support={s_u:.4f} "
          f"verdict={verdict_for(s_u)}  (seuil refus={SUPPORT_LOW})")
    print(f"  => gate union eviterait le refus : {s_u >= SUPPORT_LOW}")

    # 3) claim court PUR (formulation historique 0.93-0.97)
    SHORT = ("Retrieval-Augmented Generation (RAG) is a paradigm that allows "
             "large language models (LLMs) to efficiently utilize external "
             "knowledge.")
    probs_s = verifier.score_pairs([premises[0]], [SHORT])
    s_s = compute_support_score(probs_s[0][0], probs_s[0][1], probs_s[0][2])
    print(f"\n  CLAIM COURT vs source[1] support={s_s:.4f} "
          f"verdict={verdict_for(s_s)}")

    # 4) repli extractif : claims verbatim de extract_grounded_claims
    from generate_answer import extract_grounded_claims, decide_response  # noqa: E402
    fb_claims = extract_grounded_claims(QUERY, sources_map)
    print(f"\n  REPLI EXTRACTIF : {len(fb_claims)} claims")
    fb_pairs = []
    for c in fb_claims:
        for i in c.citations:
            fb_pairs.append((c, i, extract_local_premise(
                sources_map[i - 1]["text"], c.text)))
    if fb_pairs:
        probs_f = verifier.score_pairs(
            [p for _, _, p in fb_pairs], [c.text for c, _, _ in fb_pairs])
        for (c, i, _), p in zip(fb_pairs, probs_f):
            s_f = compute_support_score(p[0], p[1], p[2])
            print(f"    doc[{i}] support={s_f:.4f} verdict={verdict_for(s_f)} "
                  f"text={c.text[:70]!r}")
    print(f"\n  reponse extractive rendue :\n  {decide_response(fb_claims)[:200]}")

    # 5) fenetre +-1 vs phrase de definition SEULE (dilution de la fenetre ?)
    from generate_answer import split_sentences, _tokenize  # noqa: E402
    sents = split_sentences(sources_map[0]["text"])
    qt = _tokenize(CLAIM)
    best, best_ov = None, -1
    for s in sents:
        ov = len(_tokenize(s) & qt)
        if ov > best_ov:
            best, best_ov = s, ov
    print(f"\n  phrase de definition seule (overlap={best_ov}) :")
    print(f"    {best[:150]!r}")
    probs_d = verifier.score_pairs([best, best], [CLAIM, SHORT])
    for label, p in (("CLAIM mixte", probs_d[0]), ("CLAIM court", probs_d[1])):
        s_d = compute_support_score(p[0], p[1], p[2])
        print(f"    {label} vs phrase seule : support={s_d:.4f} "
              f"verdict={verdict_for(s_d)}")


if __name__ == "__main__":
    main()
