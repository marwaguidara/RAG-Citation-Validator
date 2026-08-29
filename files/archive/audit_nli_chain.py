"""Audit chaîne complète generation -> NLI.

Pour chaque claim vérifié par verify_citations() :
  1. claim généré (hypothesis envoyée à roberta-large-mnli)
  2. chunk cité (document, pages, texte)
  3. fenêtre locale réellement utilisée (extract_local_premise, rejouée)
  4. hypothesis
  5. scores complets : entailment, neutral, contradiction, support_score
  6. verdict
  7. analyse (métriques lexicales objectives + termes du claim absents du chunk)

Rejoue le pipeline EXACT de app.run_pipeline. Aucun module modifié.
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import (  # noqa: E402
    MAX_SOURCES, build_provider, generate_answer, _tokenize,
    load_chunk_text_index, locate_corpus,
)
from citation_verifier import (  # noqa: E402
    MNLICitationVerifier, extract_local_premise, verify_citations,
)

QUERY = (
    "What is Retrieval-Augmented Generation (RAG) "
    "and what are its main benefits?"
)


def overlap_claim(claim: str, text: str) -> float:
    """Part des tokens du claim présents dans text (couverture claim)."""
    tc, tt = _tokenize(claim), _tokenize(text)
    return len(tc & tt) / len(tc) if tc else 0.0


def main() -> None:
    lines: list[str] = []

    def say(line: str = "") -> None:
        lines.append(line)

    chunk_index, _ = load_chunk_text_index(locate_corpus("chunks.json"))
    engine = get_engine(EngineConfig(fetch_k=20))
    reranker = CrossEncoderReranker(device="auto")
    verifier = MNLICitationVerifier(device="auto")
    provider = build_provider("ollama", "qwen2.5:3b")

    t0 = time.perf_counter()
    hybrid = engine.search(query=QUERY, top_k=20)
    reranked = rerank_results(
        query=QUERY, hybrid_results=hybrid.results, chunk_index=chunk_index,
        reranker=reranker, top_k=5, pool_size=20,
    )
    say(f"retrieval: hybrid={len(hybrid.results)} reranked={len(reranked.results)} "
        f"[{(time.perf_counter()-t0)*1000:.0f} ms]")
    for i, r in enumerate(reranked.results[:5], 1):
        say(f"  rerank[{i}] doc={r.document_id} p.{r.page_start}-{r.page_end} "
            f"score={getattr(r, 'reranker_score', float('nan')):.4f} chunk={r.chunk_id}")

    answer_resp = generate_answer(
        query=QUERY, reranked_results=reranked.results,
        chunk_index=chunk_index, provider=provider, temperature=0.2,
    )
    say(f"\n=== REPONSE FINALE (generate_answer) ===\n{answer_resp.answer}")
    say(f"generation_latency={answer_resp.generation_latency} ms")

    sources = [
        {
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "page_start": int(r.page_start),
            "page_end": int(r.page_end),
            "theme": "",
        }
        for r in reranked.results[:MAX_SOURCES]
        if str(r.chunk_id) in chunk_index
    ]
    verified, segments = verify_citations(
        query=QUERY, answer=answer_resp.answer, sources=sources,
        chunk_index=chunk_index, verifier=verifier,
    )

    say(f"\n=== SEGMENTS (segment_claims) : {len(segments)} ===")
    for si, seg in enumerate(segments):
        say(f"  seg[{si}] cits={seg.citation_indices} "
            f"text={seg.claim_text[:90]}")

    # Reconstruit l'ordre des paires (segment_index, source_index) comme fait
    # par verify_citations pour associer chaque résultat à sa fenêtre locale.
    pair_keys: list[tuple[int, int]] = []
    for seg_idx, segment in enumerate(segments):
        for src_idx in segment.citation_indices:
            if 1 <= src_idx <= len(sources):
                passage = chunk_index.get(str(sources[src_idx - 1]["chunk_id"]))
                if passage is None:
                    continue
                pair_keys.append((seg_idx, src_idx))
    assert len(pair_keys) == len(verified), (
        f"mismatch pairs={len(pair_keys)} results={len(verified)}"
    )

    say(f"\n{'='*80}\nDETAILL PAR PAIRE (claim, source) — {len(verified)} paires\n{'='*80}")
    for (seg_idx, src_idx), v in zip(pair_keys, verified):
        seg = segments[seg_idx]
        source = sources[src_idx - 1]
        chunk = chunk_index.get(str(source["chunk_id"]))
        chunk_text = str(chunk.get("text", "")) if chunk else ""
        window = extract_local_premise(chunk_text, seg.claim_text)

        say(f"\n--- PAIR [{seg_idx},{src_idx}] source=[{src_idx}] "
            f"doc={v.document_id} p.{v.page_start}-{v.page_end} id={v.citation_id}")
        say(f"\n[1] CLAIM (hypothesis) :\n    {seg.claim_text}")

        saysrc = " ".join(chunk_text.split())
        say(f"\n[2] CHUNK CITE (entier, {len(chunk_text)} chars) :\n"
            f"    {saysrc[:1500] + (' [...]' if len(saysrc) > 1500 else '')}")

        say(f"\n[3] FENETRE LOCALE NLI (extract_local_premise, {len(window)} chars) :\n"
            f"    {' '.join(window.split())}")
        say(f"\n[4] HYPOTHESIS envoyée à MNLI :\n    {seg.claim_text}")

        say(f"\n[5] SCORES COMPLETS :")
        say(f"    entailment      = {v.entailment_score}")
        say(f"    neutral         = {v.neutral_score}")
        say(f"    contradiction   = {v.contradiction_score}")
        say(f"    support_score   = {v.support_score}")
        say(f"\n[6] VERDICT : {v.verdict}")

        # ---- 7. analyse objective ----
        cov_win = overlap_claim(seg.claim_text, window)
        cov_chunk = overlap_claim(seg.claim_text, chunk_text)
        tokens_claim = _tokenize(seg.claim_text)
        terms_missing = sorted(
            tc for tc in (tokens_claim - _tokenize(chunk_text)) if len(tc) >= 4
        )
        terms_in_window = sorted(
            tc for tc in (tokens_claim - _tokenize(window)) if len(tc) >= 4
        )
        say(f"\n[7] ANALYSE :")
        say(f"    couverture tokens claim / fenêtre      = {cov_win:.3f}")
        say(f"    couverture tokens claim / chunk entier = {cov_chunk:.3f}")
        say(f"    termes claim absents du chunk (>=4c)   = {terms_missing}")
        say(f"    termes claim absents de la fenêtre     = {terms_in_window}")
        ent = v.entailment_score
        if v.contradiction_score > ent and v.contradiction_score >= 0.5:
            verdict = "A: CONTRADICTION avec la source (le chunk contredit le claim)"
        elif cov_chunk < 0.30 and len(terms_missing) >= 3:
            verdict = "A: claim réellement absent du chunk (faible chevauchement lexical)"
        elif ent <= 0.3:
            verdict = "B: paraphrase/synthèse correcte que MNLI évalue 'neutral' (limite du modèle NLI)"
        else:
            verdict = "A/B: chevauchement partiel — à juger sur le fond"
        say(f"    DIAGNOSTIC : {verdict}")

    scored_flat = [v.support_score for v in verified]
    verdicts = [v.verdict for v in verified]
    say(f"\n=== SYNTHESE ===\n    scores={scored_flat}\n    verdicts={verdicts}")

    Path(__file__).with_name("audit_nli_chain_out.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[saved] files/audit_nli_chain_out.txt  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
