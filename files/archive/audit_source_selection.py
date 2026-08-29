"""Audit sélection de source : pourquoi la meilleure source n'est pas citée.

Usage : python audit_source_selection.py <label>
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
import generate_answer as ga  # noqa: E402

QUERY = "What is Retrieval-Augmented Generation?"
POOL_SIZE, TOP_K, MODEL = 20, 5, "qwen2.5:3b"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "RUN"
TARGET_DOC = "2402.12317v2"


def main() -> None:
    out: list[str] = []

    def say(line: str = "") -> None:
        out.append(line)
        print(line)

    corpus = ga.locate_corpus("chunks.json")
    chunk_index, _ = ga.load_chunk_text_index(corpus)
    engine = get_engine(EngineConfig(fetch_k=POOL_SIZE))
    hybrid = engine.search(query=QUERY, top_k=POOL_SIZE)
    reranked = rerank_results(
        query=QUERY, hybrid_results=hybrid.results, chunk_index=chunk_index,
        reranker=CrossEncoderReranker(device="auto"),
        top_k=TOP_K, pool_size=POOL_SIZE,
    )

    # ---- 1) chunks injectés : doc, score, thème, pertinence ----------------
    say("=" * 76)
    say(f"[{LABEL}] 1) CHUNKS INJECTES ({ga.MAX_SOURCES})")
    say("=" * 76)
    q_tokens = ga._tokenize(QUERY)
    sources_map, sources = [], []
    target_rank = None
    for rank, r in enumerate(reranked.results[:ga.MAX_SOURCES], start=1):
        chunk = chunk_index.get(str(r.chunk_id))
        if chunk is None:
            continue
        text = str(chunk.get("text", ""))
        overlap = len(q_tokens & ga._tokenize(text)) / max(len(q_tokens), 1)
        theme = str(getattr(r, "theme", "") or "?")
        say(f"  [{rank}] doc={r.document_id} p.{r.page_start}-{r.page_end} "
            f"reranker={getattr(r, 'reranker_score', float('nan')):.4f} "
            f"theme={theme!r} overlap_q={overlap:.3f} chars={len(text)}")
        say(f"      extrait: {text[:160]!r}")
        if r.document_id.startswith(TARGET_DOC[:10]):
            target_rank = rank
        sources_map.append({
            "chunk_id": str(r.chunk_id), "text": text,
            "document_id": str(r.document_id),
            "page_start": int(r.page_start), "page_end": int(r.page_end),
        })
        sources.append({
            "chunk_id": str(r.chunk_id), "document_id": str(r.document_id),
            "page_start": int(r.page_start), "page_end": int(r.page_end),
            "theme": theme,
        })

    # ---- 2) présence de la source cible ------------------------------------
    say("")
    say(f"[{LABEL}] 2) SOURCE {TARGET_DOC} PRESENTE : "
        f"{'OUI (rang ' + str(target_rank) + ')' if target_rank else 'NON'}")

    # ---- 3-4) prompt exact + réponse brute ---------------------------------
    provider = ga.OllamaProvider(model=MODEL)
    captured: dict[str, str] = {}
    orig_generate = provider.generate

    def wrapped_generate(prompt: str, temperature: float = 0.2) -> str:
        captured["prompt"] = prompt
        raw = orig_generate(prompt, temperature=temperature)
        captured["raw"] = raw
        return raw

    provider.generate = wrapped_generate
    t0 = time.perf_counter()
    claims = provider.generate_claims(QUERY, sources_map, temperature=0.2)
    t_gen = (time.perf_counter() - t0) * 1000

    say("")
    say(f"[{LABEL}] 3) PROMPT EXACT ({len(captured.get('prompt', ''))} chars)")
    say("-" * 76)
    say(captured.get("prompt", "<capture manquée>"))
    say("-" * 76)
    say("")
    say(f"[{LABEL}] 4) REPONSE BRUTE DU LLM (gen={t_gen:.0f} ms)")
    say("-" * 76)
    say(repr(captured.get("raw", "<capture manquée>")))
    say("-" * 76)
    Path(__file__).with_name(f"audit_selection_{LABEL}.txt").write_text(
        "\n".join(out), encoding="utf-8")
    print(f"[saved part1] files/audit_selection_{LABEL}.txt")

    # ---- 5) claims après parsing -------------------------------------------
    say("")
    say(f"[{LABEL}] 5) CLAIMS APRES PARSING ({len(candidates)})") if False else None
    candidates = ga.parse_claim_json(captured.get("raw", ""))
    say(f"[{LABEL}] 5) CLAIMS APRES PARSING ({len(candidates)})")
    for c in candidates:
        grounded = ga._claim_grounded(c, sources_map)
        say(f"  cits={c.citations} grounded={grounded} text={c.text[:100]!r}")

    # ---- 6) citations après grounding + sanitation -------------------------
    say("")
    say(f"[{LABEL}] 6) CLAIMS FINAUX (grounding + sanitation): {len(claims)}")
    for c in claims:
        say(f"  cits={c.citations} text={c.text[:100]!r}")
    answer = ga.decide_response(claims)
    say("")
    say(f"[{LABEL}] REPONSE RENDUE (decide_response) : {answer}")

    # ---- 7) scores NLI par citation retenue (gate -> _LAST_NLI_CACHE) ------
    ga._LAST_NLI_CACHE.clear()
    refus = ga._all_citations_unsupported(answer, sources_map)
    say("")
    say(f"[{LABEL}] 7) SCORES NLI PAR CITATION (gate, refus={refus})")
    for rec in ga.get_last_nli_scores():
        doc = sources_map[rec["source_index"] - 1]["document_id"]
        say(f"  source[{rec['source_index']}] doc={doc} "
            f"support={rec['support_score']:.4f} verdict={rec['verdict']}")
        say(f"    premise={rec['premise'][:110]!r}")
        say(f"    hypothesis={rec['hypothesis'][:110]!r}")

    # ---- 8) diagnostic ------------------------------------------------------
    say("")
    say(f"[{LABEL}] 8) DIAGNOSTIC SELECTION DE SOURCE")
    best = None
    for i, s in enumerate(sources_map, start=1):
        overlap = len(q_tokens & ga._tokenize(s["text"])) / max(len(q_tokens), 1)
        if best is None or overlap > best[1]:
            best = (i, overlap, s["document_id"])
    say(f"  meilleure source par overlap lexical : "
        f"[{best[0]}] doc={best[2]} overlap={best[1]:.3f}")
    cited = sorted({rec["source_index"] for rec in ga.get_last_nli_scores()})
    say(f"  sources réellement citées : {cited}")
    if best[0] not in cited:
        say(f"  => MEILLEURE SOURCE [best={best[0]}] NON CITEE par le LLM.")

    Path(__file__).with_name(f"audit_selection_{LABEL}.txt").write_text(
        "\n".join(out), encoding="utf-8")
    print(f"\n[saved] files/audit_selection_{LABEL}.txt")


if __name__ == "__main__":
    main()
