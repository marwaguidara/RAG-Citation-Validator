"""Audit latence NLI (duplication gate -> verify_citations) + sur-citation.

Instrumentation par monkey-patch (aucun composant modifié) :
  - MNLICitationVerifier.score_pairs -> compteur d'inférences + latence + hash
    des paires (premise, hypothesis) pour détecter les calculs redondants.
Scénarios mesurés :
  A. gate _all_citations_unsupported (generate_answer.py, comme generate_answer())
  B. verify_citations (citation_verifier.py, comme app.py)
  -> comparaison des paires : duplication = paires identiques calculées 2x.
"""
import hashlib
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from citation_verifier import (  # noqa: E402
    MNLICitationVerifier, SUPPORT_LOW, verify_citations,
)
import generate_answer as ga  # noqa: E402

QUERY = "What is Retrieval-Augmented Generation?"
POOL_SIZE, TOP_K, MODEL = 20, 5, "qwen2.5:3b"

# ---------------------------------------------------------------- instrumentation
NLI_CALLS: list[dict] = []          # une entrée par appel score_pairs
_orig_score_pairs = MNLICitationVerifier.score_pairs
_CURRENT_CALLER = ["?"]


def _instrumented_score_pairs(self, premises, hypotheses, batch_size=8):
    t0 = time.perf_counter()
    probs = _orig_score_pairs(self, premises, hypotheses, batch_size=batch_size)
    ms = (time.perf_counter() - t0) * 1000
    NLI_CALLS.append({
        "caller": _CURRENT_CALLER[0],
        "n_pairs": len(premises),
        "ms": ms,
        "keys": [
            hashlib.md5(f"{p}||{h}".encode()).hexdigest()[:10]
            for p, h in zip(premises, hypotheses)
        ],
        "premises": list(premises),
        "hypotheses": list(hypotheses),
    })
    return probs


MNLICitationVerifier.score_pairs = _instrumented_score_pairs


def main() -> None:
    corpus = ga.locate_corpus("chunks.json")
    chunk_index, _ = ga.load_chunk_text_index(corpus)
    engine = get_engine(EngineConfig(fetch_k=POOL_SIZE))
    t0 = time.perf_counter()
    hybrid = engine.search(query=QUERY, top_k=POOL_SIZE)
    t_hybrid = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    reranked = rerank_results(
        query=QUERY, hybrid_results=hybrid.results, chunk_index=chunk_index,
        reranker=CrossEncoderReranker(device="auto"),
        top_k=TOP_K, pool_size=POOL_SIZE,
    )
    t_rerank = (time.perf_counter() - t0) * 1000

    sources_map, sources = [], []
    for r in reranked.results[:ga.MAX_SOURCES]:
        chunk = chunk_index.get(str(r.chunk_id))
        if chunk is None:
            continue
        sources_map.append({
            "chunk_id": str(r.chunk_id), "text": str(chunk.get("text", "")),
            "document_id": str(r.document_id),
            "page_start": int(r.page_start), "page_end": int(r.page_end),
        })
        sources.append({
            "chunk_id": str(r.chunk_id), "document_id": str(r.document_id),
            "page_start": int(r.page_start), "page_end": int(r.page_end),
            "theme": "",
        })

    provider = ga.OllamaProvider(model=MODEL)
    t0 = time.perf_counter()
    claims = provider.generate_claims(QUERY, sources_map, temperature=0.2)
    t_gen = (time.perf_counter() - t0) * 1000
    answer = ga.decide_response(claims)

    print("=" * 76)
    print(f"REPONSE : {answer}")
    print("=" * 76)
    print("\nCITATIONS PAR CLAIM :")
    for c in claims:
        print(f"  citations={c.citations}  text={c.text[:90]!r}")

    _CURRENT_CALLER[0] = "A_gate_generate_answer"
    t0 = time.perf_counter()
    gate_refus = ga._all_citations_unsupported(answer, sources_map)
    t_gate = (time.perf_counter() - t0) * 1000

    verifier = MNLICitationVerifier(device="auto")
    _CURRENT_CALLER[0] = "B_verify_citations_app"
    t0 = time.perf_counter()
    verified, segments = verify_citations(
        query=QUERY, answer=answer, sources=sources,
        chunk_index=chunk_index, verifier=verifier,
    )
    t_verify = (time.perf_counter() - t0) * 1000
    _CURRENT_CALLER[0] = "?"
    print("LATENCES -> (voir fin)")
    print(f"LAT gen={t_gen:.0f}ms gate={t_gate:.0f}ms verify={t_verify:.0f}ms "
          f"hybrid={t_hybrid:.0f}ms rerank={t_rerank:.0f}ms refus={gate_refus}")
    save_report(NLI_CALLS, verified, segments, t_hybrid, t_rerank,
                t_gen, t_gate, t_verify, gate_refus, answer)


def save_report(nli_calls, verified, segments, t_hybrid, t_rerank,
                t_gen, t_gate, t_verify, gate_refus, answer) -> None:
    out = ["=" * 76, f"REPONSE : {answer}", "=" * 76, "",
           "LATENCES PAR ETAPE",
           f"  hybrid_search   : {t_hybrid:8.1f} ms",
           f"  rerank (BGE)    : {t_rerank:8.1f} ms",
           f"  generation LLM  : {t_gen:8.1f} ms",
           f"  A) gate NLI     : {t_gate:8.1f} ms  (refus={gate_refus})",
           f"  B) verify_cites : {t_verify:8.1f} ms", "",
           "INFERENCES NLI (paires premise/hypothesis)"]
    for call in nli_calls:
        out.append(f"  [{call['caller']}] {call['n_pairs']} paires, "
                   f"{call['ms']:.1f} ms")
        for k, h in zip(call["keys"], call["hypotheses"]):
            out.append(f"    hash={k}  hyp={h[:70]!r}")
    keys_a = {k for c in nli_calls if c["caller"].startswith("A_") for k in c["keys"]}
    keys_b = {k for c in nli_calls if c["caller"].startswith("B_") for k in c["keys"]}
    dup = keys_a & keys_b
    out += ["", "DUPLICATION A(gate) vs B(verify_citations)",
            f"  paires uniques A={len(keys_a)}  B={len(keys_b)}  "
            f"communes={len(dup)}  duplication={bool(dup)}",
            "", "SUPPORT NLI PAR CITATION (table verify_citations)"]
    for v in verified:
        out.append(f"  seg={v.claim_text[:40]!r} doc={v.document_id} "
                   f"p.{v.page_start} support={v.support_score} "
                   f"verdict={v.verdict}")
    n_claims = len(segments)
    n_cit = sum(len(s.citation_indices) for s in segments)
    out += ["",
            f"  claims={n_claims}  citations_totales={n_cit}  "
            f"moyenne={n_cit / max(n_claims, 1):.2f} citations/claim",
            f"  supports={[v.support_score for v in verified]}  "
            f"tous<{SUPPORT_LOW}={all(v.support_score < SUPPORT_LOW for v in verified)}"]
    report = "\n".join(out)
    print(report)
    Path(__file__).with_name("audit_latency_output.txt").write_text(
        report, encoding="utf-8")
    print("\n[saved] files/audit_latency_output.txt")


if __name__ == "__main__":
    main()
