"""REPLAY AUDIT : rejoue la sortie brute Ollama capturée (échec de parse)
sur le parseur corrigé de generate_answer.py. Aucun appel Ollama.
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from generate_answer import (
    parse_claim_json, _claim_grounded, _tokenize, GROUNDING_COVERAGE,
    MIN_CLAIMS_FOR_ANSWER, MAX_CLAIMS,
)

# --- Sortie brute Ollama EXACTE capturée dans audit_v3_output.txt ---
RAW_CAPTURED = (
    '{"claims":[{"text":"Retrieval-Augmented Generation (RAG) is a technique used by '
    'THaMES to improve text generation, combining model predictions with external data '
    'retrieval.","citations":[3,4,2512.13930v1]},{"text":"In-Context Learning (ICL) is '
    'another technique used by THaMES to mitigate hallucinations, which involves using '
    'context to improve model performance.","citations":[1,2512.13930v1]},{"text":'
    '"Parameter-efficient fine-tuning methods, like Peft, are used to improve the '
    'efficiency of large language models without sacrificing performance.",'
    '"citations":[2409.11353v3,2022]}]}'
)

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "chunks.json")

def main():
    print("=" * 70)
    print("REPLAY: captured raw Ollama output -> fixed parser")
    print("=" * 70)
    print("RAW:", RAW_CAPTURED[:120], "...")

    claims = parse_claim_json(RAW_CAPTURED)
    print(f"\nParsed claims: {len(claims)}")
    for c in claims:
        print(f"  text={c.text[:70]!r} cits={c.citations}")

    # Grounding check against the same RAG sources used by audit_v3
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    matched = [c for c in chunks if "retrieval-augment" in c.get("text", "").lower()]
    if not matched:
        matched = [c for c in chunks if "augmented generation" in c.get("text", "").lower()]
    if not matched:
        matched = [c for c in chunks if c.get("theme") == "rag"]
    sources = [
        {"document_id": str(c["document_id"]), "text": str(c.get("text", ""))}
        for c in matched[:5]
    ]
    print(f"\nSources: {len(sources)}  GROUNDING_COVERAGE={GROUNDING_COVERAGE}")
    print(f"MIN_CLAIMS_FOR_ANSWER={MIN_CLAIMS_FOR_ANSWER} MAX_CLAIMS={MAX_CLAIMS}")

    grounded, rejected = [], []
    for c in claims:
        tokens = _tokenize(c.text)
        ok = _claim_grounded(c, sources)
        covs = []
        for idx in c.citations:
            if 1 <= idx <= len(sources):
                st = _tokenize(str(sources[idx - 1].get("text", "")))
                cov = len(tokens & st) / max(len(tokens), 1) if tokens else 0
                covs.append((idx, round(cov, 2)))
            else:
                covs.append((idx, "OOR"))
        if ok:
            grounded.append(c)
            print(f"  OK  {c.text[:60]!r} cits={c.citations} covs={covs}")
        else:
            if not c.citations:
                reason = "no valid cits (all unparseable/OOR)"
            elif all(v == "OOR" for _, v in covs):
                reason = "all citations out of range"
            else:
                reason = f"max coverage < {GROUNDING_COVERAGE}"
            rejected.append((c, reason, covs))
            print(f"  REJ {c.text[:60]!r} cits={c.citations} covs={covs} reason={reason}")

    print(f"\nGrounded={len(grounded)} Rejected={len(rejected)} Total={len(claims)}")
    if len(grounded) >= MIN_CLAIMS_FOR_ANSWER:
        print("=> DECISION: LLM claims USED (no fallback) -- fix effective")
    elif claims:
        print("=> DECISION: all claims rejected by grounding filter => fallback")
    else:
        print("=> DECISION: parse produced 0 claims => fallback (STILL BROKEN)")

if __name__ == "__main__":
    main()
