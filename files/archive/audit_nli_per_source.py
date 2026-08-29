"""Scores NLI réels d'un claim contre les 5 sources (prémisse = gate).

Répond à : pourquoi la gate rejette le claim-réponse de Q1 même réparé ?
Aucun composant modifié.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_answer import (  # noqa: E402
    _gate_premise, compute_support_score, locate_corpus,
)
from citation_verifier import MNLICitationVerifier  # noqa: E402

CLAIM = (
    "Retrieval-Augmented Generation (RAG) is a technique that allows large "
    "language models (LLMs) to utilize external knowledge beyond their "
    "training data by retrieving relevant documents."
)

if __name__ == "__main__":
    corpus = locate_corpus("chunks.json")
    from generate_answer import load_chunk_text_index
    chunk_index, _ = load_chunk_text_index(corpus)
    # Rerank fixe (audit précédent) : les 5 chunk_id injectés pour Q1
    ids = [
        "d28bd74e-b4c7-5d0b-8b8c-1e93cacc42e7",  # [1] 2402.12317v2 p.8 (def RAG)
        "0ef01ac4-80e3-5dcb-8498-35f17e2f1369",  # [2] 2309.15217v2 p.1 (Ragas)
        "5f4162f9-1cfb-5d1d-9db0-d83e54d52a1b",  # [3] 2402.12317v2 p.1 (EVOR)
        "9bb9381e-a4f1-5a09-aebb-5db962463c16",  # [4] 2506.06962v3 p.2 (images)
        "92b45aeb-ee0f-5284-9609-235407aa14ff",  # [5] 2402.12317v2 p.1 (RACG)
    ]
    premises, labels = [], []
    for n, cid in enumerate(ids, start=1):
        text = str(chunk_index[cid]["text"])
        premises.append(_gate_premise(text, CLAIM))
        labels.append(f"[{n}] {cid[:8]} ({len(text)} chars)")
    verifier = MNLICitationVerifier(device="auto")
    probs = verifier.score_pairs(premises, [CLAIM] * len(ids))
    print(f"CLAIM: {CLAIM!r}\n")
    for label, (c, ne, e) in zip(labels, probs):
        s = compute_support_score(c, ne, e)
        verdict = "Supported" if s >= 0.70 else (
            "Weak" if s >= 0.40 else "Unsupported")
        print(f"  {label:34s} support={s:.4f} ({verdict})  "
              f"entail={e:.3f} neutral={ne:.3f} contra={c:.3f}")
