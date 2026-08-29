"""Ablation controlee : meme fenetre NLI, 3 hypotheses -> discriminer cas A/B.

Fenetre = pair [2,3] reelle (chunk EVOR, doc 2402.12317v2 p.1-1) ou le claim
verbatim a obtenu support=0.799 (Supported). On rejoue avec :
  H1 = claim verbatim (extrait de la source)          -> attendu eleve (cas OK)
  H2 = paraphrase CORRECTE du meme fait               -> cas B (limite NLI ?)
  H3 = contrefactuel faux (contenu absente de source) -> cas A (hallucination)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_answer import load_chunk_text_index, locate_corpus  # noqa: E402
from citation_verifier import (  # noqa: E402
    MNLICitationVerifier, compute_support_score, extract_local_premise,
    verdict_for,
)

CHUNK_ID = "5f4162f9-1cfb-5d1d-9db0-d83e54d52a1b"  # EVOR p.1-1 (source[3])

H1 = ("RAG has been successfully applied in code generation, where it achieves "
      "two to four times of execution accuracy compared to other methods.")
H2 = ("Large language models can reach markedly higher execution accuracy on "
      "coding tasks by leveraging retrieved documents, about two to four times "
      "better than baseline approaches.")
H3 = ("EVOR relies on generative adversarial networks and adversarial training "
      "to build a synthetic corpus of natural images.")

HYPOTHESES = {
    "H1 VERBATIM (dans la source)  ": H1,
    "H2 PARAPHRASE CORRECTE        ": H2,
    "H3 CONTREFACTUEL FAUX         ": H3,
}


def main() -> None:
    chunk_index, _ = load_chunk_text_index(locate_corpus("chunks.json"))
    chunk = chunk_index[CHUNK_ID]
    chunk_text = str(chunk["text"])
    verifier = MNLICitationVerifier(device="auto")

    print(f"chunk: doc={chunk['document_id']} p.{chunk['page_start']}-{chunk['page_end']} "
          f"({len(chunk_text)} chars)")

    print(f"\n{'='*82}\nABLATION SUR LA MEME FENETRE NLI (extract_local_premise)\n{'='*82}")
    for label, hyp in HYPOTHESES.items():
        premise = extract_local_premise(chunk_text, hyp)
        probs = verifier.score_pairs([premise], [hyp])[0]
        contradiction, neutral, entailment = probs
        support = compute_support_score(contradiction, neutral, entailment)
        print(f"\n--- {label}")
        print(f"    hypothesis : {hyp}")
        print(f"    fenetre ({len(premise)} chars) : {' '.join(premise.split())[:220]}...")
        print(f"    entailment={entailment:.4f} neutral={neutral:.4f} "
              f"contradiction={contradiction:.4f} support={support:.4f} "
              f"verdict={verdict_for(support)}")


if __name__ == "__main__":
    main()