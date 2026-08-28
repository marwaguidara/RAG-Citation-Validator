import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_answer import Claim, OllamaProvider, render_claims_answer  # noqa: E402

cases = [
    Claim(text="RAG allows LLMs to use external knowledge, as described in [1] and [3].",
          citations=[2, 3]),
    Claim(text="RAG retrieves documents [1][2][3][4][5] to ground generation.",
          citations=[1, 2, 3, 4, 5]),
    Claim(text="Bad claim citing [9] only.", citations=[9]),
]
out = OllamaProvider._sanitize_claim_citations(cases, 5)
for o in out:
    print("cits=", o.citations, " text=", repr(o.text))
print("RENDU :", render_claims_answer(out))
