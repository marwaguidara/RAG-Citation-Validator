"""Vérifie si le grounding filter aurait rejeté le claim fautif."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "files")
from generate_answer import (  # noqa: E402
    GROUNDING_COVERAGE, Claim, _claim_grounded, load_chunk_text_index,
    locate_corpus,
)

CLAIM_TEXT = (
    "Retrieval-Augmented Generation (RAG) is a paradigm that allows Large "
    "Language Models (LLMs) to efficiently utilize external knowledge by "
    "integrating real-world images as additional references"
)
CITED = [1, 3]

path = locate_corpus("chunks.json")
index, _ = load_chunk_text_index(path)

sources_map = [
    {"document_id": "2402.12317v2", "page_start": 8, "page_end": 8,
     "text": index["d28bd74e-b4c7-5d0b-8b8c-1e93cacc42e7"]["text"]},
    {"document_id": "2309.15217v2", "page_start": 1, "page_end": 1,
     "text": index["0ef01ac4-80e3-5dcb-8498-35f17e2f1369"]["text"]},
    {"document_id": "2402.12317v2", "page_start": 1, "page_end": 1,
     "text": index["5f4162f9-1cfb-5d1d-9db0-d83e54d52a1b"]["text"]},
    {"document_id": "2506.06962v3", "page_start": 2, "page_end": 2,
     "text": index["9bb9381e-a4f1-5a09-aebb-5db962463c16"]["text"]},
    {"document_id": "2402.12317v2", "page_start": 1, "page_end": 1,
     "text": index["92b45aeb-ee0f-5284-9609-235407aa14ff"]["text"]},
]

claim = Claim(text=CLAIM_TEXT, citations=CITED)
print(f"GROUNDING_COVERAGE = {GROUNDING_COVERAGE}")
print(f"grounded (au moins une citation >= seuil) : {_claim_grounded(claim, sources_map)}")

from generate_answer import _tokenize  # noqa: E402
tokens = set(_tokenize(CLAIM_TEXT))
for idx in CITED:
    src_tokens = set(_tokenize(sources_map[idx - 1]["text"]))
    cov = len(tokens & src_tokens) / len(tokens)
    print(f"  couverture [{idx}] (doc={sources_map[idx-1]['document_id']} "
          f"p.{sources_map[idx-1]['page_start']}): {cov:.3f}")

src4_tokens = set(_tokenize(sources_map[3]["text"]))
print(f"  couverture vs SOURCE [4] (vrai conteneur du passage): "
      f"{len(tokens & src4_tokens) / len(tokens):.3f}")

