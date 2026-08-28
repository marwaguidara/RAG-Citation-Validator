"""Sonde d'API unique pour app.py (lecture seule, jetable)."""
import sys
import inspect
import dataclasses

sys.path.insert(0, r"c:/marwaguidara/RAG-Citation-Validator/files")


def sig(mod_name: str, fn_name: str) -> None:
    mod = __import__(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        print(f"{mod_name}.{fn_name}: ABSENT")
        return
    try:
        print(f"{mod_name}.{fn_name}{inspect.signature(fn)}")
    except (TypeError, ValueError) as exc:
        print(f"{mod_name}.{fn_name}: <no sig> {exc}")


def fields(mod_name: str, cls_name: str) -> None:
    mod = __import__(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None or not dataclasses.is_dataclass(cls):
        print(f"{mod_name}.{cls_name}: ABSENT or not dataclass")
        return
    names = [f.name for f in dataclasses.fields(cls)]
    print(f"{mod_name}.{cls_name}: {names}")


print("== SIGNATURES ==")
for m, f in [
    ("hybrid_search", "get_engine"),
    ("hybrid_search", "EngineConfig"),
    ("hybrid_search", "load_chunk_text_index"),
    ("rerank_results", "rerank_results"),
    ("rerank_results", "CrossEncoderReranker"),
    ("generate_answer", "generate_answer"),
    ("generate_answer", "build_provider"),
    ("generate_answer", "extract_grounded_claims"),
    ("generate_answer", "decide_response"),
    ("generate_answer", "locate_corpus"),
    ("citation_verifier", "verify_citations"),
    ("citation_verifier", "MNLICitationVerifier"),
    ("citation_verifier", "segment_claims"),
]:
    sig(m, f)

print("== DATACLASS FIELDS ==")
for m, c in [
    ("hybrid_search", "SearchResult"),
    ("hybrid_search", "HybridResponse"),
    ("hybrid_search", "EngineConfig"),
    ("rerank_results", "RerankedResult"),
    ("generate_answer", "AnswerResponse"),
    ("generate_answer", "Claim"),
    ("generate_answer", "Citation"),
    ("generate_answer", "SourceRef"),
    ("citation_verifier", "ClaimSegment"),
    ("citation_verifier", "VerificationResult"),
    ("citation_verifier", "CitationVerificationResult"),
]:
    fields(m, c)

print("== CHUNK INDEX FORMAT ==")
from generate_answer import load_chunk_text_index  # noqa: E402

idx, sha = load_chunk_text_index(r"c:/marwaguidara/RAG-Citation-Validator/files/corpus/chunks.json")
first_key = next(iter(idx))
entry = idx[first_key]
if isinstance(entry, dict):
    print(f"chunk_index[{first_key!r}] keys: {list(entry.keys())}")
else:
    print(f"chunk_index[{first_key!r}] type={type(entry).__name__} value={str(entry)[:120]}")

print("== ENGINE SEARCH ==")
eng_cls = getattr(__import__("hybrid_search"), "get_engine")
print("get_engine sig above; engine.search:")
