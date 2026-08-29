"""Probe runtime API signatures for app.py orchestration (read-only)."""

import inspect
import json
import sys
from pathlib import Path

FILES = Path(r"c:/marwaguidara/RAG-Citation-Validator/files")
sys.path.insert(0, str(FILES))

import citation_verifier  # noqa: E402
import generate_answer  # noqa: E402
import hybrid_search  # noqa: E402
import rerank_results  # noqa: E402


def sig(obj: object) -> str:
    try:
        return f"{obj.__qualname__}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return repr(obj)


def fields(cls: type) -> list[str]:
    import dataclasses

    if dataclasses.is_dataclass(cls):
        return [f.name for f in dataclasses.fields(cls)]
    return []


out: list[str] = []
out.append("=== FUNCTIONS ===")
for mod, names in [
    (generate_answer, ["generate_answer", "build_provider", "load_chunk_text_index", "decide_response"]),
    (citation_verifier, ["verify_citations", "split_sentences"]),
    (rerank_results, ["rerank_results"]),
    (hybrid_search, ["get_engine"]),
]:
    out.append(f"-- {mod.__name__} --")
    for name in names:
        fn = getattr(mod, name, None)
        out.append(f"  {sig(fn) if fn else f'{name}: MISSING'}")

out.append("=== CLASSES ===")
classes = [
    ("AnswerResponse", generate_answer),
    ("Citation", generate_answer),
    ("Claim", generate_answer),
    ("ClaimSegment", citation_verifier),
]
for name, mod in classes:
    cls = getattr(mod, name, None)
    if cls is None:
        out.append(f"  {mod.__name__}.{name}: MISSING")
        continue
    out.append(f"  {cls.__module__}.{cls.__name__}: {fields(cls)}")

cvr = getattr(citation_verifier, "CitationVerificationResult", None)
if cvr is not None:
    out.append(f"  CitationVerificationResult: {fields(cvr)}")

srs = [getattr(hybrid_search, n, None) for n in ("SearchResult", "HybridResponse")]
for cls in srs:
    if cls is not None:
        out.append(f"  {hybrid_search.__name__}.{cls.__name__}: {fields(cls)}")

rr = getattr(rerank_results, "RerankedResult", None)
if rr is not None:
    out.append(f"  rerank_results.RerankedResult: {fields(rr)}")
for name in ("RerankedChunk", "RerankedResultChunk"):
    cls = getattr(rerank_results, name, None)
    if cls is not None:
        out.append(f"  rerank_results.{name}: {fields(cls)}")

out.append("=== MNLICitationVerifier methods ===")
mv = getattr(citation_verifier, "MNLICitationVerifier", None)
if mv is not None:
    out.append(f"  __init__{inspect.signature(mv.__init__)}")
    for name in sorted(vars(mv)):
        member = getattr(mv, name)
        if callable(member) and not name.startswith("_"):
            out.append(f"  {sig(member)}")

print("\n".join(out))

chunks = json.load(open(FILES / "corpus" / "chunks.json", encoding="utf-8"))
items = chunks["chunks"] if isinstance(chunks, dict) and "chunks" in chunks else chunks
print("CHUNK KEYS:", sorted(items[0].keys()))
