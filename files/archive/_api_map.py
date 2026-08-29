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
for cls in [
    generate_answer.AnswerResponse,
    generate_answer.Citation,
    generate_answer.Claim,
    citation_verifier.CitationVerificationResult,
    citation_verifier.ClaimSegment,
    hybrid_search.SearchResult,
    hybrid_search.HybridResponse,
]:
    out.append(f"  {cls.__module__}.{cls.__name__}: {fields(cls)}")
    init = getattr(cls, "__init__", None)
    if init is not None:
        out.append(f"      {sig(init)}")

out.append("=== MNLICitationVerifier methods ===")
for name in sorted(vars(citation_verifier.MNLICitationVerifier)):
    member = getattr(citation_verifier.MNLICitationVerifier, name)
    if callable(member):
        out.append(f"  {sig(member)}")

print("\n".join(out))

chunks = json.load(open(FILES / "corpus" / "chunks.json", encoding="utf-8"))
items = chunks["chunks"] if isinstance(chunks, dict) and "chunks" in chunks else chunks
first = items[0]
print("CHUNK KEYS:", sorted(first.keys()))

"""Inspection runtime des API des modules du pipeline (lecture seule)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

BASE = Path(r"c:/marwaguidara/RAG-Citation-Validator/files")


def dump_class_fields(tree: ast.Module, names: tuple[str, ...]) -> None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            print("CLASS", node.name)
            for sub in node.body:
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    print("   field:", sub.target.id, ast.unparse(sub.annotation))


def dump_function(tree: ast.Module, name: str) -> None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            args = ", ".join(a.arg for a in node.args.args)
            ret = ast.unparse(node.returns) if node.returns else ""
            print(f"def {name}({args}) -> {ret}")


# 1) citation_verifier
src_cv = (BASE / "citation_verifier.py").read_text(encoding="utf-8")
tree_cv = ast.parse(src_cv)
print("=== citation_verifier ===")
dump_class_fields(tree_cv, ("CitationVerificationResult", "ClaimSegment", "CitationVerdict"))
dump_function(tree_cv, "verify_citations")
dump_function(tree_cv, "compute_support_score")
dump_function(tree_cv, "verdict_for")

# 2) generate_answer
src_ga = (BASE / "generate_answer.py").read_text(encoding="utf-8")
tree_ga = ast.parse(src_ga)
print("=== generate_answer ===")
dump_class_fields(tree_ga, ("Claim", "AnswerResponse", "SourceRef", "Citation"))
dump_function(tree_ga, "generate_answer")
dump_function(tree_ga, "build_provider")
dump_function(tree_ga, "load_chunk_text_index")
dump_function(tree_ga, "extract_grounded_claims")
dump_function(tree_ga, "decide_response")

# 3) hybrid_search
src_hs = (BASE / "hybrid_search.py").read_text(encoding="utf-8")
tree_hs = ast.parse(src_hs)
print("=== hybrid_search ===")
dump_class_fields(
    tree_hs,
    ("EngineConfig", "HybridResult", "HybridSearchResponse", "LatencyBreakdown"),
)

# 4) rerank_results
src_rr = (BASE / "rerank_results.py").read_text(encoding="utf-8")
tree_rr = ast.parse(src_rr)
print("=== rerank_results ===")
dump_class_fields(tree_rr, ("RerankedResult", "SearchResult"))
dump_function(tree_rr, "rerank_results")

# 5) chunks.json structure
data = json.loads((BASE / "corpus" / "chunks.json").read_text(encoding="utf-8"))
print("=== chunks.json ===")
print("top-level:", type(data).__name__)
chunks = data.get("chunks") if isinstance(data, dict) else data
entry = chunks[0]
print("chunk keys:", list(entry.keys()))
sample = {k: str(v)[:60] for k, v in entry.items() if isinstance(v, (str, int, float))}
print(json.dumps(sample, ensure_ascii=False)[:600])

# 6) load_chunk_text_index return structure (runtime)
sys_path_note = "run via import below"
print(sys_path_note)
eule)."""
from __future__ import annotations

import dataclasses
import inspect
import sys

sys.path.insert(0, r"c:/marwaguidara/RAG-Citation-Validator/files")

import citation_verifier as cv  # noqa: E402
import generate_answer as ga  # noqa: E402
import hybrid_search as hs  # noqa: E402
import rerank_results as rr  # noqa: E402


def fields_of(cls: type) -> str:
    try:
        return ", ".join(f.name for f in dataclasses.fields(cls))
    except TypeError:
        return "<not a dataclass>"


def sig_of(fn: object) -> str:
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "?"


print("== generate_answer ==")
print("generate_answer", sig_of(ga.generate_answer))
print("decide_response", sig_of(ga.decide_response))
print("load_chunk_text_index", sig_of(ga.load_chunk_text_index))
print("Claim:", fields_of(getattr(ga, "Claim")))
print("AnswerResponse:", fields_of(getattr(ga, "AnswerResponse")))

print("== hybrid_search ==")
for name in ("get_engine", "EngineConfig"):
    obj = getattr(hs, name, None)
    print(name, sig_of(obj) if callable(obj) else fields_of(obj))
for name in ("HybridResponse", "SearchResult"):
    cls = getattr(hs, name, None)
    if cls is not None:
        print(f"{name}:", fields_of(cls))
eng = getattr(hs, "HybridSearchEngine", None)
if eng is not None:
    print("engine.search", sig_of(eng.search))

print("== rerank_results ==")
print("rerank_results", sig_of(rr.rerank_results))
print("CrossEncoderReranker.__init__", sig_of(rr.CrossEncoderReranker.__init__))
print("RerankedResult:", fields_of(getattr(rr, "RerankedResult")))
rc = getattr(rr, "RerankedChunk", None)
if rc is not None:
    print("RerankedChunk:", fields_of(rc))
rs = getattr(rr, "RankedChunk", None)
if rs is not None:
    print("RankedChunk:", fields_of(rs))

print("== citation_verifier ==")
print("verify_citations", sig_of(cv.verify_citations))
print("MNLICitationVerifier.__init__", sig_of(cv.MNLICitationVerifier.__init__))
print("CitationVerificationResult:", fields_of(getattr(cv, "CitationVerificationResult")))
for name in ("Citation", "SourceRef", "ClaimSegment"):
    cls = getattr(cv, name, None)
    if cls is not None:
        print(f"{name}:", fields_of(cls))

"""Inspection temporaire : carte des API pour app.py (lecture seule)."""
import ast
import json
from pathlib import Path

BASE = Path(r"c:/marwaguidara/RAG-Citation-Validator/files")

# 1) CitationVerificationResult + verify_citations + MNLICitationVerifier.__init__
src = (BASE / "citation_verifier.py").read_text(encoding="utf-8")
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in (
        "CitationVerificationResult", "MNLICitationVerifier", "ClaimSegment",
    ):
        print("CLASS", node.name)
        for sub in node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                print("   field:", sub.target.id, ast.unparse(sub.annotation))
            elif isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                print("   __init__ args:", [a.arg for a in sub.args.args])

fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_citations")
print("verify_citations args:", [(a.arg, ast.unparse(a.annotation) if a.annotation else "") for a in fn.args.args])
ret = ast.unparse(fn.returns) if fn.returns else ""
print("verify_citations ->", ret)

# 2) generate_answer: Claim, decide_response signature
src_ga = (BASE / "generate_answer.py").read_text(encoding="utf-8")
tree_ga = ast.parse(src_ga)
for node in tree_ga.body:
    if isinstance(node, ast.ClassDef) and node.name == "Claim":
        print("CLASS Claim:")
        for sub in node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                print("   field:", sub.target.id, ast.unparse(sub.annotation))
    if isinstance(node, ast.FunctionDef) and node.name in ("decide_response",):
        print("decide_response args:", [a.arg for a in node.args.args],
              "->", ast.unparse(node.returns) if node.returns else "")

# 3) hybrid_search: EngineConfig fields, SearchResponse/results fields, get_engine
src_hs = (BASE / "hybrid_search.py").read_text(encoding="utf-8")
tree_hs = ast.parse(src_hs)
for node in tree_hs.body:
    if isinstance(node, ast.ClassDef) and node.name in ("EngineConfig", "HybridResult", "HybridSearchResponse", "LatencyBreakdown"):
        print("CLASS", node.name)
        for sub in node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                print("   field:", sub.target.id, ast.unparse(sub.annotation))

# 4) rerank_results: RerankedResult fields + rerank_results signature
src_rr = (BASE / "rerank_results.py").read_text(encoding="utf-8")
tree_rr = ast.parse(src_rr)
for node in tree_rr.body:
    if isinstance(node, ast.ClassDef) and "Reranked" in node.name:
        print("CLASS", node.name)
        for sub in node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                print("   field:", sub.target.id, ast.unparse(sub.annotation))
    if isinstance(node, ast.FunctionDef) and node.name == "rerank_results":
        print("rerank_results args:", [a.arg for a in node.args.args])

# 5) chunks.json real structure
ch_path = BASE / "corpus" / "chunks.json"
data = json.loads(ch_path.read_text(encoding="utf-8"))
print("chunks.json top-level type:", type(data).__name__)
if isinstance(data, dict):
    print("keys:", list(data.keys())[:6])
    for k, v in list(data.items())[:1]:
        print("sample key:", k)
        if isinstance(v, dict):
            print("chunk keys:", list(v.keys()))
            print(json.dumps(v, ensure_ascii=False)[:500])
        elif isinstance(v, list):
            print("list len:", len(v), "first item:", json.dumps(v[0], ensure_ascii=False)[:500])
elif isinstance(data, list):
    print("len:", len(data))
    print("first item:", json.dumps(data[0], ensure_ascii=False)[:500])
