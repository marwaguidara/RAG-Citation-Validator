"""Audit script v3: directly invoke Ollama provider internals to capture raw output."""
import json, os, sys, time
os.environ["RAG_TRACE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from generate_answer import (
    generate_answer, TemplateProvider, OllamaProvider,
    MAX_SOURCES, GROUNDING_COVERAGE, MIN_CLAIMS_FOR_ANSWER, MAX_CLAIMS,
    build_prompt, parse_claim_json, _claim_grounded, _tokenize,
    extract_grounded_claims,
)
from rerank_results import RerankedResult

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "chunks.json")

def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)["chunks"]

def find_rag_chunks(chunks, limit=5):
    matched = [c for c in chunks if "retrieval-augment" in c.get("text", "").lower()]
    if not matched:
        matched = [c for c in chunks if "augmented generation" in c.get("text", "").lower()]
    if not matched:
        matched = [c for c in chunks if c.get("theme") == "rag"]
    return matched[:limit]

def build_chunk_index(chunks):
    return {str(c["chunk_id"]): c for c in chunks}

def build_sources_map(chunks, chunk_index):
    rag = find_rag_chunks(chunks)
    sources_map = []
    for i, chunk in enumerate(rag[:MAX_SOURCES]):
        sources_map.append({
            "document_id": str(chunk["document_id"]),
            "page_start": int(chunk.get("page_start", 1)),
            "page_end": int(chunk.get("page_end", 1)),
            "text": str(chunk.get("text", "")),
            "_chunk_id": str(chunk["chunk_id"]),
        })
    return sources_map


def audit_ollama(query, sources_map):
    provider = OllamaProvider(model="qwen2.5:3b")
    print(f"Provider: name={provider.name}, _template=None: {provider._template is None}")
    prompt = build_prompt(query, sources_map)
    t0 = time.perf_counter()
    raw = provider.generate(prompt, temperature=0.2)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n--- RAW OLLAMA OUTPUT ({elapsed:.0f} ms) ---")
    print(repr(raw))
    print(f"Raw length: {len(raw)} chars")

    candidates = parse_claim_json(raw)
    print(f"\n--- PARSED CLAIMS: {len(candidates)} ---")
    for c in candidates:
        print(f"  text={repr(c.text[:100])} cits={c.citations}")

    print(f"\n--- GROUNDING CHECK (threshold={GROUNDING_COVERAGE}) ---")
    grounded, rejected = [], []
    for c in candidates:
        tokens = _tokenize(c.text)
        ok = _claim_grounded(c, sources_map)
        covs = []
        for idx in c.citations:
            if 1 <= idx <= len(sources_map):
                st = _tokenize(str(sources_map[idx-1].get("text", "")))
                cov = len(tokens & st) / max(len(tokens), 1) if tokens else 0
                covs.append((idx, round(cov, 2)))
            else:
                covs.append((idx, "OOR"))
        if ok:
            grounded.append(c)
            print(f"  OK  text={repr(c.text[:60])} cits={c.citations} covs={covs}")
        else:
            reason = "no cits" if not c.citations else ("all OOR" if all(v=="OOR" for _,v in covs) else f"cov<{GROUNDING_COVERAGE}")
            rejected.append(c)
            print(f"  REJ text={repr(c.text[:60])} cits={c.citations} covs={covs} reason={reason}")
    print(f"\nGrounded={len(grounded)} Rejected={len(rejected)} Total={len(candidates)}")
    if len(candidates) == 0:
        print("=> LLM output NOT parsed as JSON claims => fallback")
    elif len(grounded) == 0:
        print("=> All claims REJECTED by grounding filter => fallback")
    return raw, candidates, grounded, rejected


def main():
    chunks = load_chunks()
    sources_map = build_sources_map(chunks, None)
    query = "What is retrieval augmented generation?"
    print(f"Sources: {len(sources_map)}")
    for i, s in enumerate(sources_map):
        print(f"  [{i+1}] doc={s['document_id'][:40]} text={repr(s['text'][:100])}")
    print(f"GROUNDING_COVERAGE={GROUNDING_COVERAGE} MIN_CLAIMS_FOR_ANSWER={MIN_CLAIMS_FOR_ANSWER}")

    print("\n=== OLLAMA DIRECT AUDIT ===")
    raw, candidates, grounded, rejected = audit_ollama(query, sources_map)

    fb = extract_grounded_claims(query, sources_map)
    tp = TemplateProvider()
    tp_claims = tp.generate_claims(query, sources_map, temperature=0.2)
    print(f"\nFallback claims: {len(fb)}")
    print(f"Template claims: {len(tp_claims)}")
    print(f"Fallback == Template: {fb == tp_claims}")

    chunk_index = build_chunk_index(chunks)
    mock_results = []
    for i, s in enumerate(sources_map):
        mock_results.append(RerankedResult(
            final_rank=i+1, chunk_id=s["_chunk_id"], document_id=s["document_id"],
            theme="", page_start=s["page_start"], page_end=s["page_end"],
            hybrid_rank=i+1, hybrid_score=0.5, reranker_score=1.0,
            channels=["dense","bm25"], text_preview="",
        ))

    print("\n=== FULL PIPELINE ===")
    resp_t = generate_answer(query, mock_results, chunk_index, provider=TemplateProvider(), temperature=0.2)
    op = OllamaProvider(model="qwen2.5:3b")
    resp_o = generate_answer(query, mock_results, chunk_index, provider=op, temperature=0.2)
    print(f"Template answer: {repr(resp_t.answer[:200])}")
    print(f"Ollama  answer: {repr(resp_o.answer[:200])}")
    print(f"Identical: {resp_t.answer == resp_o.answer}")
    print(f"Template lat={resp_t.generation_latency}ms  Ollama lat={resp_o.generation_latency}ms")


if __name__ == "__main__":
    main()
