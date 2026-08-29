"""Smoke test temporaire : exécute run_pipeline via app.py (mode bare)."""
import sys
import time

sys.path.insert(0, r"c:/marwaguidara/RAG-Citation-Validator")

import app  # noqa: E402

t0 = time.perf_counter()
result = app.run_pipeline(
    "What is retrieval-augmented generation?",
    pool_size=10,
    top_k=5,
    provider_name="template",
    model_name="llama3.1",
    temperature=0.2,
)
print("LOAD_MS:", round(time.perf_counter() - t0, 1))
print("ANSWER_HEAD:", result["answer"][:220])
print("N_VERIFIED:", len(result["verified"]))
for v in result["verified"][:10]:
    print(" -", v.verdict, "| score", round(v.support_score, 3), "|",
          v.document_id, "p.", v.page_start, "-", v.page_end)
print("LAT hybrid/rerank/gen/total(ms):",
      round(result["hybrid_latency_ms"], 1),
      round(result["rerank_latency_ms"], 1),
      round(result["generation_latency_ms"], 1),
      round(result["total_ms"], 1))