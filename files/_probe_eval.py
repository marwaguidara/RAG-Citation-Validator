"""Probe: how the evaluator builds sources + chunk_index for verify_citations."""
import re

PATH = r"c:/marwaguidara/RAG-Citation-Validator/files/generate_evaluation_artifacts.py"
lines = open(PATH, encoding="utf-8").read().splitlines()

pat = re.compile(
    r"def |sources\.append|chunk_index\[|\"text\"|'text'|page_start|document_id|theme|verify_citations\(|generate_answer\(",
)
out = [f"{i}: {l}" for i, l in enumerate(lines, 1) if pat.search(l)]
print("\n".join(out))
