"""Dump les lignes construisant `sources` dans generate_evaluation_artifacts.py (audit app.py)."""
import re
from pathlib import Path

src = Path(r"c:/marwaguidara/RAG-Citation-Validator/files/generate_evaluation_artifacts.py").read_text(encoding="utf-8")
lines = src.splitlines()
out = []
for i, line in enumerate(lines, 1):
    if re.search(r"sources(\s*=|\.append|\.text|\[)", line) or "generate_claim_answer" in line or "run_verification" in line:
        out.append(f"{i:5d} | {line.rstrip()[:170]}")
Path(__file__).with_name("_sources_map.txt").write_text("\n".join(out), encoding="utf-8")
print("written", len(out))
