"""Vérification de cohérence : artefacts d'évaluation vs dashboard (usage unique)."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime


def ts(path: str) -> str:
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


BASE = r"c:/marwaguidara/RAG-Citation-Validator"
FILES = {
    "generate_answer.py": BASE + "/files/generate_answer.py",
    "evaluate_pipeline.py": BASE + "/files/evaluate_pipeline.py",
    "generate_evaluation_artifacts.py": BASE + "/files/generate_evaluation_artifacts.py",
    "comparison_table.json": BASE + "/files/corpus/comparison_table.json",
    "evaluation_report.json": BASE + "/files/corpus/evaluation_report.json",
    "evaluation_results.csv": BASE + "/files/corpus/evaluation_results.csv",
}

print("=== 1) Horodatages ===")
for name, path in FILES.items():
    print(f"{name:38s} {ts(path) if os.path.exists(path) else 'ABSENT'}")

print("\n=== 2) comparison_table.json (source dashboard ?) ===")
ct = json.load(open(FILES["comparison_table.json"], encoding="utf-8"))
for row in ct.get("configs", ct if isinstance(ct, list) else []):
    if isinstance(row, dict):
        keys = [k for k in row if any(s in k.lower() for s in ("faith", "citation", "config", "name"))]
        print({k: row[k] for k in keys})

print("\n=== 3) evaluation_report.json : agrégats par config ===")
rep = json.load(open(FILES["evaluation_report.json"], encoding="utf-8"))


def walk(obj: object, depth: int = 0) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            low = str(key).lower()
            if any(s in low for s in ("faithfulness", "citation_accuracy")) and not isinstance(val, (dict, list)):
                print("  " * depth + f"{key} = {val}")
            elif isinstance(val, (dict, list)) and depth < 3:
                print("  " * depth + f"[{key}]")
                walk(val, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:6]:
            walk(item, depth)


walk(rep)

print("\n=== 4) evaluation_results.csv : moyennes par config ===")
rows = list(csv.DictReader(open(FILES["evaluation_results.csv"], encoding="utf-8")))
cols = [c for c in rows[0] if any(s in c.lower() for s in ("faith", "citation", "recall", "mrr", "latency", "config", "system", "variant"))]
groups: dict[str, list[dict[str, str]]] = {}
gcol = next((c for c in cols if c.lower() in ("config", "configuration", "system", "variant")), None)
if gcol:
    for r in rows:
        groups.setdefault(r[gcol], []).append(r)
    for g, rs in groups.items():
        means = {c: sum(float(r[c]) for r in rs if r[c]) / len(rs) for c in cols if c != gcol and all(r.get(c) for r in rs)}
        fmt = {c: round(v, 4) for c, v in means.items()}
        print(f"{g}: n={len(rs)} {fmt}")
else:
    print("colonnes:", rows[0].keys())

print("\n=== 5) Recherche exhaustive '0.739|73.9|62.5|0.625' et '26.9|16.7' ===")
HITS_PATTERNS = ("73.9", "62.5", "26.9", "16.7")
for root, _dirs, names in os.walk(BASE):
    if any(seg in root for seg in (".git", "__pycache__", ".venv", "node_modules")):
        continue
    for name in names:
        if name.endswith((".json", ".csv", ".md", ".py", ".txt")):
            p = os.path.join(root, name)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if any(pat in line for pat in HITS_PATTERNS):
                            print(f"{p}:{i}: {line.strip()[:120]}")
            except OSError:
                pass
print("--- fin recherche ---")
