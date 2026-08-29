# -*- coding: utf-8 -*-
"""Sonde #4 : structure complete des entrees par requete + colonnes CSV."""
import json
import csv

BASE = r"c:/marwaguidara/RAG-Citation-Validator"
rep = json.load(open(BASE + "/files/corpus/evaluation_report.json", encoding="utf-8"))

q0 = rep["queries"][2]
print("QUERY KEYS:", list(q0.keys()))
cfg_d = "Hybrid + Rerank + Verification"
c = q0["configs"][cfg_d]
for k, v in c.items():
    if isinstance(v, str):
        print(f"  STR {k}: len={len(v)} :: {v[:220]!r}")
    elif isinstance(v, list):
        print(f"  LIST {k}: len={len(v)} :: [0]={json.dumps(v[0], ensure_ascii=False)[:300] if v and isinstance(v[0], (dict,str)) else v[:3]}")
    elif isinstance(v, dict):
        print(f"  DICT {k}: {json.dumps(v, ensure_ascii=False)[:400]}")
    else:
        print(f"  {k} = {v}")

with open(BASE + "/files/corpus/evaluation_results.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    row_d = None
    rows = []
    for r in reader:
        rows.append(r)
print("\nCSV HEADER:", header)
d_idx = [i for i, h in enumerate(header) if "config" in h.lower() or "d" == h.lower()]
print("n_rows:", len(rows))
print("row example:", json.dumps(rows[2], ensure_ascii=False)[:600])