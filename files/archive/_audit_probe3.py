# -*- coding: utf-8 -*-
"""Sonde d'audit lecture seule #3 : structure du rapport detaille evaluate_pipeline.py."""
import json

p = r"c:/marwaguidara/RAG-Citation-Validator/files/corpus/evaluation/evaluation_report.json"
d = json.load(open(p, encoding="utf-8"))

cfgs = d["configurations"]
print("CONFIGS:", list(cfgs.keys()))
for name, c in cfgs.items():
    print("\n==", name, "keys:", list(c.keys()))
    for k, v in c.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print("   LIST", k, "len=", len(v), "item keys:", list(v[0].keys()))
        elif isinstance(v, dict):
            print("   DICT", k, list(v.keys())[:15])

# Chercher des resultats par citation (claim_text / support_score)
def walk(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        if "claim_text" in obj or "support_score" in obj:
            hits.append((path, list(obj.keys())))
        for k, v in obj.items():
            hits += walk(v, path + "/" + str(k))
    elif isinstance(obj, list) and obj:
        hits += walk(obj[0], path + "[0]")
    return hits

hits = walk(d)
print("\nPER-CITATION NODES (premier echantillon):")
for path, keys in hits[:5]:
    print("  ", path, "->", keys)
print("total noeuds par citation:", len(hits))