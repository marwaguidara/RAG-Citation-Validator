# -*- coding: utf-8 -*-
"""Audit lecture seule #2 : position des claims dans leur chunk + correlation verdicts."""
import json, re, sys
from collections import Counter

BASE = r"c:/marwaguidara/RAG-Citation-Validator"
sys.path.insert(0, BASE + "/files")

rep = json.load(open(BASE + "/files/corpus/evaluation_report.json", encoding="utf-8"))
chunks_raw = json.load(open(BASE + "/files/corpus/chunks.json", encoding="utf-8"))
if isinstance(chunks_raw, dict):
    chunks_raw = chunks_raw.get("chunks") or list(chunks_raw.values())
# index chunk -> texte
chunk_texts = {}
for ch in chunks_raw:
    cid = ch.get("chunk_id") or ch.get("id")
    chunk_texts[str(cid)] = ch.get("text") or ch.get("content") or ""
print("CHUNKS LOADED:", len(chunk_texts))

def norm(t):
    return " ".join(t.split()).lower()

cfg_d = "Hybrid + Rerank + Verification"
rows = []
for q in rep["queries"]:
    c = q["configs"][cfg_d]
    ans = c.get("answer") or ""
    vd = c.get("verification_details") or {}
    vc = vd.get("verdict_counts", {})
    # segmentation approximative : phrases reliees a leurs [N]
    for m in re.finditer(r"([^.\n]+?)\s*\[(\d+)\]", ans):
        sent = norm(m.group(1))
        idx = int(m.group(2)) - 1
        rows.append({
            "qid": q["query_id"],
            "sent": sent[:60],
            "cited": idx,
            "verdict_totals": sum(vc.values()),
            "weak": vc.get("Weak Support", 0),
            "supp": vc.get("Supported", 0),
            "unsupp": vc.get("Unsupported", 0),
        })

print("CLAIM/CITATION PAIRS PARSED:", len(rows))

# Il faut aussi recuperer l'ordre des chunks utilises comme sources pour chaque query.
# Cherchons si le rapport stocke retrieved_chunks / sources par requete/config :
q0 = rep["queries"][0]["configs"][cfg_d]
print("\nCONFIG D KEYS:", list(q0.keys()))
for k, v in q0.items():
    if isinstance(v, list) and v and isinstance(v[0], dict):
        print(f"LIST FIELD '{k}': len={len(v)}, keys={list(v[0].keys())}")
    elif isinstance(v, dict):
        print(f"DICT FIELD '{k}': keys={list(v.keys())}")

# Test litteral : pour quelques claims, retrouver s'ils apparaissent mot pour mot
# dans AU MOINS UN chunk du corpus, et a quelle position relative.
found_anywhere = 0
positions = []
for r in rows[:80]:
    s = r["sent"]
    if len(s) < 25:
        continue
    hit = None
    for cid, txt in chunk_texts.items():
        nt = norm(txt)
        pos = nt.find(s)
        if pos >= 0:
            hit = (cid, pos / max(len(nt), 1))
            break
    if hit:
        found_anywhere += 1
        positions.append(hit[1])

print(f"\nVERBATIM FOUND IN SOME CHUNK: {found_anywhere}/{len([r for r in rows[:80] if len(r['sent'])>=25])}")
if positions:
    positions.sort()
    import statistics
    print("POSITION RATIO (0=start, 1=end) QUANTILES:")
    print("   min=%.2f med=%.2f max=%.2f" % (
        positions[0], statistics.median(positions), positions[-1]))
