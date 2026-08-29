"""Audit lecture seule : taux de rejet et impact des règles de generate_answer.py."""
import json
from collections import Counter

BASE = r"c:/marwaguidara/RAG-Citation-Validator"
rep = json.load(open(BASE + "/files/corpus/evaluation_report.json", encoding="utf-8"))

queries = rep["queries"]
n = len(queries)
print(f"N_QUERIES = {n}")

# ---- agrégats config D ----
agg = rep["aggregate"]
d_cfg = agg["Hybrid + Rerank + Verification"] if isinstance(agg, dict) else None
print("AGG config D:", json.dumps(d_cfg, ensure_ascii=False)[:800])

# ---- structure d'une verification_details ----
cfg_d = "Hybrid + Rerank + Verification"
q0 = queries[0]["configs"][cfg_d]
vd0 = q0.get("verification_details")
print("\nVD0 TYPE:", type(vd0).__name__)
# ---- mesure complète sur les 30 requêtes ----
import re

tot = Counter()
per_q = []
refusal_pat = re.compile(r"(je ne peux pas|impossible de r|aucune information|pas d'information|ne contient pas|insuffisant)", re.I)

for q in queries:
    c = q["configs"][cfg_d]
    vd = c.get("verification_details") or {}
    vc = vd.get("verdict_counts", {})
    n_cit = vd.get("citations_verified", 0)
    ans = (c.get("answer") or "")
    has_cit = bool(re.search(r"\[\d+\]", ans))
    refused = bool(refusal_pat.search(ans)) and not has_cit
    for k, v in vc.items():
        tot[k] += v
    tot["_citations"] += n_cit
    tot["_answers"] += 1
    if refused: tot["_refused"] += 1
    if not has_cit: tot["_no_citation"] += 1
    per_q.append((q["query_id"], n_cit, dict(vc), len(ans), has_cit, refused))

print("\nTOTALS:")
for k in ("_answers", "_citations", "_refused", "_no_citation",
          "Supported", "Weak Support", "Unsupported"):
    print(f"  {k} = {tot.get(k)}")

S = tot["Supported"]; W = tot["Weak Support"]; U = tot["Unsupported"]
T = S + W + U
print(f"\nCand formulas vs aggregate (faith=0.1927, citacc=0.1583):")
if T:
    print(f"  S/T                  = {S/T:.4f}")
    print(f"  (S+W)/T              = {(S+W)/T:.4f}")
    print(f"  (S+0.5W)/T           = {(S+0.5*W)/T:.4f}")
print(f"  refusal rate         = {tot['_refused']/n:.4f}")
print(f"  no-citation rate     = {tot['_no_citation']/n:.4f}")

# ---- détail : une entrée verification_details complète ----
vd_full = queries[2]["configs"][cfg_d].get("verification_details") or {}
print("\nVD KEYS:", list(vd_full.keys()))
items = vd_full.get("items") or vd_full.get("citations") or vd_full.get("results") or []
print("ITEMS TYPE/LEN:", type(items).__name__, len(items) if isinstance(items, list) else "-")
if isinstance(items, list) and items:
    print("ITEM[0]:", json.dumps(items[0], ensure_ascii=False)[:1500])
else:
    # autre structure possible
    for k in vd_full:
        v = vd_full[k]
        if isinstance(v, list) and v:
            print(f"FIELD {k}: list len={len(v)}")
            print(f"  [0] = {json.dumps(v[0], ensure_ascii=False)[:1200]}")

print("\nPER QUERY (id, citations, verdicts, ans_len, has_cit):")
for row in per_q:
    print(" ", row)


