"""Sonde d'audit (temporaire, lecture seule) : fonctions + test troncature NLI."""
import json
import re

BASE = r"c:/marwaguidara/RAG-Citation-Validator/files"

# ---------------------------------------------------------------- 1) fonctions
src_ga = open(BASE + r"/generate_answer.py", encoding="utf-8").read()
for fname in ["extract_grounded_claims", "decide_response"]:
    m = re.search(r"def " + fname + r"\(.*?(?=\ndef |\nclass |\Z)", src_ga, re.S)
    print("=" * 30, fname)
    if m:
        body = re.sub(r'^\s*""".*?"""', "", m.group(0), flags=re.S)
        print("\n".join(l for l in body.splitlines() if l.strip())[:2600])

src_cv = open(BASE + r"/citation_verifier.py", encoding="utf-8").read()
print("=" * 30, "constants")
for c in re.findall(r"^(?:MAX_SEQ_LENGTH|DEFAULT_MODEL_NAME|SUPPORT_\w+|WEIGHT\w*)\s*=.*$", src_cv, re.M):
    print(c)

# ------------------------------------------------- 2) test de troncature NLI
import os
_cand = [BASE + r"/corpus/evaluation_report.json",
         BASE + r"/corpus/evaluation/evaluation_report.json"]
rep_path = next(p for p in _cand if os.path.exists(p) and
                b"config_d" in open(p, "rb").read()[:200000])
print("rapport détaillé utilisé:", rep_path)
rep = json.load(open(rep_path, encoding="utf-8"))
# ------------------------------------------------- 2) test de troncature NLI
rep = json.load(open(BASE + r"/corpus/evaluation_report.json", encoding="utf-8"))
chunks = json.load(open(BASE + r"/corpus/chunks.json", encoding="utf-8"))["chunks"]
cindex = {str(c["chunk_id"]): c for c in chunks}

q0 = rep["queries"][0]
print("config keys:", list(q0["configs"].keys()))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("roberta-large-mnli")

def probe(qentry):
    cfgs = qentry["configs"]
    key = [k for k in cfgs if k.startswith("config_d") or "verification" in str(k).lower()]
    k = key[0] if key else list(cfgs)[-1]
    details = cfgs[k].get("verification_details") or []
    answer = cfgs[k].get("answer") or cfgs[k].get("generated_answer") or ""
    print(f"\n### {qentry['query_id']} | detail_key={k} | n_details={len(details)} | answer_len={len(answer)}")
    kept = 0
    for i, det in enumerate(details[:6]):
        cid = str(det.get("chunk_id"))
        premise_raw = str(cindex[cid]["text"]) if cid in cindex else None
        hyp = det["claim_text"]
        sup = det["support_score"]
        verd = det["verdict"]
        res = "?"
        if premise_raw is not None:
            enc = tok(premise_raw, hyp, truncation=True, max_length=512,
                      return_token_type_ids=False)
            kept_ids = tok.build_inputs_with_special_tokens(enc["input_ids"])
            decoded = tok.decode(kept_ids, skip_special_tokens=True)
            words = [w for w in re.findall(r"[a-zA-Z]{4,}", hyp.lower())][:8]
            found = sum(1 for w in words if w in decoded.lower())
            res = f"{found}/{len(words)} mots survivent; tokens={min(len(enc['input_ids']),512)}"
            kept += (found == len(words))
        print(f"  [{i}] verdict={verd:<12} support={sup:.4f} prem_len={len(premise_raw or '')} -> {res}")
    return kept

total_survived = 0
for qe in rep["queries"][1:4]:
    total_survived += probe(qe)
print(f"\nclaims dont le texte survit entierement a la troncature: {total_survived}")
