# -*- coding: utf-8 -*-
"""Sonde NLI decisive : un claim reel 'Unsupported' avec chunk entier vs tronque."""
import json, re, sys

BASE = r"c:/marwaguidara/RAG-Citation-Validator"
sys.path.insert(0, BASE + "/files")

rep = json.load(open(BASE + "/files/corpus/evaluation_report.json", encoding="utf-8"))
chunks_raw = json.load(open(BASE + "/files/corpus/chunks.json", encoding="utf-8"))
if isinstance(chunks_raw, dict):
    chunks_raw = chunks_raw.get("chunks") or list(chunks_raw.values())
chunk_texts = {str(c.get("chunk_id") or c.get("id")): (c.get("text") or c.get("content") or "")
               for c in chunks_raw}

cfg_d = "Hybrid + Rerank + Verification"

def norm(t):
    return " ".join(t.split())

q03 = rep["queries"][2]["configs"][cfg_d]
ans = norm(q03["answer"])
ids = q03["retrieved_chunk_ids"]
print("Q03 ANSWER:", ans[:300])
print("\nRETRIEVED CHUNKS:", ids)
for cid in ids:
    txt = chunk_texts.get(str(cid), "")
    print(f"   {cid}: len_chars={len(txt)}")

# Prendre une phrase citee, retrouver son chunk hote.
import itertools
pair = None
for m in re.finditer(r"([^.]+?)\s*\[(\d+)\]", ans):
    sent = m.group(1).strip()
    idx = int(m.group(2)) - 1
    if len(sent) < 40:
        continue
    for j, cid in enumerate(ids):
        host = norm(chunk_texts.get(str(cid), ""))
        pos = host.find(sent)
        if pos >= 0:
            pair = (sent, str(cid), pos, len(host))
            break
    if pair:
        break

print(f"\nPAIR FOUND: sent={pair[0][:80]!r}... chunk={pair[1]} char_pos={pair[2]}/{pair[3]} "
      f"ratio={pair[2]/max(pair[3],1):.2f}")

sent, cid, pos, total = pair
host = chunk_texts[cid]

import inspect  # noqa: E402
from citation_verifier import MNLICitationVerifier  # noqa: E402

verifier = MNLICitationVerifier()
sig = inspect.signature(verifier.score_pairs)
print("\nscore_pairs SIGNATURE:", sig)

# Variantes de premisse pour isoler l'effet de troncature / granularite :
variants = {
    "full_chunk": host,
    "first_1024_chars": host[:1024],
    "window_+-250": host[max(0, pos - 250): pos + len(sent) + 250],
}
pairs = [(name, text, sent) for name, text in variants.items()]
premises = [t for _, t, _ in pairs]
hyps = [s for _, _, s in pairs]
res = verifier.score_pairs(premises, hyps)
print("SCORE_PAIRS RAW:", json.dumps(res, ensure_ascii=False, default=str)[:1500])

def to_str(r):
    try:
        return [f"{x['premise_name']}: {json.dumps(x, ensure_ascii=False)[:200]}" for x in r]
    except Exception:
        return [str(x)[:200] for x in r]

for line in to_str(res):
    print("  ", line)

