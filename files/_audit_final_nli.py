# -*- coding: utf-8 -*-
"""Sonde finale d'audit (lecture seule des artefacts, aucun recalcul du pipeline).

Objectif : elucider pourquoi 148/171 citations extraites MOT POUR MOT des chunks
sont jugees Unsupported par le verificateur NLI.

Protocole, sur les requetes q01..q08 (config D = Hybrid + Rerank + Verification):
  1. Parser chaque reponse en claims cites ([N] phrase).
  2. Retrouver litteralement chaque claim dans un des retrieved_chunk_ids.
  3. Scorer la paire (premise=chunk, hypothesis=claim) avec roberta-large-mnli :
     (a) premise ENTIere  -> plafond theorique du support ;
     (b) premise telle que le verificateur la passe reellement
         (pair encoding, truncation=True, max_length=512)
         -> support effectif.
  4. Correler : position du claim dans le chunk x survie apres troncature
     x probabilite d'entailment.
"""
import json
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = r"c:/marwaguidara/RAG-Citation-Validator/files"
REPORT = BASE + r"/corpus/evaluation_report.json"
CHUNKS = BASE + r"/corpus/chunks.json"
MODEL = "roberta-large-mnli"
MAX_LEN = 512

rep = json.load(open(REPORT, encoding="utf-8"))
chunks = json.load(open(CHUNKS, encoding="utf-8"))["chunks"]
cindex = {str(c["chunk_id"]): c["text"] for c in chunks}

print("Loading", MODEL, "...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
LABEL_ENTAIL = next(i for i, l in enumerate(model.config.id2label.values())
                    if l.lower() == "entailment")


def ent_prob(premise: str, hyp: str) -> tuple[float, int]:
    """P(entailment) + nb de tokens premisse conserves (meme encodage que le verif.)."""
    enc = tok(premise, hyp, truncation=True, max_length=MAX_LEN,
              return_token_type_ids=False)
    kept_premise_tokens = min(
        len(enc["input_ids"]), MAX_LEN
    ) - len(tok(hyp)["input_ids"]) - 2
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor([enc["input_ids"]])).logits, dim=-1
        )[0]
    return float(probs[LABEL_ENTAIL]), max(kept_premise_tokens, 0)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize(t: str) -> str:
    return " ".join(t.split()).lower()


rows = []          # (qid, idx, prem_chars, rel_pos, kept_tok, p_full, p_eff)
full_count = eff_count = survive_full = survive_eff = 0
NQ = 8

for q in rep["queries"][:NQ]:
    cfg = q["configs"]["Hybrid + Rerank + Verification"]
    answer = cfg.get("answer") or cfg.get("generated_answer") or ""
    chunk_ids = [str(c) for c in cfg.get("retrieved_chunk_ids", [])]
    # Claims = segments "[N] phrase"
    segs = [(m.group(1), m.group(2).strip())
            for m in re.finditer(r"\[(\d+)\]\s*([^\[]+)", answer)]
    if not segs:
        print(f"{q['query_id']}: pas de segments [N] (refus ?) -> skip")
        continue
    for i, (num, claim) in enumerate(segs[:6]):
        # retrouver le chunk contenant litteralement le claim
        norm_claim = normalize(claim)
        home = None
        for cid in chunk_ids:
            text = cindex.get(cid, "")
            if norm_claim and norm_claim in normalize(text):
                home = (cid, text)
                break
        if home is None:
            print(f"{q['query_id']} [{num}] claim NON retrouve litteralement !")
            continue
        cid, text = home
        pos = normalize(text).find(norm_claim)
        rel_pos = pos / max(len(normalize(text)), 1)
        p_full, tok_full_all = ent_prob(text, claim)
        p_eff, kept_tok = ent_prob(text, claim)  # meme appel que le verificateur
        # survie : nb de tokens de la premisse conserves vs taille totale premisse
        prem_total = len(tok(text)["input_ids"])
        survived = kept_tok >= prem_total - 5
        rows.append((q["query_id"], num, len(text), round(rel_pos, 2),
                     f"{kept_tok}/{prem_total}", round(p_full, 4),
                     round(p_eff, 4)))
        full_count += 1
        survive_full += (p_full >= 0.40)
        survive_eff += (p_eff >= 0.40)

print("\nqid src prem_chars rel_pos kept_tok/prem_toks p_FULL p_EFFECTIVE")
for r in rows:
    print(" ", r)

n = len(rows)
if n:
    outside = sum(1 for r in rows if float(r[3]) > 0.55)
    low_eff = sum(1 for r in rows if r[6] < 0.40)
    both = sum(1 for r in rows if float(r[3]) > 0.55 and r[6] < 0.40)
    print(f"\nechantillon={n}")
    print(f"claims a position>55% dans le chunk : {outside} ({outside/n:.0%})")
    print(f"p_effective<0.40 (=Unsupported)      : {low_eff} ({low_eff/n:.0%})")
    print(f"position>55% ET p_eff<0.40           : {both} ({both/n:.0%})")
