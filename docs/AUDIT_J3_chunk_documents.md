# AUDIT JOUR 3 — `chunk_documents.py`

> Audit complet du script de chunking du RAG Citation Validator.
> Périmètre : JOUR 3 uniquement. Aucun élément J4+ créé.
> Méthode : revue de code + mesures objectives sur `corpus/chunks.json`
> (outil d'audit réutilisable : `files/audit_chunks.py`) + validation par exécution.

---

## 0. Contraintes respectées

- ❌ Pas de JOUR 4+ créé (pas de Qdrant, BM25, embeddings, retrieval, FastAPI).
- ✅ `chunk_documents.py` **non recréé** : patchs chirurgicaux sur le script existant.
- ✅ Travail à partir des artefacts existants (`documents.json`, `chunks.json`).

---

## 1. Revue de code détaillée

| Élément | Verdict |
|---|---|
| `get_logger`, `locate_corpus`, `load_documents` | ✅ Sains. Ancrage multi-candidats robuste (script dir / CWD). |
| `estimate_tokens` | ⚠️ `len(text)/4` : calibration moyenne OK (ratio réel mesuré 2.71–4.21, mean 4.02) mais **sous-estime** le texte math/code dense (ratio 2.71 ⇒ ~740 tokens réels pour un `tokens_est`=500). Voir reco §6.2. |
| `to_units` | ⚠️ Regex `(?<=[.;!?])\s+` : coupe après « Fig. », « et al. », « i.e. » → unités fragmentées (« Fig. », « 2 »). Pas de dépendance lourde ajoutée (voir reco §6.1). |
| `split_long_unit` | ✅ Coupe sur espace (jamais en plein mot), pièces rattachées à la bonne page. |
| `chunk_document` | 🔴 **Bug majeur corrigé** : pas de borne haute → chunks jusqu'à 996 tokens (≈2× la cible). Détail §3.1. |
| `flush()` | 🔴 Overlap coupé au caractère → mots tronqués en tête de chunk. Corrigé §3.2. |
| Résidu final | 🟠 Réémettait la queue d'overlap seule → chunks doublons. Corrigé §3.3. |
| `write_chunks` | ✅ Payload propre ; config désormais complète (hard_cap, bge_max_tokens). |
| CLI `parse_args`/`main` | ✅ Validation des bornes ; isolation par document ; stats enrichies. |

Points forts conservés : uuid5 déterministe (reproductibilité), attribution de page
par unité, gestion d'erreur par document, logs clairs, artefact canonique unique.

---

## 2. Vérification logique du chunking

Algorithme (fenêtre glissante sur phrases) : logiquement **correct** après patch.

1. Pages → phrases (`to_units`) → pièces si phrase > target (`split_long_unit`).
2. Accumulation dans `buffer` jusqu'à `target_tokens` → `flush()`.
3. **NOUVEAU** — early-flush avant dépassement de `hard_cap` : empêche la
   combinaison de 2 grosses unités (~2×500) en un seul chunk.
4. **NOUVEAU** — `units_since_flush` : le résidu final n'est réémis que s'il
   contient du contenu réel (pas seulement la queue d'overlap).
5. Overlap : queue du chunk émis, alignée **frontière de mot** (forward-trim),
   donc bornée ≤ `overlap_tokens`.

Cas limites vérifiés :
- unité == target exactement → non splittée, émise seule ✅
- document vide / sans texte → 0 chunk, pas d'erreur ✅
- `page_start ≤ page_end` sur les 2026 chunks ✅
- IDs uniques : 2026/2026, aucun doublon ✅
- Déterminisme : 2 exécutions → même empreinte de chunk_ids
  (`2026:5023ce362650bd6d`) ⇒ upserts Qdrant idempotents ✅

---

## 3. Bugs détectés & correctifs appliqués

### 3.1 🔴 CRITIQUE (J4/J6) — Aucune borne haute de taille
Deux unités de ~499 tokens (< cible, non splittées) s'accumulaient → flush à ~998.
**Mesuré avant : max = 996** ; 11 chunks > 768 ; 904 chunks > 512 (57 %).
**Correctif** : paramètre `hard_cap_tokens` (CLI `--hard-cap`, défaut
`target + overlap`) + early-flux avant dépassement.
**Après : max = 512, 0 chunk > cap, 0 chunk > 512 BGE.**

### 3.2 🔴 HAUT (J5/J6) — Overlap coupé au caractère
`text[-400:].lstrip(...)` démarrait l'overlap **en plein mot**
(« …relati|on… » → chunk suivant commençant par « on ») : dégrade BM25
(termes partiels non matchés) et la cohérence du reranker.
**Correctif** : `extract_overlap_tail()` aligne la coupe sur une frontière de mot.
**Après : 98 % des overlaps démarrent sur une frontière de mot** (41 paires
résiduelles = mots longs type URL, récupérables dans le chunk précédent).

### 3.3 🟠 MOYEN — Chunks « doublons queue »
Le résidu final pouvait être la queue d'overlap seule (~100 tokens déjà présents
dans le chunk précédent) et était réémis tel quel.
**Mesuré avant : 3 chunks purement redondants. Après : 0.**

### 3.4 🟠 MOYEN — Config incohérente avec l'objectif BGE-512
Le docstring annonçait « pas de troncature des vecteurs denses » avec BGE ~512,
mais `target=500 + overlap=100` produit des chunks 600→996 ⇒ troncature massive.
**Correctif** : défaut `DEFAULT_TARGET_TOKENS = 412` (412 + 100 = 512 exactement),
warning automatique si `hard_cap > 512`, docstring corrigée.
L'ancien comportement reste accessible via `--target-tokens 500`.

### 3.5 🟢 MINEUR — Stats de monitoring insuffisantes
Ajouté : `p90_tokens`, `hard_cap_tokens`, `chunks_over_cap`, `chunks_over_bge`,
`per_theme[*].tokens_est`, lignes console dédiées, `config.bge_max_tokens`.

---

## 4. Qualité des chunks produits (mesures avant → après)

| Indicateur | AVANT (500/100) | APRÈS (412/100, cap 512) |
|---|---|---|
| Chunks | 1 577 | 2 026 |
| Tokens totaux (est.) | 821 452 | 862 511 (+ overlap propre) |
| min / médiane / p90 / max | 100 / 515 / 553 / **996** | 100 / 427 / 458 / **512** |
| chunks > 512 (BGE dense) | 904 (57 %) | **0** |
| chunks > 768 | 11 | **0** |
| Overlap réel moyen (~tokens) | 100 ✅ mais coupé au caractère | 98 ✅ frontière de mot |
| Overlap en plein mot | ≈100 % des paires | **2 %** (41 paires) |
| Doublons « queue » | 3 | **0** |
| Multi-pages | 725 (46 %), span max 20 p. | 739 (**36,5 %**), span max 15 p. |
| Commence par une minuscule | 74,9 % | 62,3 % (milieu de phrase légitime) |
| IDs uniques / déterministes | oui | oui (vérifié 2 runs) |

Lecture :
- La distribution est désormais **serrée et bornée** : p25=417, médiane=427,
  p75=440, p95=471, max=512. Plus aucun outlier à 996.
- Les 18 chunks < 200 tokens sont des fins de document/tableaux (≥ `MIN_CHUNK_TOKENS`).
- Le chunk « 15 pages » est une section sparse (références/annexes) : peu de
  texte par page, la fenêtre met du temps à atteindre la cible. Acceptable.

---

## 5. Compatibilité future (validation)

### Qdrant — JOUR 4 (dense)
✅ Schéma suffisant : `chunk_id` (uuid5 stable ⇒ upsert idempotent vérifié),
`document_id`, `theme`, `page_start/page_end`, `text`, `tokens_est`.
✅ Filtrage payload possible sur `theme` / `document_id`.
✅ **≤512 tokens partout** : embedding BGE sans troncature.
⚠️ Pas de champ `title`/`section` (l'ancien `extract_chunks.py` en avait un).
Si besoin de filtrer par section en J5/J7 : enrichir côté J4 depuis
`documents.json` plutôt que de complexifier le chunking maintenant.

### BM25 — JOUR 5
✅ `text` lexicalment propre (mots entiers, pas de fragments d'overlap).
✅ Densité lexicale suffisante à ~427 tokens ; BM25 insensible à la longueur,
donc le passage à ≤512 ne pose aucun problème.
⚠️ Quelques chunks très « chiffres » (tableaux PDF) apporteront du bruit lexical
→ voir reco §6.3 si le recall BM25 en souffre.

### BGE Reranker — JOUR 6
✅ Cross-encoder ~512 tokens : **0 troncature**, le rerank voit le chunk entier.
✅ Frontières de phrases/mots propres → passages cohérents pour le scoring.

### Citation Verification (RoBERTa-MNLI) — JOUR 7
✅ `page_start/page_end` exacts par unité (provenance préservée).
✅ URLs/filenames longs restent dans le **corps** du chunk (le forward-trim ne
touche que la queue d'overlap, jamais le contenu principal).
✅ 36,5 % de chunks multi-pages (essentiellement 2 pages) : acceptable ; la page
précise reste retrouvable via l'overlap avec le chunk mono-page adjacent.
ℹ️ 4 paires d'overlap quasi nul (<10 tokens) : continuité réduite localement,
sans perte d'information (contenu présent dans le chunk précédent).

---

## 6. Propositions d'amélioration — niveau AI Engineer 2026

*(non implémentées ici : hors périmètre JOUR 3 / contraintes)*

1. **Sentence boundary detection robuste** — la regex `(?<=[.;!?])\s+` casse sur
   « Fig. », « et al. », « i.e. » et ne gère pas les équations. Options : regex
   enrichie (liste d'abréviations arXiv) ou tokenizer dédié. À évaluer au J4+
   avec des métriques de retrieval ; pas de dépendance ajoutée maintenant
   (`nltk`/`spacy` absents de `requirements.txt`).
2. **Budget tokens exact plutôt qu'estimé** — `len(text)/4` sous-estime le texte
   math/code dense (ratio réel mesuré jusqu'à 2.71). Au J4, compter les tokens
   avec le vrai tokenizer BGE (`AutoTokenizer`) pour garantir ≤512 en *vrais*
   tokens, et remonter l'info dans `tokens_est`.
3. **Filtre qualité des chunks « bruit de tableau »** — 18 chunks <200 tokens et
   quelques chunks quasi numériques (tableaux PDF) apportent peu de signal
   sémantique. Option : score de densité alphanumérique + flag payload
   (`is_table_like`) pour déprioriser en J5/J6 sans les supprimer.
4. **Chunks multi-pages & J7** — si la vérification de citations demande une page
   unique, ajouter une option « couper aux frontières de page quand le span >2 »,
   au prix de chunks plus courts. À décider avec les retours du J7.
5. **Rapport de build** — `chunk_documents.py` n'écrit que `chunks.json`
   (les stats sont dedans). Produire aussi un `chunks_report.md` lisible comme
   le faisait l'ancien script, pour le suivi J4→J7.
6. **Tests unitaires** — `pytest` est déjà dans requirements : couvrir
   `extract_overlap_tail` (mots longs/URLs), le hard cap, le résidu final et le
   déterminisme des uuid5 avant d'industrialiser.

---

## 7. ⚠️ Risques annexes détectés (hors patch, à arbitrer)

1. **`files/extract_chunks.py` peut écraser l'artefact canonique.**
   Il écrit lui aussi `corpus/chunks.json` (ancien format, borné à UNE page,
   cible 512/cap 768/overlap 40) et son dernier run était un test `--limit-docs 3`
   (79 chunks / 3 docs). Une relance accidentelle remplacerait le `chunks.json`
   validé. Recommandation : le renommer (`_deprecated_extract_chunks.py`),
   l'archiver ou lui faire écrire un autre nom de fichier.
2. **`corpus/chunks_report.json` et `.md` sont obsolètes** : ils décrivent encore
   ce vieux run (generator = `extract_chunks.py`, 79 chunks / 3 docs). À régénérer
   ou supprimer pour éviter toute confusion lors du J4.
3. `__pycache__/` versionné dans le dépôt — ajouter un `.gitignore`.

---

## 8. Fichiers modifiés / ajoutés

| Fichier | Statut |
|---|---|
| `files/chunk_documents.py` | **Patché** (hard cap strict, overlap frontière-mot, anti-doublon résidu, `BGE_MAX_TOKENS`, CLI `--hard-cap`, stats riches, docstrings corrigées). Défaut : 412/100 → cap 512. |
| `files/corpus/chunks.json` | **Régénéré** (artefact canonique J4/J5) : 2 026 chunks, max 512, 0 > cap. |
| `files/audit_chunks.py` | **Ajouté** — outil d'audit réutilisable (distribution, overlap réel, frontières de mot, doublons, multi-pages, UUID). |
| `AUDIT_J3_chunk_documents.md` | **Ajouté** — ce rapport. |

Validation finale exécutée :
`python chunk_documents.py` → 40 docs, 0 erreur, 2 026 chunks, min 100 /
médiane 427 / p90 458 / **max 512**, 0 chunk > cap, 0 chunk > BGE 512,
739 multi-pages, IDs 2026/2026 uniques, déterminisme confirmé.



