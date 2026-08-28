# RAG Citation Validator

## Rapport de Projet de Fin d'Études (PFE) — Master Ingénierie IA

**Niveau :** Master / Ingénieur IA — Promotion 2026
**Auteur :** [Étudiant.e], École d'ingénieur / Université
**Encadrant :** [Nom de l'encadrant]
**Date :** Août 2026
**Technologies :** Python 3.11 · Streamlit · Qdrant · BAAI/bge-small-en-v1.5 · BAAI/bge-reranker-base · roberta-large-mnli · Ollama (qwen2.5:3b)

---

## Table des matières

1. [Résumé](#1-résumé)
2. [Introduction](#2-introduction)
3. [Contexte du RAG](#3-contexte-du-rag)
4. [Problématique](#4-problématique)
5. [État de l'art](#5-état-de-lart)
6. [Architecture globale](#6-architecture-globale)
7. [Ingestion documentaire](#7-ingestion-documentaire)
8. [Chunking](#8-chunking)
9. [Dense Retrieval](#9-dense-retrieval)
10. [BM25](#10-bm25)
11. [Hybrid Search](#11-hybrid-search)
12. [RRF](#12-rrf)
13. [Reranking BGE](#13-reranking-bge)
14. [Génération de réponse](#14-génération-de-réponse)
15. [Ollama et qwen2.5:3b](#15-ollama-et-qwen253b)
16. [Citation Verification](#16-citation-verification)
17. [Métriques utilisées](#17-métriques-utilisées)
18. [Framework d'évaluation](#18-framework-dévaluation)
19. [Dashboard](#19-dashboard)
20. [Interface Streamlit](#20-interface-streamlit)
21. [Résultats expérimentaux](#21-résultats-expérimentaux)
22. [Analyse détaillée des métriques](#22-analyse-détaillée-des-métriques)
23. [Difficultés rencontrées](#23-difficultés-rencontrées)
24. [Bugs majeurs identifiés](#24-bugs-majeurs-identifiés)
25. [Corrections apportées](#25-corrections-apportées)
26. [Analyse des hallucinations](#26-analyse-des-hallucinations)
27. [Limites du système](#27-limites-du-système)
28. [Perspectives d'amélioration](#28-perspectives-damélioration)
29. [Conclusion](#29-conclusion)
30. [Compétences acquises](#30-compétences-acquises)
31. [Questions potentielles d'entretien et réponses attendues](#31-questions-potentielles-dentretien-et-réponses-attendues)

---

## 1. Résumé

**RAG Citation Validator** est un système de *Retrieval-Augmented Generation* (RAG) complet, conçu pour répondre à une question en langage naturel en s'appuyant exclusivement sur un corpus de documents scientifiques (articles arXiv), en **citant systématiquement les sources** utilisées et en **vérifiant automatiquement chaque citation** par inférence NLI (*Natural Language Inference*).

Le pipeline retenu combine une recherche hybride (plongements denses **BGE-small-en-v1.5** + recherche lexicale **BM25**, fusionnées par **RRF**), un réordonnancement par cross-encodeur (**BGE-reranker-base**), une génération de réponse ancrée par un LLM local (**qwen2.5:3b** via **Ollama**), et une **vérification des citations** par **roberta-large-mnli**. L'ensemble est exposé au travers d'une interface **Streamlit** et évalué sur des métriques standard de la littérature.

Les résultats mesurés sur le corpus de validation sont les suivants :

| Brique | Métrique | Résultat |
|---|---|---|
| Chunking | Chunks produits | 2 026 |
| Chunking | Médiane de taille | 427 tokens |
| Chunking | Chunks > 512 tokens | 0 (0 %) |
| Retrieval | Recall@5 | **75,0 %** |
| Retrieval | MRR | **72,8 %** |
| Génération + Vérification | Faithfulness | **72,7 %** |
| Génération + Vérification | Citation Accuracy | **80,0 %** |

Au-delà des performances, le projet documente un travail d'**ingénierie de la fiabilité** : les nombreux défauts rencontrés sur la chaîne (parsing JSON fragile du LLM, hallucinations inter-domaines, sur-citation, biais *neutral* du modèle NLI sur les paraphrases, rejets injustifiés) ont été identifiés, instrumentés par des audits ciblés, puis corrigés par des patches minimaux dont l'impact a été mesuré sur des exécutions réelles. Le rapport présente l'architecture, les choix techniques et leurs justifications, les métriques, l'analyse critique des limites, ainsi que les perspectives d'amélioration.

---

## 2. Introduction

La recherche d'information et la génération de texte ont longtemps été traitées comme deux problèmes distincts : d'un côté les moteurs de recherche (BM25, plongements vectoriels), de l'autre les modèles génératifs (GPT, LLaMA, etc.). L'émergence des LLM a révélé une limite majeure de la génération *isolée* : l'**hallucination** — la production de contenus fluides mais faux — et l'incapacité à se référer à des sources externes vérifiables.

Le paradigme **RAG** (Retrieval-Augmented Generation) répond à ces deux problèmes en **connectant une base de connaissances externe au processus de génération**. Introduit par Lewis et al. (2020), il consiste à (1) *récupérer* les passages pertinents pour la question, puis (2) *générer* la réponse conditionnellement à ces passages. Cette approche améliore la factualité, réduit l'hallucination et permet une mise à jour des connaissances sans réentraîner le modèle.

Cependant, le RAG n'élimine pas l'hallucination par lui-même : un LLM peut ignorer le contexte fourni, synthétiser des faits non présents dans les sources, ou citer des passages qui ne soutiennent pas réellement ses affirmations. C'est précisément le problème traité par ce projet : **ajouter un étage de vérification automatique des citations** et garantir que *chaque affirmation de la réponse est supportée par les sources citées*.

Le présent rapport s'articule en trois temps :

1. **Conception** : état de l'art, architecture, ingestion, chunking, retrieval hybride, reranking, génération ancrée, vérification NLI ;
2. **Réalisation** : modules implémentés, pseudo-code, justification des choix techniques, interface ;
3. **Évaluation critique** : métriques, résultats, difficultés, bugs, correctifs, limites et perspectives.
---

## 3. Contexte du RAG

### 3.1. Du moteur de recherche au générateur augmenté

**Recherche d'information (RI).** La RI classique repose sur des modèles lexicaux : **BM25** (Robertson et Zaragoza, 2009) pondère les termes par leur fréquence normalisée (TF) et leur rareté dans le corpus (IDF). Son avantage : robustesse, zéro entraînement, excellente précision sur les termes exacts. Sa limite : *inexactitude sémantique* — la synonymie et la paraphrase ne sont pas capturées.

**Recherche dense.** Les modèles à plongements projettent textes et requêtes dans un espace vectoriel où la similarité cosinus mesure la proximité sémantique. Des modèles comme **BGE** (BAAI), *E5*, *GTE* fournissent des représentations contextualisées apprises sur de larges corpus. La limite inverse : sensibilité aux termes rares, coût d'indexation, moindre précision sur les identifiants exacts.

**RAG.** Le RAG combine ces deux familles : le contexte récupéré est injecté dans le prompt du LLM, qui répond **uniquement** à partir de ces passages. Formellement, pour une question \(q\), la génération modélise :

\[
P(r \mid q) = \sum_{D} P_{\text{retr}}(D \mid q) \cdot P_{\text{gen}}(r \mid q, D)
\]

Le projet suit le schéma *retrieve puis read* : recherche déterministe, puis génération conditionnée au contexte.

### 3.2. Les deux enjeux du RAG

1. **Pertinence du contexte** : si les passages récupérés ne contiennent pas l'information, aucune génération ne pourra répondre correctement. D'où l'architecture hybride + reranking qui maximise couverture et précision.
2. **Fidélité au contexte** : le LLM doit répondre *à partir des sources*, pas de sa mémoire. C'est le problème de la *grounding* : formulation du prompt, filtrage lexical d'ancrage, **gate NLI** de rejet, et **vérification NLI** des citations en aval.

### 3.3. Un système mesurable, pas un prototype

Chaque brique expose des métriques (Recall@k, MRR, Faithfulness, Citation Accuracy, latences), chaque réponse est accompagnée de ses citations vérifiées avec leur score d'implication, et le corpus de validation permet de reproduire les résultats.

---

## 4. Problématique

### 4.1. Énoncé

> Comment construire un système RAG **fiable** — qui répond en citant des sources réelles, ne produit pas d'affirmation non supportée, et dont la qualité est **mesurable objectivement** — sans dépendre d'un LLM propriétaire ?

### 4.2. Sous-problèmes

1. **Représentation du corpus** : découper des documents longs en unités cohérentes et vectorisables sans perte d'information aux frontières ;
2. **Récupération** : combiner robustement recherche lexicale et sémantique, puis réordonner par pertinence fine ;
3. **Génération ancrée** : forcer le LLM à répondre *uniquement* à partir du contexte, avec des citations numériques exploitables, et à refuser poliment si l'information est absente ;
4. **Vérification** : vérifier automatiquement, par un modèle d'implication, que chaque phrase répondante est supportée par la source citée ;
5. **Mesure** : quantifier chaque étage (Recall, MRR, Faithfulness, Citation Accuracy) de façon reproductible.

### 4.3. Hypothèse de travail

Le projet postule qu'un **contrôle en aval (NLI)** de la fidélité des citations, couplé à un **filtrage lexical en amont** de l'ancrage, suffit à garantir un niveau de fiabilité élevé — à condition d'instrumenter finement les échecs (parsing, grounding, NLI, fallback) pour distinguer *hallucination réelle* de *biais du vérifieur*.

---

## 5. État de l'art

### 5.1. Modèles d'embeddings denses

**bge-small-en-v1.5** (24 M de paramètres, dimension 384, contexte 512 tokens) montre qu'une taille modeste suffit pour des scores de retrieval proches des modèles géants, tout en restant exécutable sur CPU. Choix justifié par : (i) le compromis qualité/vitesse, (ii) la compatibilité dimension 384 / Qdrant sans réglage, (iii) la stabilité des classements sur les benchmarks BEIR.

### 5.2. Recherche hybride et fusion

| Méthode | Principe | Forces | Faiblesses |
|---|---|---|---|
| Score pondéré | Combinaison linéaire de scores normalisés | Simple | Calibration des poids fragile |
| RRF | \(\sum_{d} \frac{1}{k + \text{rank}(d)}\) | Aucune calibration, robuste aux échelles | Récompense la présence plus que la précision |
| Fusion apprise | Modèle entraîné sur les deux canaux | Optimale | Nécessite des données annotées |

Le projet retient la **RRF** (Cormack et al., 2009) avec \(k = 60\), pour sa robustesse et son absence de calibration sur les données.

### 5.3. Reranking par cross-encodeur

Les *bi-encodeurs* (embedding models) calculent la similarité une fois par document — efficaces mais approximatifs. Les *cross-encodeurs* (ex. **bge-reranker-base**) apparient requête et document dans un même Transformer, permettant des interactions mot-à-mot fines. Utilisés sur la courte liste des candidats hybrides, ils améliorent nettement la précision finale.

### 5.4. Défense anti-hallucination

- **Prompt engineering** : consignes strictes d'utilisation exclusive du contexte, refus explicite si information manquante ;
- **Extraction de claims et ancrage** : découper la réponse en phrases vérifiables rattachées à leurs sources ;
- **Vérification NLI automatique** (RAGAS, TrueTeacher) : vérifier que la source *implique* le claim, via roberta-large-mnli ;
- **Gates de rejet** : ne pas afficher une réponse dont les claims ne passent pas les contrôles.

### 5.5. Cadres d'évaluation

**RAGAS** (Es et al., 2023) définit *Faithfulness*, *Answer Relevancy* et *Context Relevancy*. Le projet s'aligne sur cette grille et ajoute des métriques retrieval (Recall@k, MRR) et de citation (Citation Accuracy).
---

## 6. Architecture globale

### 6.1. Vue d'ensemble

```
 Question utilisateur
        │
        ▼
 ┌───────────────────────── HYBRID SEARCH ─────────────────────────┐
 │  Dense Retrieval (BGE-small-en-v1.5, Qdrant, cosinus)            │
 │                    +                                             │
 │  BM25 (index lexical, TF-IDF pondéré)                            │
 │                    │                                             │
 │  Fusion RRF  (Reciprocal Rank Fusion, k=60)                      │
 └────────────────────────────┬────────────────────────────────────┘
                              │ top-20 candidats
                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  Reranker BGE cross-encodeur (BAAI/bge-reranker-base)            │
 │  re-score des 20 candidats → top-5 ordonné par pertinence fine   │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ 5 chunks + métadonnées
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  Génération LLM (qwen2.5:3b via Ollama)                          │
 │   prompt = contextes numérotés + SYSTEM_PROMPT (règles strictes) │
 │   → claims JSON {text, citations}                                │
 │   parsing tolérant + sanitation + grounding lexical + gate NLI   │
 │   fallback TemplateProvider si Ollama indisponible               │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ réponse finale + claims ancrés
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  Citation Verification (roberta-large-mnli)                      │
 │   segmenter la réponse en phrases → fenêtre locale → entailment  │
 │   → support_score → verdict (Supported / Weak / Unsupported)     │
 └────────────────────────────────┬─────────────────────────────────┘
                                  ▼
                    Réponse finale citée + table de vérification
```

### 6.2. Justifications des choix

| Étage | Choix | Justification |
|---|---|---|
| Base vectorielle | **Qdrant** | Moteur local léger, index HNSW, aucun serveur externe requis, compatible Windows |
| Embeddings | **bge-small-en-v1.5** (dim. 384) | Compromis qualité/latence, contexte 512 tokens, aucun GPU requis |
| Fusion | **RRF** (k=60) | Aucune calibration, fusion robuste de classements d'échelles hétérogènes |
| Reranker | **bge-reranker-base** | Cross-encodeur : interactions mot-à-mot, gain mesuré de précision |
| LLM | **qwen2.5:3b** (Ollama local) | Inférence locale, souveraine, sans clé API ; taille adaptée à une machine de dev |
| Vérification | **roberta-large-mnli** | Référence académique du NLI, 3 classes (entailment/neutral/contradiction) |

### 6.3. Traçabilité de bout en bout

Chaque requête génère un rapport structuré : latences par étage, top-20 hybrides, top-5 rerankés, prompt envoyé, réponse brute du LLM, claims extraits et ancrés, scores NLI par citation, verdicts. Cette *traçabilité* a été déterminante pour les audits de bugs (sections 23-25).

---

## 7. Ingestion documentaire

### 7.1. Chaîne d'extraction

```
 documents PDF / texte
        │
        ▼
 extract_text.py ──► textes bruts + métadonnées (titre, auteurs, section)
        │
        ▼
 chunk_documents.py ──► chunks {id, texte, document_id, page, section}
        │
        ├──► build_dense_index.py ──► Qdrant (vecteurs BGE 384d)
        └──► build_bm25_index.py  ──► index BM25 (memoire / disque)
```

`extract_text.py` assure l'extraction du texte (PDF/textuel), le nettoyage des artefacts typographiques et la conservation des métadonnées de pagination indispensables aux citations.

### 7.2. Validation du corpus

- `validate_chunks.py` : unicité des UUID, non-vacuité, respect de la borne hard-cap, absence de doublons textuels ;
- `validate_corpus.py` : cohérence entre manifeste, fichiers sources et pagination.

Résultat d'audit : **2 026 chunks**, **0 doublon**, **0 chunk > 512 tokens**, UUID uniques, **98 % de frontières propres**.
---

## 8. Chunking

### 8.1. Paramètres retenus

| Paramètre | Valeur | Justification |
|---|---|---|
| `target_tokens` | **412** | Granularité «sous-section scientifique» ; 412 + 100 = 512 = borne BGE (aucune troncature) |
| `overlap_tokens` | **100** (≈ 24 %) | Une information sur une frontière reste dans le chunk adjacent (dense ET BM25) |
| `hard_cap_tokens` | **512** | Borne stricte == contexte max des embeddings BGE |
| Taille minimale | ≈ 64 tokens | En dessous : bruit de page non exploitable |

### 8.2. Le problème calibré : 412 + 100 = 512

L'ancien défaut `target=500, overlap=100` produisait des chunks de 600 à 996 tokens **tronqués par BGE** (contexte 512), dégradant silencieusement les vecteurs. Correction : `target=412` → **412 (neuf) + 100 (recouvrement) = 512 exactement** → aucun embedding sur texte tronqué.

### 8.3. Résultats d'audit du chunking

| Indicateur | Valeur mesurée |
|---|---|
| Nombre de chunks | 2 026 |
| Médiane de taille | 427 tokens |
| Taille maximale | 512 tokens |
| Chunks > 512 | 0 (0 %) |
| Doublons | 0 |
| Overlap moyen réel | ≈ 98 tokens |
| Frontières propres | 98 % |

### 8.4. Pseudo-code

```
function decouper(document, target=412, overlap=100, cap=512):
    phrases = segmenter(document)          # découpage phrase par phrase
    chunks = []
    buffer  = []
    buffer_tokens = 0
    for phrase in phrases:
        t = tokens(phrase)
        if buffer_tokens + t > cap:        # depasse la borne stricte
            chunks.append(fusionner(buffer))
            buffer = dernieres_phrases(buffer, pour_tokens=overlap)
            buffer_tokens = tokens(buffer)
        buffer.append(phrase)
        buffer_tokens += t
    si buffer_tokens >= MIN_TOKENS_TO_KEEP:
        chunks.append(fusionner(buffer))
    return [Chunk(id=uuid4(), texte, page, section) pour chaque chunk]
```
---

## 9. Dense Retrieval

### 9.1. Principe

La recherche dense encode la question et chaque chunk en vecteurs de dimension fixe (384 pour BGE-small-en-v1.5) ; la pertinence est la **similarité cosinus** :

\[
\text{sim}(q, c) = \frac{\mathbf{e}_q \cdot \mathbf{e}_c}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_c\|}
\]

### 9.2. Indexation (build_dense_index.py)

```
pour chaque chunk du corpus:
    tokens = tokeniser(chunk.texte)[:512]      # borne modele
    vecteur = bge_small_en_v1_5.encoder(tokens) # 384d, normalise
    qdrant.upsert(point_id=chunk.uuid,
                  vector=vecteur,
                  payload={document_id, page_start, page_end, theme})
```

Stockage **Qdrant** avec index HNSW, qui offre un compromis latence/recall réglable via les paramètres `ef_construct`/`m`.

### 9.3. Justification du modèle

- Taille 24 M → exécution CPU compatible, chargement rapide ;
- Performance retrieval supérieure à des modèles plus anciens et plus lourds (SBERT) sur BEIR ;
- Normalisation L2 systématique → les scores sont comparables entre requêtes.

---

## 10. BM25

### 10.1. Formule

BM25 (Best Matching 25, Robertson & Zaragoza 2009) attribue à chaque terme \(t\) du document \(d\) un poids :

\[
\text{BM25}(d,q) = \sum_{t \in q} \text{IDF}(t) \cdot
\frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
\]

avec \(k_1 = 1.2\), \(b = 0.75\) (valeurs canoniques). L'IDF empêche les termes trop fréquents de dominer ; la normalisation par longueur moyenne (`avgdl`) pénalise les documents longs.

### 10.2. Rôle dans le pipeline

BM25 excelle sur les **termes exacts** : identifiants, formules, noms de méthodes, notations de variables — typiquement absents de la sémantique vectorielle. Il sert donc de canal *lexical* complémentaire.

### 10.3. Index BM25 (build_bm25_index.py)

- Tokenisation (minuscules, suppression de la ponctuation) ;
- Calcul des statistiques documentaires (`df`, `avgdl`) sur le corpus complet ;
- Index inversé terme → postings ;
- Requête : scoring des documents contenant au moins un terme de la requête.

---

## 11. Hybrid Search

### 11.1. Pourquoi hybride ?

Mesures sur le corpus (Recall@5) montrent des profils complémentaires :

| Canal | Force | Faiblesse |
|---|---|---|
| Dense seul | Paraphrases, synonymes, ordre des mots | Termes rares, identifiants, chiffres précis |
| BM25 seul | Termes exacts, identifiants | Paraphrases, polysémie |
| **Hybride** | **Les deux** | (coût léger) |

### 11.2. Implémentation (hybrid_search.py)

```
def search(query, top_k=20):
    dense_results  = qdrant.search(embed(requete), limit=fetch_k)
    bm25_results   = bm25_index.search(query, limit=fetch_k)
    fused          = rrf_fusion(dense_results, bm25_results, k=60)
    return RRF(fused[:top_k])
```

Les deux canaux récupèrent un pool large (`fetch_k`), la fusion intervient **sur les rangs** (pas sur les scores bruts, non comparables).

### 11.3. Pseudo-code RRF

```
function rrf_fusion(listes_rangs, k=60):
    scores = {}
    for liste in listes_rangs:
        for rang, doc in enumerate(liste, start=1):
            scores[doc] += 1.0 / (k + rang)
    return trier(scores, decroissant)
```

### 11.4. Paramètres

- `fetch_k` (pool) : 20 par défaut (slider Streamlit 10-50) ;
- `k` RRF : 60 (valeur canonique de la littérature) ;
- `top_k` final : 5 (== MAX_SOURCES).

---

## 12. RRF

### 12.1. Propriétés mathématiques

La contribution d'un document classé rang \(r\) est \(\frac{1}{k+r}\) (avec \(k=60\)). Exemple de comparaison :

- Document A présent au **rang 1 d'un seul canal** : \(\frac{1}{61} \approx 0.0164\) ;
- Document B présent au **rang 10 des deux canaux** : \(2 \times \frac{1}{70} \approx 0.0286\) > 0.0164.

La fusion récompense donc le **consensus entre canaux** plus que la première position isolée, sans calibration entre échelles de scores (cosinus ∈ [-1,1] vs BM25 ∈ ℝ⁺, non comparables).

### 12.2. Pourquoi pas une fusion apprise ?

Une fusion supervisée nécessiterait des annotations de pertinence inverses ; la RRF offre une performance proche pour zéro coût d'entraînement et zéro risque de sur-apprentissage sur le petit jeu de validation.
---

## 13. Reranking BGE

### 13.1. Principe du cross-encodeur

Contrairement aux bi-encodeurs qui encodent requête et document séparément, le cross-encodeur **bge-reranker-base** concatène la paire `[CLS] requête [SEP] document [SEP]` dans un unique Transformer et prédit un score de pertinence par couche d'interactions mot-à-mot. Le coût (une inférence par paire) est acceptable car il ne porte que sur le **pool hybride** (top-20), pas sur le corpus entier.

### 13.2. Pipeline de reranking

```
pool = results_hybrides[:pool_size]               # 20 candidats
textes = [chunk_index[result.chunk_id].texte pour result en pool]
scores = reranker.score_pairs(requete, textes)    # 20 inferences
classement = trier(pool, par -score, puis chunk_id)
return classement[:top_k]                          # top-5
```

### 13.3. Effet mesuré

Dans les traces d'exécution réelles, le reranker modifie fortement l'ordre initial : des chunks classés bas par la fusion hybride remontent en tête (ex. 2402.12317v2 p.8 resélectionné au rang 1), prouvant l'apport de la *compréhension* de la paire au-delà de la seule similarité de plongement. Le tie-break par `chunk_id` garantit un classement **déterministe**.

---

## 14. Génération de réponse

### 14.1. Objectif et contraintes

Le générateur doit produire une réponse qui :
1. **répond d'abord à la question** (claims[0] = réponse directe) ;
2. **ajoute les preuves issues des sources** (claims[1..2]) ;
3. **cite chaque fait** par les indices [N] des chunks ;
4. **refuse** explicitement si l'information manque.

### 14.2. Pipeline de génération (generate_answer.py)

```
generate_answer(query, top5_rerankes, chunk_index, provider):
    sources_map = chunks[0..MAX_SOURCES] avec texte entier

    # -- etape 1 : le LLM propose des claims ancres
    claims = provider.generate_claims(query, sources_map, temperature)

    # -- etape 2 : politique de rendu
    answer = decide_response(claims)          # 1..MAX_CLAIMS claims sinon REFUS

    # -- etape 3 : gate NLI anti-hallucination
    claims, answer = apply_nli_gate(query, claims, answer, sources_map)
    #    -> si toutes les citations < SUPPORT_LOW :
    #       0. tolerance paraphrase (claims ancres + non contredits) -> conserve
    #       1. sinon repli extractif determine (verbatim, filtre pertinence)
    #       2. sinon REFUS_RESPONSE explicite

    return AnswerResponse(answer, citations, sources, latence)
```

### 14.3. Politique de réponse

| Constante | Valeur | Rôle |
|---|---|---|
| `MIN_CLAIMS_FOR_ANSWER` | 1 | Plancher de claims ancrés pour accepter |
| `MAX_CLAIMS` | 5 | Plafond |
| `GROUNDING_COVERAGE` | 0.45 | Couverture lexicale minimale d'un claim de preuve par sa source |
| `ANSWER_GROUNDING_COVERAGE` | 0.15 | Seuil relâché pour le claim-réponse (synthèse) |
| `PARAPHRASE_COVERAGE_RESCUE` | 0.40 | Tolérance paraphrase du gate NLI |
| `REFUSAL_RESPONSE` | constante | Message de refus normalisé |

### 14.4. Grounding lexical

Un claim est **ancré** si au moins une de ses citations couvre ≥ seuil de ses tokens :

\[
\text{cov}(claim, source_i) = \frac{| \text{tokens}(claim) \cap \text{tokens}(source_i) |}{|\text{tokens}(claim)|}
\]

Seuils calibrés par audits : 0.45 pour les preuves, 0.15 pour la réponse-synthèse (reformulation naturelle plus diffuse).

---

## 15. Ollama et qwen2.5:3b

### 15.1. Choix d'un LLM local

| Critère | qwen2.5:3b |
|---|---|
| Taille | 3,1 Md de paramètres (≈ 2,1 Go quantisé Q4) |
| Exécution | Locale (Ollama), sans API ni coût |
| Qualité | Instruction-tuning correct, format JSON respecté en grande majorité |
| Latence typique | 6 s – 23 s par génération (CPU) |

Le Fallback **TemplateProvider** (extraction déterministe de phrases verbatim) garantit la disponibilité du système hors ligne, Ollama indisponible ou modèle absent.

### 15.2. Appel Ollama

```
POST /api/generate  {model: "qwen2.5:3b",
                     prompt: <contexte numerote + SYSTEM_PROMPT + question>,
                     stream: false,
                     options: {temperature: 0.2}}
→ reponse JSON {"response": "..."}
```

### 15.3. Prompt final

```
[1] <texte du chunk 1 (≤1200 caracteres)>
[2] <texte du chunk 2>
...
[5] <texte du chunk 5>

Question:
<question utilisateur>

Instructions:
<SYSTEM_PROMPT: regles strictes 1..7>
```

Le SYSTEM_PROMPT impose : réponse-question d'abord, preuves ensuite, citations 1-2 sources les plus directes, interdiction d'inventer, consistance citation-contenu, refus si information insuffisante, format JSON `{"claims":[{"text":"...","citations":[...]}]}`.

### 15.4. Parsing tolérant

Le parsing JSON (`parse_claim_json`) a été rendu robuste : `json.loads(strict=False)` (tolère les retours-chariots littéraux) puis, en cas d'échec, **fallback regex** qui extrait chaque paire `{"text","citations"}` et **neutralise les citations non entières** (ex. identifiants arXiv `2512.13930v1` produits par le modèle).
---

## 16. Citation Verification

### 16.1. Le problème

Une réponse peut être *fluide, citée* et pourtant **fausse** (le LLM invente un fait et l'attribue à une source). Il faut donc vérifier, pour chaque phrase citée, que **la source citée implique réellement l'affirmation**.

### 16.2. La tâche NLI

La *Natural Language Inference* détermine si une hypothèse \(h\) est **entailée** (impliquée), **contredite** ou **neutre** par rapport à une prémisse \(p\). roberta-large-mnli produit les trois probabilités \((P_c, P_n, P_e)\).

### 16.3. Score de support

\[
\text{support} = \min\left(1,\; P_e \cdot \frac{P_e}{P_e + P_c + \varepsilon}\right)
\]

Le premier facteur \(P_e\) récompense l'implication ; le ratio \(P_e/(P_e+P_c)\) pénalise la contradiction. Un "neutral" dominant (P_e faible) donne un support bas — c'est la faille structurelle identifiée, analysée en section 26.

### 16.4. Seuils de verdict

| Verdict | Condition |
|---|---|
| Supported | support ≥ SUPPORT_HIGH (0.70) |
| Weak Support | 0.40 ≤ support < 0.70 |
| Unsupported | support < 0.40 |

### 16.5. Fenêtre locale NLI

**Problème mesuré** : prémisse = chunk complet (400-500 tokens) → entailment dilué (roberta entraînée sur paires courtes). **Solution** : `extract_local_premise` découpe le chunk en phrases, sélectionne la phrase à plus fort overlap lexical avec le claim, et prend ±1 phrase autour :

```
fenetre = phrase_la_plus_proche(claim) ± 1 phrase
(3 phrases, fallback = chunk complet si aucun overlap)
```

Mesures d'audit : claim pur : entailment 0.34 (fenêtre) vs 0.48 (phrase seule) sur la même source — la fenêtre d'une seule phrase améliore encore, mais l'implémentation applique ±1 pour concilier contexte et finesse.

### 16.6. Segmentation des claims

`segment_claims` découpe la réponse sur `[.!?] + espace`, extrait les marqueurs [N] et **déduplique** les indices par phrase (correction du bug de doublons évoqué en section 24).

### 16.7. pseudo-code verify_citations

```
function verify_citations(query, answer, sources, chunk_index, verifier):
    segments = segment_claims(answer)        # phrases + [N]
    for seg in segments:
        for i in seg.citation_indices:
            chunk = chunk_index[sources[i-1].chunk_id]
            premise = extract_local_premise(chunk.texte, seg.claim_text)
            p = verifier.score_pairs([premise], [seg.claim_text])[0]
            support = compute_support_score(p)
            resultat = CitationVerificationResult(claim_text, doc, pages,
                                                  entailment, neutral,
                                                  contradiction, support,
                                                  verdict_for(support))
    return resultats, segments
```

---

## 17. Métriques utilisées

### 17.1. Métriques de retrieval

**Recall@k** : proportion des documents pertinents retrouvés parmi les k premiers :

\[
\text{Recall@k} = \frac{|\text{pertinents} \cap \text{top-}k|}{|\text{pertinents}|}
\]

Interprétation : couverture du contexte (l'information nécessaire au LLM est-elle dans le top-5 ?).

**MRR (Mean Reciprocal Rank)** :

\[
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
\]

où \(\text{rank}_q\) est le rang du premier document pertinent. Interprétation : capacité à placer la bonne source en tête.

### 17.2. Métriques de génération

**Faithfulness** (définition RAGAS) : proportion des *claims* de la réponse qui sont **supportés** par le contexte :

\[
\text{Faithfulness} = \frac{|\text{claims supportés}|}{|\text{claims totaux}|}
\]

**Citation Accuracy** : proportion des citations (claim, source) pour lesquelles la source citée **implique** réellement le claim (entailment ≥ seuil), mesurée par NLI.

### 17.3. Lien entre les métriques

- Si Recall@5 ∈ top-5 est faible → la réponse ne peut pas être fidèle (informations absentes).
- Si Recall est bon mais Faithfulness faible → problème de génération (hallucination).
- Si Faithfulness est bonne mais Citation Accuracy faible → mauvais *ancrage* des citations (faits bons attribués à la mauvaise source).

Ce diagnostic à trois niveaux a structuré les audits.

---

## 18. Framework d'évaluation

### 18.1. Jeu de validation

Un ensemble de questions avec leurs documents pertinents annotés sert de référence. Le pipeline complet (retrieval → reranking → génération → vérification) est exécuté en batch, avec reproductibilité (température, seeds, versions).

### 18.2. Module evaluate_pipeline.py

```
pour chaque question annotée q:
    hybrid = engine.search(q, top_k=pool)
    reranked = rerank(hybrid.results, top_k=MAX_SOURCES)
    answer = generate_answer(q, reranked, chunk_index, provider)
    verified = verify_citations(q, answer, sources, chunk_index)
    res = {
        recall@5, mrr        # vs annotations
        faithfulness,        # claims supportés / claims totaux
        citation_accuracy,   # (claim, source) entailment ≥ seuil
        latences par étage
    }
agreger -> rapport JSON + artefacts reproductibles
```

### 18.3. Autres modules d'audit

Des scripts dédiés (`audit_qfocus`, `audit_nli_chain`, `replay_*`, `parse_trace`) rejouent des réponses brutes capturées pour isoler chaque étage sans nouvelle inférence LLM : rejouabilité et déterminisme ont guidé la démarche.
---

## 19. Dashboard

`dashboard_evaluation.py` agrège les rapports d'évaluation en un tableau de bord :

- **Vues par métrique** : Recall@5, MRR, Faithfulness, Citation Accuracy, par question et en moyenne ;
- **Vue temporelle** : évolution des métriques au fil des correctifs (traçabilité des régressions) ;
- **Détail par question** : réponse, claims, citations vérifiées, verdicts, latences ;
- **Export** : tableaux CSV/JSON réutilisables pour le rapport.

Objectif : rendre les **régressions visibles** (ex. : un correctif de parsing a d'abord fait chuter le grounding 0.75 → 0.45 visuellement), fondamental en industrie où l'on déploie par itérations.

---

## 20. Interface Streamlit

### 20.1. Fonctionnalités

`app.py` expose le pipeline complet dans une interface :

- **Paramètres** : fournisseur LLM (template / ollama), modèle, température, pool de recherche (10-50), top-K reranker (3-5) ;
- **Question** : zone de saisie + bouton « Exécuter le pipeline » ;
- **Rendu** : réponse générée, **table des citations vérifiées** (claim, support score, verdict, document, pages, thème), latences par étape.

### 20.2. Reprise des modules backend

L'interface **ne duplique aucune logique métier** : elle importe `get_engine`, `rerank_results`, `generate_answer`, `verify_citations` depuis `files/` (sys.path dédié). Toute la qualité (gate, fallback, vérification) est donc active dans l'interface.

---

## 21. Résultats expérimentaux

### 21.1. Synthèse

| Étage | Métrique | Résultat |
|---|---|---|
| Chunking | Chunks | 2 026 |
| Chunking | Médiane / max | 427 / 512 tokens |
| Retrieval | **Recall@5** | **75,0 %** |
| Retrieval | **MRR** | **72,8 %** |
| Génération | **Faithfulness** | **72,7 %** |
| Vérification | **Citation Accuracy** | **80,0 %** |

### 21.2. Détail retrieval

Le canal hybride (dense + BM25 + RRF) atteint Recall@5 = 75 % : pour 3 questions sur 4, la source pertinente est dans le top-5 injecté au LLM. Le MRR de 72,8 % indique que la source pertinente est en moyenne très bien classée (rang 1 dans la majorité des cas), condition favorable à une génération fidèle.

### 21.3. Détail génération + vérification

Faithfulness 72,7 % : la majorité des affirmations sont appuyées par les sources. Citation Accuracy 80 % : lorsque l'on vérifie les paires (claim, source) par NLI, 80 % atteignent le seuil d'implication. L'écart de ~7 points entre les deux mesures s'explique par le **biais neutral** du vérifieur sur les paraphrases (section 26) et par les cas d'ancrage imparfait.

### 21.4. Latences observées (machine de développement, CPU)

| Étage | Latence |
|---|---|
| Hybrid search (pool 20) | 65-85 ms |
| Reranking (20 paires) | 4-12 s (rechargement possible du modèle) |
| Génération Ollama qwen2.5:3b | 6-23 s |
| Gate NLI (quelques paires) | 1-2 s |
| Vérification NLI | ~1 s |
| Template (hors ligne) | < 1 ms |

Ollama domine largement le budget temps : choix assumé d'une inference locale souveraine, documentée comme perspective (section 28).

---

## 22. Analyse détaillée des métriques

### 22.1. Recall@5 vs MRR : deux facettes du retrieval

Le couple (75 % ; 72,8 %) est cohérent : un Recall@5 élevé avec un MRR proche indique que la bonne source est **souvent en tête**. Un système à Recall@5 = 75 % mais MRR = 40 % aurait le même top-5 mais une qualité d'ordre dégradée — le reranker BGE y contribue directement.

### 22.2. Faithfulness < Citation Accuracy : que dit l'écart ?

Quand un claim n'est pas supporté, deux causes possibles :
1. **hallucination** : pas d'information dans le contexte (le NLI le relève en *neutral* ou *contradiction*) ;
2. **bonne information, mauvaise source citée** : le LLM attribue un fait correct à un chunk qui ne le contient pas.

L'écart 72,7 % vs 80 % montre que la *citation* est plus souvent correcte que l'*exhaustivité* des claims — les réponses sont plus « sûres dans ce qu'elles citent » qu'« complètes ». C'est une propriété souhaitable pour un système anti-hallucination (prudence > exhaustivité).

### 22.3. Sensibilité de Faithfulness au biais NLI

L'audit d'ablation (même fenêtre, 3 hypothèses) a mesuré :

| Hypothèse | entailment | neutral | contradiction | support |
|---|---|---|---|---|
| H1 verbatim (dans source) | 0.818 | 0.158 | 0.024 | **0.794** |
| H2 paraphrase correcte | 0.272 | **0.719** | 0.009 | **0.263** |
| H3 contrefactuel faux | 0.010 | 0.232 | **0.758** | **0.0001** |

Lecture critique : le NLI **détecte bien l'hallucination** (H3 → contradiction), mais **pénalise la paraphrase** (H2 → neutral → support 0.26 < 0.40). La Faithfulness mesurée à 72,7 % est donc une **borne basse conservatrice** : une partie des claims « non supportés » sont en réalité des paraphrases correctes que le vérifieur ne reconnaît pas. C'est l'un des résultats d'analyse les plus importants du projet.
---

## 23. Difficultés rencontrées

### 23.1. Découpage et troncature des embeddings

Le premier pipeline utilisait `target=500 + overlap=100` → chunks jusqu'à 996 tokens **tronqués silencieusement par BGE** (contexte 512). L'audit a révélé que la moitié du signal était perdue. Correction : calibration 412 + 100 = 512 et `hard_cap = target + overlap`.

### 23.2. Volatilité du LLM local

qwen2.5:3b, même à température 0.2, produit des réponses **non déterministes** : tantôt le bon format JSON, tantôt de la prose ; tantôt `[1]`, tantôt `[1,2,3,4,5]`. D'où le **contrôle déterministe en aval** (sanitation, grounding, gate) plutôt qu'une confiance dans le prompt seul.

### 23.3. Le parsing JSON comme goulot silencieux

Sur plusieurs questions, le LLM répondait correctement mais **produisait un JSON invalide** (identifiants arXiv dans les tableaux de citations, retours-charnière littéraux). Si le parsing échouait → 0 claims → **fallback Template = assemblage hors sujet**, masquant le vrai problème.

### 23.4. Confondre « rejeté par la pipeline » et « mal généré »

Le symptôme initial (« mode Ollama = identique au mode Template ») a mené une première enquête vers grounding/nli/fallback. La **preuve** a montré que la sortie **était bien reçue puis rejetée au parsing** : l'instrumentation de traces (RAG_TRACE) a été décisive (sections 24-25).

---

## 24. Bugs majeurs identifiés

| # | Bug | Symptôme | Cause racine |
|---|---|---|---|
| B1 | **Sortie Ollama identique au Template** | 23 s de génération jetées, réponse extractive | JSON invalide : citations non entières (arXiv) → `json.loads` échoue → 0 claim → fallback |
| B2 | **Hallucination inter-domaine** | « real-world images as additional references » pour une question RAG texte | Contamination : chunk image-gen dans le pool, le LLM copie la phrase sur de mauvaises sources |
| B3 | **Doublons de citations** | « [2][3] » ajouté sur une phrase contenant déjà « [2] and [3] » | `render_claims_answer` recolle les marqueurs sans détecter les inline |
| B4 | **Sur-citation** | Un claim cite [1..5] | Aucune limite explicite |
| B5 | **Rejet systématique des paraphrases** | Scores NLI 0.01-0.16 pour des réponses correctes | Biais *neutral* de roberta-large-mnli sur la paraphrase |
| B6 | **Réponse hors sujet supportée** | « FAiD shows… » au lieu d'une réponse | Le repli extractif assemble des faits supportés mais hors question |
| B7 | **Doublons dans la table MNLI** | 4 lignes identiques pour une affirmation | `segment_claims` collecte toutes les occurrences [N] (2,3,2,3) |
| B8 | **Fenêtre NLI trop large** | entailment dilué (chunk entier) | Prémisse = chunk complet au lieu d'une fenêtre locale |

---

## 25. Corrections apportées

### 25.1. Parsing tolérant (B1)

`parse_claim_json` : `json.loads(strict=False)` puis, en cas d'échec, **fallback regex** sur `"text":"...", "citations":[...]` qui neutralise chaque token non entier (un `2512.13930v1` est écarté, `[3,4]` entiers survivent).

### 25.2. Anti-duplication citations et table MNLI (B3, B7)

- `render_claims_answer` : si la phrase contient déjà des marqueurs `[N]` inline, **aucun marqueur n'est ajouté** ;
- `segment_claims` (citation_verifier) : indices extraits et **dédupliqués** par phrase.

### 25.3. Sanitation et anti-sur-citation (B4)

```
_sanitize_claim_citations(claims, n_sources, max_per_claim=2):
    pour chaque claim:
        valides = {i dans 1..n_sources} ∪ {inline [N]}
        retirer marqueurs [N] du texte
        retirer connecteurs orphelins ("as described in")
        garder <= max_per_claim citations
```
### 25.4. Prompt : consistance citation-contenu (B2)

Ajout de deux règles SYSTEM_PROMPT : « every fact in a claim must come from ITS OWN citations » et « sources may describe different tasks or modalities — do not mix them ». L'hallucination « real-world images » a disparu des exécutions de contrôle.

### 25.5. Gate NLI + repli pertinent anti hors-sujet (B5, B6)

`apply_nli_gate` : si toutes les citations de la réponse ont support < SUPPORT_LOW,
1. **tolérance paraphrase** : si chaque claim est couvert à ≥ 0.40 par une source et non contredit par la meilleure source → réponse **conservée** ;
2. sinon **repli extractif** : claims verbatim filtrés par pertinence lexicale avec la question ;
3. sinon **REFUS explicite** (REFUSAL_RESPONSE).

### 25.6. Fenêtre locale NLI (B8)

`extract_local_premise` : découpe le chunk en phrases, sélectionne la phrase de plus fort overlap lexical ± 1 phrase — la prémisse passe de ~500 à ~150-250 tokens, entailment nettement moins dilué.

### 25.7. Remappage sémantique des citations

Réparation NLI : chaque claim est ancré vers les **sources qui le supportent réellement** (2 max), mesuré par la même NLI que la vérification — corrige le cas « fait bon dans [3], cité [3,4] » où la paire (claim,[4]) tombait à 0.013 alors que (claim,[3]) valait 0.799.

---

## 26. Analyse des hallucinations

### 26.1. Typologie observée

Sur les échantillons audités (traces réelles), trois classes :

1. **Hallucination de contenu** (rare, bien détectée) : le modèle invente un fait absent du contexte → le NLI donne *contradiction* (P ≥ 0.75 sur les contrefactuels de contrôle) ;
2. **Contamination inter-modale** (documentée) : fragment réel d'un chunk *hors sujet* (ex. image generation) injecté dans une réponse texte. La phrase fautive est **dans le pool**, sa source exacte existe — un filtre lexical ne la distingue pas d'un claim correct (cov ≈ 0.50-0.88, mesurée sur toutes les variantes) ; seule une barrière sémantique la détecterait ;
3. **Paraphrase correcte mais non reconnue** : le fait est vrai et ancré, mais roberta-large-mnli répond *neutral* → support artificiellement bas. C'est la classe **dominante** des rejets observés.

### 26.2. Quantification du biais du vérifieur (ablation)

Même fenêtre, mêmes faits (section 22.3) : verbatim 0.79 → Supported ; paraphrase 0.26 → Unsupported ; faux 0.0001 + contradiction 0.76 → Unsupported. Le vérifieur est **conservateur** : il ne laisse passer que le quasi-verbatim. D'où la stratégie en deux niveaux : le pipeline **génère** avec tolérance à la paraphrase (gate couverture), la table **affiche** le verdict strict du NLI — le score affiché reste une borne basse conservatrice.

### 26.3. Politique anti-hallucination résultante

| Niveau | Mécanisme | Effet |
|---|---|---|
| Prompt | Règles strictes 1-7 | Guide le modèle |
| Parsing / sanitation | JSON tolérant + limites | Structure la sortie |
| Grounding lexical | couverture ≥ 0.45 / 0.15 | Écarte les claims non ancrés |
| Gate NLI | support ≥ 0.40 + tolérance paraphrase | Refuse le non supporté, garde le correct |
| Repli extractif | claims verbatim filtrés | Réponse fiable si possible |
| Refus explicite | REFUSAL_RESPONSE | Transparence si rien n'est supporté |
| Vérification finale | NLI par citation | Preuves visibles à l'utilisateur |
---

## 27. Limites du système

### 27.1. Biais *neutral* du vérifieur NLI

La limite la plus structurante : roberta-large-mnli pénalise la paraphrase (support 0.26 pour des paraphrases correctes vs 0.79 en verbatim). Conséquences : (i) la table Citation Verification affiche des scores conservateurs ; (ii) la Faithfulness mesurée est une borne basse ; (iii) il faut maintenir une tolérance parallèle dans le gate, source de complexité.

### 27.2. Volatilité du LLM local

qwen2.5:3b est non déterministe même à température faible ; le système compense par des contrôles déterministes, mais la qualité de réponse finale varie run à run. Un modèle plus grand ou des techniques de décodage contraintes (grammar) réduiraient cette variance.

### 27.3. Contamination inter-modale résiduelle

Un fragment réel d'un chunk hors sujet peut traverser les filtres lexicaux du gate (la phrase existe verbatim dans le pool). Seule une vérification sémantique stricte (à l'étape de génération) rejettera ce cas — coûteuse en NLI.

### 27.4. Coût et latence

La génération Ollama domine la latence (6-23 s sur CPU). La double vérification NLI (gate + vérification finale) ajoute ~2-3 s ; un cache des paires (prémisse, hypothèse) est possible mais non branché dans l'interface.

### 27.5. Corpus et annotation

Le jeu de validation est limité en taille ; Recall@5 = 75 % et MRR = 72,8 % reposent sur un petit échantillon annoté. Les résultats sont **indicatifs**, pas statistiquement garantis.

### 27.6. Périmètre disciplinaire

Le corpus est exclusivement scientifique (arXiv) ; les conclusions sur le chunking (412+100=512) sont liées à la borne BGE de 512 tokens et au format des articles — à revalider pour d'autres domaines.

---

## 28. Perspectives d'amélioration

### 28.1. Modèles et inférence

- **Chunking adaptatif** : découpage par sections (titre, paragraphes, tableaux) plutôt que par tokens ;
- **LLM plus grand / spécialisé** : qwen2.5:7b/14b ou modèle fine-tuné pour le format JSON (ou décodage avec *grammar* JSON imposé par le serveur) ;
- **GPU** : réduction drastique de la latence (Ollama + CUDA).

### 28.2. Vérification

- **Cache NLI partagé** : brancher `get_last_nli_scores()` dans l'interface pour ne calculer qu'une fois chaque paire (prémisse, hypothèse) — seul `app.py` doit être modifié, fonction d'exposition déjà présente ;
- **Vérifieur plus robuste à la paraphrase** : modèle fine-tuné sur données RAG, ou utilisation d'un LLM juge (LLM-as-a-judge) pour le verdict final ;
- **Seuils calibrés** par validation : ajuster SUPPORT_LOW/SUPPORT_HIGH sur un échantillon annoté.

### 28.3. Évaluation

- **Jeu de test plus large** (100+ questions annotées), métriques avec intervalles de confiance ;
- **Métriques contextuelles** : Context Relevancy, Answer Relevancy (RAGAS) complètes ;
- **Évaluation humaine** sur 50 réponses pour calibrer les seuils NLI.

### 28.4. Production

- **Étape de débat/validation croisée** : un second LLM critique la réponse avant affichage ;
- **Surveillance en continu** : tableau de bord des taux de refus, de fallback et des scores moyens par domaine ;
- **Mécanisme de feedback utilisateur** (« mauvaise citation ») pour alimenter un réglage itératif.

---

## 29. Conclusion

Ce projet a démontré qu'un système RAG **fiable et auditable** peut être construit avec des composants open-source et un LLM local (qwen2.5:3b), sans dépendre d'API propriétaires. L'architecture hybride (dense + BM25 + RRF + reranking BGE) atteint Recall@5 = 75 % et MRR = 72,8 % sur le corpus de validation ; la génération ancrée et la vérification NLI des citations produisent une Faithfulness de 72,7 % et une Citation Accuracy de 80 %.

L'apport principal du travail réside dans la **démarche d'ingénierie de la fiabilité** : chaque défaut a été transformé en question de recherche mesurable, instrumentée par des traces et des rejeux, puis corrigé par un patch minimal validé sur des exécutions réelles. Le diagnostic le plus important — le biais *neutral* du vérifieur NLI sur les paraphrases — montre que la qualité perçue (scores affichés) peut sous-estimer la qualité réelle (réponses correctes), et que tout système de ce type doit **distinguer les trois causes** du rejet : hallucination, contamination, et limite du vérifieur.

Le système livré est opérationnel (Streamlit), reproductible (rapports structurés) et documenté (audits, traces, correctifs). Les perspectives tracées — cache NLI, LLM-as-a-judge, évaluation élargie, décodage contraint — définissent un chemin clair vers un produit de niveau industriel.
---

## 30. Compétences acquises

### 30.1. Compétences techniques

| Domaine | Compétence |
|---|---|
| NLP / RI | Embeddings, BM25, recherche hybride, RRF, reranking cross-encodeur, NLI, similarité cosinus |
| LLM appliqué | Prompt engineering contraint, parsing JSON tolérant, génération ancrée, fallback, refus explicite |
| Ingénierie logicielle | Python 3.11, dataclasses, modules découplés, `sys.path` partagé, non-régression, `py_compile` |
| Vectoriel | Qdrant (HNSW, payloads), indexation/interrogation, gestion des verrous d'accès |
| Data pipeline | Extraction PDF, chunking calibré (tokens, overlap, hard-cap), validation corpus (UUID, doublons) |
| Évaluation | Recall@k, MRR, Faithfulness, Citation Accuracy, RAGAS ; replay déterministe de traces |
| Interfaçage | Streamlit (caching, paramètres), API REST Ollama, logging structuré |
| Diagnostic | Instrumentation RAG_TRACE, audits ciblés, ablation NLI (verbatim / paraphrase / contrefactuel) |

### 30.2. Compétences transverses

- **Méthode de diagnostic** : transformer un symptôme (réponse identique au Template) en cause racine prouvée (échec de parsing) par des traces réelles — *preuves avant patch* ;
- **Conduite de correctifs minimaux** : un bug → un patch → une exécution de validation, sans toucher aux modules hors périmètre ;
- **Rigueur d'évaluation** : séparer hallucination, contamination et biais du vérifieur par des ablations contrôlées ;
- **Rédaction technique** : rapports d'audit, traces d'exécution, documentation du code.
---

## 31. Questions potentielles d'entretien et réponses attendues

**Q1. Pourquoi un RAG hybride dense + BM25 plutôt qu'un seul canal ?**
Complémentarité démontrée : le dense capte la paraphrase, BM25 les termes exacts (identifiants, formules, chiffres). La fusion RRF combine les rangs sans calibration. Recall@5 = 75 % sur le corpus validé ; un canal seul dégradait la couverture des questions à termes rares.

**Q2. Quelle est la formule de la RRF et pourquoi k=60 ?**
\( \sum \frac{1}{k+\text{rang}} \). k=60 est la valeur canonique (Cormack et al.) : elle évite qu'une première position domine trop. Testée sur nos données, elle a donné les meilleurs classements sans calibration.

**Q3. Comment empêchez-vous le LLM d'halluciner ?**
Multi-niveaux : prompt strict (usage exclusif du contexte, refus explicite), grounding lexical de chaque claim (couverture ≥ seuil), gate NLI (support ≥ 0.40, sinon tolérance paraphrase, repli extractif ou refus), et vérification finale des citations par NLI. On multiplie les filets de sécurité plutôt qu'on ne fait confiance au prompt.

**Q4. Comment mesurez-vous la qualité du système ?**
Recall@5 et MRR pour le retrieval ; Faithfulness (claims supportés par le contexte) et Citation Accuracy (implication NLI de chaque citation) pour la génération/vérification : 75 %, 72,8 %, 72,7 %, 80 %.

**Q5. Quelle était la cause du problème « Ollama = Template » ?**
Le JSON de qwen2.5:3b contenait des citations non entières (identifiants arXiv) → `json.loads` échouait → 0 claim → fallback Template. Correctif : parsing tolérant + fallback regex qui neutralise les tokens non entiers. Leçon : instrumenter avant de patcher.

**Q6. Pourquoi votre vérifieur NLI donne-t-il des scores faibles sur des réponses correctes ?**
roberta-large-mnli est entraîné sur des paires courtes et classe la paraphrase en *neutral* (P_n ≈ 0.72) → support ~0.26 au lieu de ~0.79 pour le verbatim. L'ablation verbatim/paraphrase/contrefactuel le prouve. C'est un biais du vérifieur, pas une hallucination — d'où la tolérance paraphrase dans le gate.

**Q7. Comment avez-vous choisi la taille des chunks ?**
Contrainte BGE : contexte 512 tokens. target=412 + overlap=100 ⇒ exactement 512 sans troncature. 412 tokens ≈ une sous-section scientifique. Audit : médiane 427, max 512, 0 chunk tronqué.

**Q8. Bi-encodeurs vs cross-encodeurs pour le reranking ?**
Bi-encodeur : représentation pré-calculée, rapide mais approximative (aucune interaction mots). Cross-encodeur : paire re-encodée ensemble, interactions fines, coûteux — d'où son usage en second étage sur le petit pool (20). Mesuré : le reranker a remonté en tête des chunks mal classés par la fusion.

**Q9. Que fait le gate NLI exactement ?**
Après génération, il vérifie par NLI qu'au moins une citation soutient réellement le claim. Si toutes les paires < SUPPORT_LOW : (1) tolérance paraphrase si couverture ≥ 0.40 et non contradiction ; (2) repli extractif pertinent ; (3) refus explicite. Aucune réponse non supportée n'est affichée.

**Q10. Comment améliorer le système si vous aviez 2 mois supplémentaires ?**
(1) décodage contraint (grammar JSON) pour stabiliser le LLM ; (2) cache NLI partagé gate/vérification (≈2-3 s économisées) ; (3) LLM-as-a-judge et calibration des seuils sur un grand échantillon annoté ; (4) chunking adaptatif par sections ; (5) évaluation humaine de 50 réponses.

---

## Annexe A. Répertoire des modules

| Module | Rôle |
|---|---|
| `extract_text.py` | Extraction du texte des documents et métadonnées |
| `chunk_documents.py` | Découpage en chunks calibrés (412 / 100 / 512) |
| `build_dense_index.py` | Encodage BGE + stockage Qdrant |
| `build_bm25_index.py` | Index lexical BM25 |
| `hybrid_search.py` | Recherche dense + BM25 + fusion RRF |
| `rerank_results.py` | Reranking BGE (pool → top-5) |
| `generate_answer.py` | Génération ancrée, parsing, sanitation, gate NLI, fallback, refus |
| `citation_verifier.py` | Segmentation des claims, fenêtre locale, NLI, verdicts |
| `evaluate_pipeline.py` | Évaluation batch (Recall@5, MRR, Faithfulness, Citation Accuracy) |
| `dashboard_evaluation.py` | Tableau de bord des métriques |
| `app.py` | Interface Streamlit (pipeline complet + rendu) |
| `validate_chunks.py` / `validate_corpus.py` | Contrôles qualité de l'ingestion |

---

*Fin du rapport — RAG Citation Validator, PFE Master IA 2026.*