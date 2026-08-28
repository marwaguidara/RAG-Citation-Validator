# RAG Citation Validator

**Système RAG Hybride avec Vérification Automatique des Citations**

---

## 🧭 Table des matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Évaluation](#évaluation)
- [Résultats](#résultats)
- [Limites & perspectives](#limites--perspectives)
- [Structure du projet](#structure-du-projet)
- [Licence](#licence)

---

## Aperçu

**RAG Citation Validator** est un pipeline de *Retrieval-Augmented Generation* complet, conçu pour répondre à une question critique : **les citations générées par le LLM sont-elles réellement supportées par les documents sources ?**

Le système combine :

- **Recherche hybride** (dense + BM25) avec fusion RRF  
- **Reranking** par cross-encoder BGE  
- **Génération de réponse** avec citations explicites (via Ollama / qwen2.5:3b)  
- **Vérification automatique des citations** par un modèle NLI (RoBERTa-large-MNLI)  
- **Tableau de bord d’évaluation** comparant 4 configurations (Dense seul, Hybride, +Rerank, +Vérification)

Ce projet a été développé dans un objectif de démonstration de compétences en **AI Engineering 2026** : reproductibilité, évaluation rigoureuse, gestion des hallucinations et traçabilité.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Corpus PDF                              │
│                       (40 articles arXiv)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ Extraction texte │
                    │   (PyMuPDF)      │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │   Chunking       │
                    │  (412/100/512)   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     ┌─────────────────┐           ┌─────────────────┐
     │   Dense Index   │           │   BM25 Index    │
     │  (BGE + Qdrant) │           │  (rank_bm25)    │
     └────────┬────────┘           └────────┬────────┘
              └──────────────┬──────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Recherche       │
                    │  Hybride (RRF)   │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │   Reranking      │
                    │ (BGE CrossEnc.)  │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  Génération LLM  │
                    │ (Ollama/qwen2.5) │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │ Vérification NLI │
                    │ (RoBERTa-MNLI)   │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │ Réponse finale   │
                    │  + support score │
                    └─────────────────┘
```

---

## Stack technique

| Composant               | Technologie / Modèle                          |
|-------------------------|-----------------------------------------------|
| Langage                 | Python 3.11                                   |
| Extraction texte        | PyMuPDF                                       |
| Indexation dense        | Qdrant (local) + BAAI/bge-small-en-v1.5      |
| Indexation lexicale     | BM25Okapi (rank_bm25)                         |
| Fusion                  | RRF (k=60)                                    |
| Reranking               | BAAI/bge-reranker-base                        |
| LLM                     | Ollama + qwen2.5:3b (fallback Template)       |
| Vérification citations  | roberta-large-mnli (HuggingFace)              |
| Interface utilisateur   | Streamlit                                     |
| Tableau de bord éval.   | Streamlit (lecture de rapports pré-calculés)  |
| Évaluation              | Script `evaluate_pipeline.py`                 |

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/RAG-Citation-Validator.git
cd RAG-Citation-Validator
```

### 2. Créer un environnement Python 3.11

```bash
python -m venv venv
source venv/bin/activate   # ou venv\Scripts\activate sous Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer Ollama (pour la génération LLM)

Suivez les instructions sur [ollama.com](https://ollama.com) puis :

```bash
ollama pull qwen2.5:3b
```

### 5. Télécharger le corpus (facultatif)

Le corpus (40 articles arXiv) est déjà inclus dans `files/corpus/`. Vous pouvez le régénérer avec :

```bash
cd files
python download_corpus.py
```

### 6. Construire les index

```bash
python build_dense_index.py
python build_bm25_index.py
```

---

## Utilisation

### Interface Streamlit (application finale)

```bash
streamlit run app.py
```

Ouvrez l'URL affichée (généralement `http://localhost:8501`). Posez une question, le système retourne une réponse avec les citations vérifiées et les scores de support.

### Tableau de bord d'évaluation

```bash
streamlit run dashboard_evaluation.py
```

Affiche les résultats pré-calculés des 4 configurations : tableaux, graphiques, KPI, analyse des gains.

### Ligne de commande – génération isolée

```bash
cd files
python generate_answer.py --question "What is Retrieval-Augmented Generation?"
```

### Ligne de commande – évaluation complète

```bash
cd files
python evaluate_pipeline.py   # toutes les configurations A/B/C/D (peut prendre ~15 min)
```

---

## Évaluation

Le framework d'évaluation mesure :

- **Recall@k** : proportion de chunks pertinents retrouvés dans le top-k  
- **MRR** : classement du premier chunk pertinent  
- **Faithfulness** : cohérence entre la réponse générée et les sources citées (via BERTScore)  
- **Citation Accuracy** : proportion de citations effectivement supportées par le NLI  
- **Latence** : temps moyen et P95 par configuration

Les métriques sont calculées sur un ensemble de **30 questions annotées** (10 RAG, 10 Agents, 10 Fine-tuning), avec un ground truth par `chunk_id`.

---

## Résultats

### Configuration comparée

| Configuration               | Recall@5 | MRR   | Faithfulness | Citation Acc. | Latence P95 |
|-----------------------------|----------|-------|--------------|---------------|-------------|
| A – Dense seul              | 61.7 %   | 0.597 | –            | –             | 31 ms       |
| B – Hybride (Dense+BM25)    | 65.0 %   | 0.660 | –            | –             | 31 ms       |
| C – Hybride + Reranking     | **75.0 %** | **0.728** | –        | –             | 7,2 s       |
| D – C + Vérification NLI    | **75.0 %** | **0.728** | **72.7 %** | **80.0 %**    | 13,1 s      |

- **BM25** apporte +5,3 points de Recall@5 (61,7 → 65,0) sans coût de latence notable.
- **Reranking** ajoute +10 points de Recall@5 (65,0 → 75,0) au prix d’une latence de 7 secondes.
- **Vérification NLI** ne modifie pas le retrieval mais fournit une **couche de transparence** : 80 % des citations sont jugées supportées, et la Faithfulness atteint 72,7 %.

---

## Limites & perspectives

### Limites connues

- **Corpus** : limité à 40 articles arXiv ; le système ne couvre pas certains sujets (ex. ReAct original absent).
- **Biais NLI** : RoBERTa-large-MNLI classe souvent les paraphrases comme « neutral » → les scores de support peuvent être sous-estimés (mais jamais surestimés).
- **Génération stochastique** : qwen2.5:3b produit des variations d’un run à l’autre ; la reproductibilité est assurée par les traces et le parsing tolérant.
- **Latence** : environ 13 secondes en CPU pour la configuration complète (domaine du reranking et du NLI).

### Perspectives d’amélioration

- **Passer à un modèle NLI plus performant** (DeBERTa-v3-large-MNLI) ou utiliser un LLM-juge avec prompt calibré.
- **Remplacer la fenêtre locale** par une méthode de *cross-attention* entre claim et chunk.
- **Paralléliser** les appels NLI et le reranking sur GPU.
- **Ajouter un cache des scores NLI** (déjà prévu via `get_last_nli_scores()`) pour éviter les doubles calculs dans `app.py`.
- **Étendre le corpus** à d’autres domaines et augmenter la taille du jeu de test annoté.

---

## Structure du projet

```
RAG-Citation-Validator/
├── app.py                      # Interface Streamlit (question/réponse)
├── dashboard_evaluation.py     # Tableau de bord d'évaluation
├── files/                      # Modules principaux
│   ├── extract_text.py
│   ├── chunk_documents.py
│   ├── build_dense_index.py
│   ├── build_bm25_index.py
│   ├── hybrid_search.py
│   ├── rerank_results.py
│   ├── generate_answer.py
│   ├── citation_verifier.py
│   ├── evaluate_pipeline.py
│   └── ...
├── corpus/                     # Données (PDF, manifest, chunks, index)
│   ├── manifest.json
│   ├── chunks.json
│   ├── dense_index_report.json
│   ├── bm25_report.json
│   ├── evaluation_report.json
│   └── ...
├── vector_store/               # Persistance Qdrant
├── requirements.txt
├── README.md
└── ...
```

---

## Licence

Ce projet est distribué sous licence **MIT**. Vous êtes libre de l’utiliser, de le modifier et de le redistribuer, à condition de conserver la mention de l’auteur original.

---



---

**Merci d’avoir pris le temps de découvrir ce projet !**