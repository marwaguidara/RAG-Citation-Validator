# Rapport d'évaluation comparative — RAG Citation Validator

*Généré le 2026-08-26T18:34:44+00:00 par `evaluate_pipeline.py`.*

## 1. Périmètre et traçabilité

- **Questions évaluées** : 12 (+27 slot(s) d'annotation non rempli(s))
- **Chunks indexés dans le corpus** : 2026
- **Chunks pertinents annotés** : 26
- SHA-256 `chunks.json` : `1e8908db880e7df5…`
- SHA-256 annotations : `356e7a898d2c28bb…`

## 2. Tableau comparatif complet

| Métrique | A · Dense | B · Hybrid (Dense+BM25) | C · Hybrid + Rerank | D · Hybrid + Rerank + Verification |
|---|---|---|---|---|
| recall_at_3 | 37.5% | 38.9% | 48.6% | 48.6% |
| recall_at_5 | 47.2% | 48.6% | 58.3% | 58.3% |
| recall_at_10 | 72.2% | 70.8% | 69.4% | 69.4% |
| mrr | 52.4% | 64.2% | 68.1% | 68.1% |
| faithfulness | N/A | N/A | N/A | 22.0% |
| citation_accuracy | N/A | N/A | N/A | 0.0% |
| mean_support_score | N/A | N/A | N/A | 21.4% |
| supported_rate | N/A | N/A | N/A | 0.0% |
| weak_support_rate | N/A | N/A | N/A | 20.6% |
| unsupported_rate | N/A | N/A | N/A | 79.4% |
| **average_latency_ms** | 25.9 | 28.3 | 6,752.6 | 13,898.1 |
| **p95_latency_ms** | 25.4 | 30.8 | 7,300.3 | 15,815.6 |

**Métriques explicitement non calculables :**
- `A` : faithfulness → étape de génération absente de cette configuration ; citation_accuracy → étape de vérification NLI absente de cette configuration
- `B` : faithfulness → étape de génération absente de cette configuration ; citation_accuracy → étape de vérification NLI absente de cette configuration
- `C` : faithfulness → étape de génération absente de cette configuration ; citation_accuracy → étape de vérification NLI absente de cette configuration

## 3. Analyse automatique des gains

### Ajout du canal BM25 (fusion RRF) (A · Dense → B · Hybrid (Dense+BM25))

- `recall_at_3` : 37.5% → 38.9% (**+1.4 pts**, +3.7 %)
- `recall_at_5` : 47.2% → 48.6% (**+1.4 pts**, +2.9 %)
- `recall_at_10` : 72.2% → 70.8% (**-1.4 pts**, -1.9 %)
- `mrr` : 52.4% → 64.2% (**+11.8 pts**, +22.6 %)
- Latence moyenne : 25.9 ms → **28.3 ms** (Δ +2.4 ms)

### Ajout du reranker cross-encoder BGE (B · Hybrid (Dense+BM25) → C · Hybrid + Rerank)

- `recall_at_3` : 38.9% → 48.6% (**+9.7 pts**, +25.0 %)
- `recall_at_5` : 48.6% → 58.3% (**+9.7 pts**, +20.0 %)
- `recall_at_10` : 70.8% → 69.4% (**-1.4 pts**, -2.0 %)
- `mrr` : 64.2% → 68.1% (**+3.9 pts**, +6.1 %)
- Latence moyenne : 28.3 ms → **6,752.6 ms** (Δ +6,724.3 ms)

### Ajout génération + vérification NLI des citations (C · Hybrid + Rerank → D · Hybrid + Rerank + Verification)

- `recall_at_3` : 48.6% → 48.6% (**+0.0 pts**, +0.0 %)
- `recall_at_5` : 58.3% → 58.3% (**+0.0 pts**, +0.0 %)
- `recall_at_10` : 69.4% → 69.4% (**+0.0 pts**, +0.0 %)
- `mrr` : 68.1% → 68.1% (**+0.0 pts**, +0.0 %)
- `faithfulness` : N/A → 22.0%
- `citation_accuracy` : N/A → 0.0%
- Latence moyenne : 6,752.6 ms → **13,898.1 ms** (Δ +7,145.5 ms)

## 4. Meilleure configuration

- **Verdict global qualité** : **D · Hybrid + Rerank + Verification**
- **Configuration la plus rapide** : A · Dense
- Meilleur `recall_at_3` : D · Hybrid + Rerank + Verification
- Meilleur `recall_at_5` : D · Hybrid + Rerank + Verification
- Meilleur `recall_at_10` : A · Dense
- Meilleur `mrr` : D · Hybrid + Rerank + Verification
- Meilleur `faithfulness` : D · Hybrid + Rerank + Verification
- Meilleur `citation_accuracy` : D · Hybrid + Rerank + Verification

## 5. Trade-offs précision / latence

| Étape | Δ Recall@5 (pts) | Δ latence moy. (ms) | ms par point de Recall@5 |
|---|---|---|---|
| B − A (BM25) | +1.39 | +2.4 | 1.7 |
| C − B (Reranker) | +9.72 | +6,724.3 | 691.8 |
| D − C (Génération + NLI) | +0.00 | +7,145.5 | pas de gain de recall (coût pur) |

## 6. Limites méthodologiques

- Jeu de 12 questions annotées à la main : ordres de grandeur fiables, pas un benchmark statistiquement significatif.
- Réponses de la config D générées par extraction déterministe (pas de serveur LLM) : la faithfulness mesure la chaîne retrieval → citations, pas la paraphrase d'un LLM.
- Latences mesurées sur CPU local : les valeurs absolues ne sont pas comparables entre machines, les deltas relatifs le sont.
