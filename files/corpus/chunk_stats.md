# Rapport qualité des chunks (JOUR 3)

- Généré le : 2026-08-24T18:06:15
- Source : `chunks.json` (generator=chunk_documents.py, généré le 2026-08-24T16:57:36)
- Config du chunking : chars_per_token=4.0, target_tokens=412, overlap_tokens=100, hard_cap_tokens=512, min_chunk_tokens=100, bge_max_tokens=512
- Seuils : petits <150 t · grands >700 t · quasi-doublon Jaccard ≥0.6

## Volumes

| Indicateur | Valeur |
|---|---|
| Chunks totaux | 2026 |
| Documents couverts | 40 |
| Thèmes | 3 |
| Tokens estimés (total) | 862,511 |
| Chunks au texte vide | 0 |

| Thème | Documents | Chunks | Tokens est. | Chunks/doc | Tokens/chunk |
|---|---|---|---|---|---|
| agents | 13 | 859 | 366,313 | 66.1 | 426.4 |
| fine_tuning | 14 | 624 | 265,158 | 44.6 | 424.9 |
| rag | 13 | 543 | 231,040 | 41.8 | 425.5 |

## Taille des chunks (tokens estimés)

| Métrique | Valeur |
|---|---|
| min | 100 |
| p05 | 402 |
| p10 | 412 |
| p25 | 417 |
| median | 427 |
| mean | 425.7 |
| p75 | 440 |
| p90 | 458 |
| p95 | 471 |
| p99 | 509 |
| max | 512 |

Distribution (bins de 100 tokens) :

| Bin (tokens) | Chunks |
|---|---|
| 100-199 | 18 |
| 200-299 | 35 |
| 300-399 | 43 |
| 400-499 | 1892 |
| 500-599 | 38 |

## Pages par chunk

| Indicateur | Valeur |
|---|---|
| Span moyen (pages) | 1.39 |
| Span max (pages) | 15 |
| Chunks mono-page | 1287 |
| Chunks multi-pages | 739 (36.5 %) |

| Pages / chunk | Chunks |
|---|---|
| 1 | 1287 |
| 2 | 713 |
| 3 | 19 |
| 4 | 3 |
| 5 | 1 |
| 6 | 2 |
| 15 | 1 |

## Chunks trop petits (11)

| Chunk | Document | Thème | Tokens | Pages | Aperçu |
|---|---|---|---|---|---|
| a3897811 | 2602.08239v1 | fine_tuning | 100 | 1 | for models in balls around the initial model n hyperparameters dataset cola sst  |
| 478f07b3 | 2204.07496v4 | rag | 100 | 1 | 10 upr re ranking results on the beir benchmark thakur et al 2021 q and e denote |
| 27998e18 | 2503.01763v2 | agents | 108 | 1 | l 2024 single task dense retrieval these methods use dual encoder models trained |
| 67b7d4a8 | 2605.28222v1 | fine_tuning | 112 | 1 | tables with all configurations are moved to the appendices in order not to overl |
| f86f0bb1 | 2605.28222v1 | fine_tuning | 114 | 2 | vs inference vram 64 preprint f1 vs groundedness pass 4 groundedness pass 4 vs l |
| 13a9e293 | 2309.15217v2 | rag | 120 | 1 | clock tower was built in 1896 the tower was named after chimnabai i 1864 1885 a  |
| 7e561af2 | 2210.12607v1 | fine_tuning | 128 | 1 | on em pirical methods in natural language processing pages 2369 2380 brussels be |
| fd2a64ec | 2512.15233v2 | fine_tuning | 130 | 1 | luo and qi tian parameter efficient tuning of large scale multimodal foundation  |
| d8e729a9 | 2605.12335v1 | rag | 133 | 2 | the selected tasks span both short term and long term clinical outcomes and refl |
| 2d4ed33e | 2606.01947v1 | fine_tuning | 144 | 1 | oneformer one transformer to rule universal image segmentation in proceedings of |
| 930720ab | 2404.01023v1 | agents | 149 | 1 | natural language processing arxiv preprint arxiv 1910 03771 2019 frank f xu uri  |

## Chunks trop grands (0)

_Aucun._

## Redondance / duplication

| Indicateur | Valeur |
|---|---|
| Doublons exacts (groupes) | 0 |
| Chunks redondants (exacts) | 0 |
| Paires quasi-doublons (même document) | 17 |
| Chunks impliqués (quasi-doublons) | 18 |
| Duplication excessive ? | non |

## Couverture documentaire

| Indicateur | Valeur |
|---|---|
| Documents | 40 |
| Chunks / document (moyen) | 50.6 |
| Chunks / document (min – max) | 11 – 351 |
| Couverture pages (globale) | 100.0 % |

| Document | Thème | Chunks | Tokens est. | Pages | Pages couvertes | Couverture |
|---|---|---|---|---|---|---|
| 1407.5416v1 | rag | 24 | 10325 | 1–6 | 6 | 100.0 % |
| 2010.08191v2 | rag | 40 | 16995 | 1–13 | 13 | 100.0 % |
| 2110.01599v1 | rag | 21 | 8878 | 1–7 | 7 | 100.0 % |
| 2110.06500v2 | fine_tuning | 49 | 20978 | 1–19 | 19 | 100.0 % |
| 2204.07496v4 | rag | 52 | 21976 | 1–18 | 18 | 100.0 % |
| 2208.02070v1 | fine_tuning | 23 | 9762 | 1–8 | 8 | 100.0 % |
| 2210.12607v1 | fine_tuning | 36 | 15231 | 1–11 | 11 | 100.0 % |
| 2308.16118v2 | agents | 18 | 7860 | 1–14 | 14 | 100.0 % |
| 2309.02144v1 | agents | 47 | 20205 | 1–19 | 19 | 100.0 % |
| 2309.15217v2 | rag | 24 | 10166 | 1–8 | 8 | 100.0 % |
| 2310.03059v8 | fine_tuning | 35 | 14849 | 1–11 | 11 | 100.0 % |
| 2312.10793v3 | fine_tuning | 24 | 10197 | 1–8 | 8 | 100.0 % |
| 2402.11651v2 | agents | 39 | 16508 | 1–13 | 13 | 100.0 % |
| 2402.12317v2 | rag | 54 | 22936 | 1–16 | 16 | 100.0 % |
| 2402.12354v2 | fine_tuning | 59 | 25113 | 1–24 | 24 | 100.0 % |
| 2404.01023v1 | agents | 36 | 15128 | 1–12 | 12 | 100.0 % |
| 2404.14464v1 | rag | 42 | 18136 | 1–17 | 17 | 100.0 % |
| 2405.07551v1 | agents | 40 | 16808 | 1–15 | 15 | 100.0 % |
| 2408.07888v2 | fine_tuning | 25 | 10742 | 1–10 | 10 | 100.0 % |
| 2409.11353v3 | agents | 46 | 19566 | 1–24 | 24 | 100.0 % |
| 2411.14961v3 | fine_tuning | 57 | 23878 | 1–15 | 15 | 100.0 % |
| 2411.18583v1 | rag | 20 | 8316 | 1–6 | 6 | 100.0 % |
| 2502.00306v2 | rag | 84 | 36044 | 1–27 | 27 | 100.0 % |
| 2503.01763v2 | agents | 86 | 36268 | 1–28 | 28 | 100.0 % |
| 2504.16021v1 | agents | 11 | 4522 | 1–4 | 4 | 100.0 % |
| 2504.16584v1 | fine_tuning | 23 | 9798 | 1–11 | 11 | 100.0 % |
| 2504.17204v1 | rag | 14 | 5782 | 1–6 | 6 | 100.0 % |
| 2506.06962v3 | rag | 51 | 22017 | 1–18 | 18 | 100.0 % |
| 2507.23334v2 | rag | 28 | 11843 | 1–8 | 8 | 100.0 % |
| 2508.04848v1 | agents | 34 | 14261 | 1–9 | 9 | 100.0 % |
| 2510.00071v2 | agents | 13 | 5253 | 1–7 | 7 | 100.0 % |
| 2512.13930v1 | agents | 98 | 42049 | 1–96 | 96 | 100.0 % |
| 2512.15233v2 | fine_tuning | 18 | 7487 | 1–5 | 5 | 100.0 % |
| 2601.12538v1 | agents | 351 | 150753 | 1–135 | 135 | 100.0 % |
| 2602.08239v1 | fine_tuning | 50 | 21146 | 1–24 | 24 | 100.0 % |
| 2604.14214v1 | agents | 40 | 17132 | 1–16 | 16 | 100.0 % |
| 2605.12335v1 | rag | 89 | 37626 | 1–31 | 31 | 100.0 % |
| 2605.28222v1 | fine_tuning | 110 | 46369 | 1–69 | 69 | 100.0 % |
| 2606.01947v1 | fine_tuning | 67 | 28788 | 1–25 | 25 | 100.0 % |
| 2607.11940v1 | fine_tuning | 48 | 20820 | 1–15 | 15 | 100.0 % |
