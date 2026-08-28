# Validation des chunks (JOUR 3) — WARNING

- Généré le : `2026-08-24T18:41:44`
- Source : `chunks.json` (sha256 `1e8908db880e7df5…`, 4,034,390 octets)
- Chunking : `chunk_documents.py` — config : chars_per_token=4.0, target_tokens=412, overlap_tokens=100, hard_cap_tokens=512, min_chunk_tokens=100, bge_max_tokens=512
- Référence couverture : `documents.json` (40 documents)
- Seuils : min_tokens=100, max_tokens=512, near_dup_jaccard=0.6, coverage_min_pct=99.0

## Résultat des vérifications

| Vérification | Statut | Détail |
|---|---|---|
| `chunk_id_unique` | ✅ PASS | 2026 chunk_id uniques |
| `required_fields` | ✅ PASS | tous les champs obligatoires sont présents |
| `empty_text` | ✅ PASS | aucun chunk au texte vide |
| `content_duplicates` | ⚠️ WARNING | 0 groupe(s) de doublons exacts · 17 paire(s) quasi-doublon(s) |
| `document_id_consistency` | ✅ PASS | 40 document_id cohérents (format + thème unique) |
| `page_coherence` | ✅ PASS | pages cohérentes sur 2026 chunks |
| `length_range` | ✅ PASS | 0 sous 100 t · 0 au-dessus de 512 t · 0 estimation(s) <= 0 |
| `document_coverage` | ✅ PASS | 40 document(s) · référence: 40 document(s) · 0 manquant(s) · 0 page(s) manquante(s) (trous) |

## Anomalies détectées

| Vérification | Sévérité | Message |
|---|---|---|
| `content_duplicates` | WARNING | 0 groupe(s) de doublons exacts · 17 paire(s) quasi-doublon(s) |

## Statistiques globales

| Indicateur | Valeur |
|---|---|
| chunks | 2026 |
| documents | 40 |
| themes | 3 |
| tokens_est_total | 862511 |
| tokens_est_min | 100 |
| tokens_est_max | 512 |
| tokens_est_mean | 425.7 |
| tokens_est_median | 427 |
| tokens_est_p05 | 402 |
| tokens_est_p25 | 417 |
| tokens_est_p75 | 440 |
| tokens_est_p90 | 458 |
| tokens_est_p95 | 471 |
| pages_span_mean | 1.39 |
| multi_page_chunks | 739 |
| multi_page_pct | 36.5 |

## Statistiques par thème

| Thème | Documents | Chunks | Tokens est. | Tokens/chunk |
|---|---|---|---|---|
| agents | 13 | 859 | 366,313 | 426.4 |
| fine_tuning | 14 | 624 | 265,158 | 424.9 |
| rag | 13 | 543 | 231,040 | 425.5 |

## Statistiques par document

| Document | Thème | Chunks | Tokens/chunk | Pages |
|---|---|---|---|---|
| 1407.5416v1 | rag | 24 | 430.2 | 1–6 |
| 2010.08191v2 | rag | 40 | 424.9 | 1–13 |
| 2110.01599v1 | rag | 21 | 422.8 | 1–7 |
| 2110.06500v2 | fine_tuning | 49 | 428.1 | 1–19 |
| 2204.07496v4 | rag | 52 | 422.6 | 1–18 |
| 2208.02070v1 | fine_tuning | 23 | 424.4 | 1–8 |
| 2210.12607v1 | fine_tuning | 36 | 423.1 | 1–11 |
| 2308.16118v2 | agents | 18 | 436.7 | 1–14 |
| 2309.02144v1 | agents | 47 | 429.9 | 1–19 |
| 2309.15217v2 | rag | 24 | 423.6 | 1–8 |
| 2310.03059v8 | fine_tuning | 35 | 424.3 | 1–11 |
| 2312.10793v3 | fine_tuning | 24 | 424.9 | 1–8 |
| 2402.11651v2 | agents | 39 | 423.3 | 1–13 |
| 2402.12317v2 | rag | 54 | 424.7 | 1–16 |
| 2402.12354v2 | fine_tuning | 59 | 425.6 | 1–24 |
| 2404.01023v1 | agents | 36 | 420.2 | 1–12 |
| 2404.14464v1 | rag | 42 | 431.8 | 1–17 |
| 2405.07551v1 | agents | 40 | 420.2 | 1–15 |
| 2408.07888v2 | fine_tuning | 25 | 429.7 | 1–10 |
| 2409.11353v3 | agents | 46 | 425.3 | 1–24 |
| 2411.14961v3 | fine_tuning | 57 | 418.9 | 1–15 |
| 2411.18583v1 | rag | 20 | 415.8 | 1–6 |
| 2502.00306v2 | rag | 84 | 429.1 | 1–27 |
| 2503.01763v2 | agents | 86 | 421.7 | 1–28 |
| 2504.16021v1 | agents | 11 | 411.1 | 1–4 |
| 2504.16584v1 | fine_tuning | 23 | 426.0 | 1–11 |
| 2504.17204v1 | rag | 14 | 413.0 | 1–6 |
| 2506.06962v3 | rag | 51 | 431.7 | 1–18 |
| 2507.23334v2 | rag | 28 | 423.0 | 1–8 |
| 2508.04848v1 | agents | 34 | 419.4 | 1–9 |
| 2510.00071v2 | agents | 13 | 404.1 | 1–7 |
| 2512.13930v1 | agents | 98 | 429.1 | 1–96 |
| 2512.15233v2 | fine_tuning | 18 | 415.9 | 1–5 |
| 2601.12538v1 | agents | 351 | 429.5 | 1–135 |
| 2602.08239v1 | fine_tuning | 50 | 422.9 | 1–24 |
| 2604.14214v1 | agents | 40 | 428.3 | 1–16 |
| 2605.12335v1 | rag | 89 | 422.8 | 1–31 |
| 2605.28222v1 | fine_tuning | 110 | 421.5 | 1–69 |
| 2606.01947v1 | fine_tuning | 67 | 429.7 | 1–25 |
| 2607.11940v1 | fine_tuning | 48 | 433.8 | 1–15 |
