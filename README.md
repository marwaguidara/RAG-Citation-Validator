# RAG Citation Validator

Système RAG hybride (recherche dense + BM25 + reranking) avec vérification automatique des citations par score de confiance NLI. Corpus : papers de recherche sur le RAG, les agents LLM et le fine-tuning.

*Projet en cours de développement — documentation complète à venir.*

## Avancement

| Jour | Étape | Statut |
| --- | --- | --- |
| J1 | Téléchargement du corpus arXiv (`files/download_corpus.py`) | ✅ |
| J2 | Nettoyage et validation du corpus (`files/validate_corpus.py`) | ✅ |
| J3 | Extraction du texte & chunking | ⏳ |

## JOUR 2 — Nettoyage et validation

Le script `files/validate_corpus.py` :
1. vérifie l'intégrité de chaque PDF (signature `%PDF`, taille, `%%EOF`, ouverture PyMuPDF) ;
2. valide la cohérence du manifest (thème, arxiv_id, chemins, doublons) ;
3. signale les papiers hors-sujet (heuristique de mots-clés sur le titre) ;
4. contrôle la qualité du texte extrait : pages vides, PDF scannés sans couche texte,
   bruit et encodage cassé (mojibake) ;
5. met en quarantaine les fichiers invalides / doublons / hors-sujet ;
6. régénère un manifest propre `corpus/manifest.json` (avec stats `text_quality`) +
   rapport `corpus/validation_report.{json,md}`.

```bash
cd files
python validate_corpus.py                        # dry-run (rapport seul)
python validate_corpus.py --quarantine           # déplace invalides/doublons
python validate_corpus.py --quarantine --quarantine-offtopic   # + hors-sujets
python validate_corpus.py --drop 1411.4510v1     # retrait manuel ciblé
```

État à la fin du JOUR 2 : **40 documents publiés (828 pages)** — rag (13), agents (13), fine_tuning (14) ; 8 fichiers écartés vers `corpus/_quarantine/`. Contrôle qualité du texte : **2 283 000 caractères / 402 700 mots extraits**, aucune page vide, aucun mojibake, aucun document sans couche texte.
