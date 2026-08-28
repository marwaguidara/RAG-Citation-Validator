"""
Extraction du texte complet des PDF vers documents.json (JOUR 3, étape A).

Objectif :
    Lire `corpus/manifest.json` (artefact validé du JOUR 2) et extraire le
    texte complet de chaque PDF avec PyMuPDF, page par page.

Sortie :
    corpus/documents.json — la couche texte brute & propre, en attendant le
    chunking (étape suivante). Chaque document expose ses pages (numérotées)
    avec le texte nettoyé. C'est l'entrée idéale de `extract_chunks` et des
    étapes J4 (indexation Qdrant) / J5 (BM25) / J6 (validation de citations).

Choix :
    - PyMuPDF page par page, avec numéro de page conservé (champ `page`).
    - Nettoyage basique : normalisation Unicode, espaces multiples réduits,
      retours de ligne normalisés. On NE regroupe PAS ici les lignes en
      paragraphes (c'est le rôle du chunking).
    - Logs clairs (module logging) et gestion d'erreur par document : un PDF
      en échec ne bloque pas les 39 autres.

Sortie :
    corpus/documents.json        les documents extraits (artefact canonique)

Usage :
    python extract_text.py [--limit-docs N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# Espaces typographiques anoriaux à remplacer par une espace standard
NBSP_RE = re.compile(r"[\u00a0\u2007\u202f\u2009]")

# Fichiers de sortie
OUTPUT_FILENAME = "documents.json"


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def get_logger() -> logging.Logger:
    """Configure et retourne un logger console avec formatage constant."""
    logger = logging.getLogger("extract_text")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def locate_corpus() -> tuple[Path, Path]:
    """Retourne (corpus_root, manifest_path) via plusieurs ancrages (script / CWD)."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "corpus" / "manifest.json",
        Path.cwd() / "corpus" / "manifest.json",
        script_dir / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent, candidate
    raise SystemExit(
        "manifest.json introuvable. Lancez le script depuis files/ "
        "ou après avoir généré le corpus (JOUR 1/2)."
    )


def load_manifest(manifest_path: Path) -> list[dict]:
    """Charge le manifest validé (champs normalisés du JOUR 2)."""
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(raw: str) -> str:
    """Nettoyage basique d'une page extraite.

    1. Normalisation Unicode (NFKC) : glyphes équivalents fusionnés.
    2. Espaces insécables remplacés par des espaces standard.
    3. Suppression des espaces multiples (espace/tabulations répétés).
    4. Espaces autour des retours à la ligne supprimés, cascades de vides
       réduites.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = NBSP_RE.sub(" ", text)
    text = text.replace("\x0c", "\n")          # form feed -> saut de page
    text = re.sub(r"[ \t]+", " ", text)         # espaces/tabulations multiples
    text = re.sub(r" *\n *", "\n", text)        # espaces autour des newlines
    text = re.sub(r"\n{3,}", "\n\n", text)      # évite les cascades de vides
    return text.strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extrait le texte courant de chaque page d'un PDF (numéro de page 1-based)."""
    pages: list[dict] = []
    with fitz.open(str(pdf_path)) as doc:
        if not doc.is_pdf:
            raise ValueError(f"non reconnu comme PDF : {pdf_path.name}")
        for i in range(doc.page_count):
            raw = doc.load_page(i).get_text("text") or ""
            pages.append({"page": i + 1, "text": clean_text(raw)})
    return pages


# ---------------------------------------------------------------------------
# Extraction d'un document
# ---------------------------------------------------------------------------

def extract_document(entry: dict, corpus_root: Path) -> dict:
    """Extrait le texte complet d'un document du manifest.

    Retourne une structure Document normalisée. Lève une exception si le PDF
    est introuvable ou illisible (gérée par l'appelant).
    """
    pdf = corpus_root / entry["file_path"]
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf}")

    pages = extract_pages(pdf)
    chars_total = sum(len(p["text"]) for p in pages)
    return {
        "id": entry["id"],
        "theme": entry.get("theme", ""),
        "title": entry.get("title", ""),
        "file_path": entry["file_path"],
        "page_count": len(pages),
        "chars": chars_total,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------------------------

def write_documents(corpus_root: Path, documents: list[dict], stats: dict[str, Any]) -> None:
    """Écrit corpus/documents.json (artefact canonique pour la suite)."""
    payload = {
        "generator": "extract_text.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "documents": documents,
    }
    (corpus_root / OUTPUT_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraction du texte complet des PDF (manifest -> documents.json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python extract_text.py                  # extraire tout le corpus\n"
            "  python extract_text.py --limit-docs 3 # test rapide sur 3 documents\n"
        ),
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        metavar="N",
        help="Traiter uniquement les N premiers documents (dev/test seulement).",
    )
    return parser.parse_args()


def main() -> None:
    log = get_logger()
    args = parse_args()

    corpus_root, manifest_path = locate_corpus()
    entries = load_manifest(manifest_path)
    if args.limit_docs:
        entries = entries[: args.limit_docs]
        log.info("Mode test : %d document(s) sélectionné(s)", len(entries))

    log.info("Corpus  : %s", corpus_root)
    log.info("Début de l'extraction de %d document(s)", len(entries))

    documents: list[dict] = []
    errors: list[dict] = []
    pages_total = 0
    chars_total = 0

    for i, entry in enumerate(
        tqdm(entries, desc="extraction", unit="doc", ncols=80), start=1
    ):
        doc_id = entry.get("id", "?")
        try:
            doc = extract_document(entry, corpus_root)
        except Exception as exc:  # noqa: BLE001 — isolation par document
            message = f"{type(exc).__name__}: {exc}"
            log.warning("[%d/%d] ÉCHEC %s — %s", i, len(entries), doc_id, message)
            errors.append({"id": doc_id, "error": message})
            continue

        documents.append(doc)
        pages_total += doc["page_count"]
        chars_total += doc["chars"]
        log.info(
            "[%d/%d] OK %s — %d pages, %d caractères",
            i, len(entries), doc_id, doc["page_count"], doc["chars"],
        )

    # Tri stable (thème, id) -> documents.json reproductible
    documents.sort(key=lambda d: (d.get("theme", ""), d.get("id", "")))

    stats = {
        "documents_requested": len(entries),
        "documents_extracted": len(documents),
        "errors": len(errors),
        "pages": pages_total,
        "chars": chars_total,
    }
    write_documents(corpus_root, documents, stats)

    # --- Résumé console ---
    log.info("Extraction terminée.")
    log.info("  Documents extraits : %d/%d", len(documents), len(entries))
    log.info("  Pages totales      : %d", pages_total)
    log.info("  Caractères totaux  : %d", chars_total)
    if errors:
        log.warning("  Échecs (%d) :", len(errors))
        for err in errors:
            log.warning("    - %s : %s", err["id"], err["error"])
    log.info("=> %s généré", corpus_root / OUTPUT_FILENAME)


if __name__ == "__main__":
    main()