"""
Extraction, nettoyage et chunking du corpus RAG (JOUR 3).

Objectif :
    Transformer les PDF validés (JOUR 2) en une collection de chunks de texte
    autonomes, la pierre angulaire des étapes suivantes :
      - J4 (Indexation Qdrant / recherche dense) : lit `chunks.json`, calcule
        les embeddings (sentence-transformers) et fait un upsert avec `id`.
      - J5 (Recherche hybride dense + BM25) : index BM25 construit sur le champ
        `text` des chunks ; les métadonnées (document_id, page, section)
        permettent de restituer la provenance exacte de chaque résultat.

Choix techniques :
    1. PyMuPDF page par page -> chaque chunk appartient à UNE SEULE page.
       Cette contrainte garde la provenance exacte (page N) pour le module de
       validation de citations (RoBERTa-MNLI, J6+) et évite les "chunks
       fantômes" traversant les limites de page.
    2. Découpage récursif par frontières sémantiques :
           titre de section / paragraphe (double saut de ligne) > phrase > mot.
       Les paragraphes trop longs sont cassés aux frontières de phrases,
       jamais en plein mot ni en plein signe.
    3. Estimation des tokens : len(text) / 4 (heuristique classique de
       ratio caractères/tokens). Cible 512 tokens (~2000 caractères),
       limite stricte 768. Chevauchement léger (~40 tokens) pour préserver
       le contexte aux frontières de chunks.
    4. IDs déterministes (uuid5) : relancer le script produit exactement les
       mêmes IDs -> reproductible et idempotent vis-à-vis de Qdrant.

Sorties (écrites sous corpus/) :
    chunks.json            la collection de chunks (artefact canonique)
    chunks_report.json     statistiques de build (machine)
    chunks_report.md       rapport lisible

Usage :
    python extract_chunks.py [--limit-docs N] [--min-section-len N]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Estimation du nombre de tokens depuis les caractères (ratio ~4 caractères/token)
CHARS_PER_TOKEN = 4.0

# Tailles des chunks (en "tokens estimés")
CHUNK_TARGET_TOKENS = 512
CHUNK_HARD_CAP_TOKENS = 768
OVERLAP_TOKENS = 40

MIN_TOKENS_TO_KEEP = 64    # en dessous : chunk jugé trop pauvre (bruit seul)

# Identifiants uuid5 : namespace constant -> IDs stables entre les exécutions
CHUNK_UUID_NAMESPACE = uuid.UUID("6fae14c2-9e8a-4b0b-9b9b-2b0d0b1a6c01")

# Détection heuristique des titres de section (utile pour les RAG, incl. les
# questions de J5+). Ordre = priorité.
SECTION_PATTERNS = [
    re.compile(r"^(abstract|introduction|related work|conclusion|references|appendix)$", re.I),
    re.compile(r"^\d+(\.\d+)*\.\s+[a-z][^\n]{3,80}$", re.I),        # 1., 1.1., 4.1.2
    re.compile(r"^[a-z][a-z0-9\s,&\-]{4,60}$", re.I),             # "Experimental Setup"
]

# Espaces typographiques qu'on normalise en espace standard
NBSP_RE = re.compile(r"[\u00a0\u2007\u202f\u2009]")


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

@dataclass
class RawPage:
    """Texte brut extrait d'une page de PDF."""
    index: int          # 1-based
    text: str


@dataclass
class Chunk:
    """Un chunk autonome, prêt à être indexé (Qdrant / BM25)."""
    id: str
    reference: str          # humain lisible : "2402.12354v2-p003-c0001"
    document_id: str
    theme: str
    title: str
    page: int
    section: str
    text: str
    tokens_est: int
    chars: int
    start_char: int           # offset dans le texte nettoyé de la page
    end_char: int


# ---------------------------------------------------------------------------
# Utilitaires de base
# ---------------------------------------------------------------------------

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
    """Charge le manifest validé du JOUR 2 (champs normalisés)."""
    with manifest_path.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    return entries


def estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (ratio caractères/token)."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def clean_text(raw: str) -> str:
    """Nettoie le texte extrait d'une page.

    1. Normalisation Unicode (NFKC) : fusionne les glyphes équivalents
       (ligatures, espaces insécables) et évite les faux négatifs de
       recherche BM25 (ex. "ﬁne-tuning" vs "fine-tuning").
    2. Normalisation des retours à la ligne : on préserve la structure de
       paragraphes (le \n\n est signifiant), tout en aplatissant les retours
       purement typographiques (fin de colonne PDF, espaces de justification).
    3. Espacement normalisé (un seul espace entre les mots, sauts multiples
       réduits) pour un découpage propre par la suite.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = NBSP_RE.sub(" ", text)
    text = text.replace("\x0c", "\n")          # form feed -> saut de page
    text = re.sub(r"[ \t]+", " ", text)         # espaces/tabulations multiples
    text = re.sub(r" *\n *", "\n", text)        # espaces autour des newlines
    text = re.sub(r"\n{3,}", "\n\n", text)      # évite les cascades de vides
    return text.strip()


def detect_section_header(line: str) -> str | None:
    """Retourne le titre de section si 'line' ressemble à un en-tête de section.

    Heuristique volontairement simple : elle sert uniquement à enrichir la
    métadonnée 'section' des chunks (et non à parser les PDF de façon rigide).
    """
    candidate = line.strip().rstrip(":").strip()
    if not (2 <= len(candidate) <= 80):
        return None
    if candidate.isupper() and " " not in candidate and len(candidate) < 20:
        return candidate
    for pattern in SECTION_PATTERNS:
        if pattern.fullmatch(candidate):
            return candidate
    return None


def extract_pages(pdf_path: Path) -> list[RawPage]:
    """Extrait le texte de chaque page du PDF avec PyMuPDF."""
    pages: list[RawPage] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            pages.append(RawPage(index=i, text=raw))
    return pages


# ---------------------------------------------------------------------------
# Découpage (chunking)
# ---------------------------------------------------------------------------

def split_paragraphs(text: str, min_len: int = 0) -> list[str]:
    """Découpe le texte en paragraphes sur les sauts de ligne.

    Un '\\n' simple est quasi toujours un retour typographique (donc ignoré),
    un '\\n\\n' un vrai séparateur de paragraphe. `min_len` filtre les segments
    trop courts (par défaut 0 : on garde les titres de sections comme 'Abstract').
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if len(p) >= min_len]


def split_with_overlap(paragraph: str, target: int, cap: int) -> list[str]:
    """Découpe un très long paragraphe aux frontières de phrases.

    Garantit que chaque morceau fait <= caps tokens et réinjecte la dernière
    phrase du morceau précédent en tête du suivant (overlap contextuel).
    """
    if estimate_tokens(paragraph) <= cap:
        return [paragraph]

    # Segments sémantiques : phrases, puis découpes plus courtes si besoin.
    sentences = re.split(r"(?<=[.;!?])\s+", paragraph)
    pieces, current = [], []
    current_len = 0
    overlap_text = ""

    def flush() -> None:
        nonlocal current, current_len, overlap_text
        if not current:
            return
        piece = " ".join(current)
        if overlap_text:
            piece = f"{overlap_text} {piece}"
        pieces.append(piece)
        # On retient ~40 tokens sous forme de phrases pour le chevauchement.
        tail_chars = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
        overlap_text = " ".join(current)[-tail_chars:].lstrip(".,;: ")
        current, current_len = [], 0

    for sent in sentences:
        sent_len = estimate_tokens(sent)
        if current_len + sent_len > target and current:
            flush()
        # Un seul "token" énorme (code, tableau) : découpage binaire au plus vite.
        while sent_len > cap:
            cut = len(sent) // 2
            pieces.append(clean_text(sent[:cut]))
            sent, sent_len = sent[cut:], estimate_tokens(sent[cut:])
        current.append(sent)
        current_len += sent_len
    flush()
    return pieces


def build_chunks_for_page(
    raw_page: RawPage,
    entry: dict,
    chunk_seq_start: int,
) -> tuple[list[Chunk], int]:
    """Construit les chunks d'une page (tous bornés à CETTE page).

    `chunk_seq_start` = nombre de chunks déjà émis pour ce document (avant cette
    page) : il sert à créer des références globalement uniques, dans l'ordre.

    Retourne (chunks, nouvelle_valeur_de_sequence).
    """
    text = clean_text(raw_page.text)
    chunks: list[Chunk] = []
    section = "unknown"
    chunk_seq = chunk_seq_start

    # -- buffer courant : accumulateur de paragraphes jusqu'à la cible ---
    current = ""
    current_len = 0
    current_start = 0          # offset du buffer dans `text` (provenance)
    cursor = 0                 # position de recherche du prochain paragraphe

    def flush() -> None:
        nonlocal current, current_len, current_start, chunk_seq
        if current and estimate_tokens(current) >= MIN_TOKENS_TO_KEEP:
            chunk_seq += 1
            tokens = estimate_tokens(current)
            ref = f"{entry['id']}-p{raw_page.index:03d}-c{chunk_seq:04d}"
            chunk_id = str(uuid.uuid5(CHUNK_UUID_NAMESPACE, ref + "|" + current[:200]))
            chunks.append(Chunk(
                id=chunk_id,
                reference=ref,
                document_id=entry["id"],
                theme=entry.get("theme", ""),
                title=entry.get("title", ""),
                page=raw_page.index,
                section=section,
                text=current,
                tokens_est=tokens,
                chars=len(current),
                start_char=current_start,
                end_char=current_start + len(current),
            ))
        current = ""
        current_len = 0
        current_start = 0

    for paragraph in split_paragraphs(text):
        # Détection d'en-tête de section (en premier : les titres sont souvent
        # des segments courts type "Abstract" — ils ne doivent pas être filtrés).
        header = detect_section_header(paragraph)
        if header is not None and estimate_tokens(paragraph) <= 8:
            section = header
            cursor += len(paragraph) + 2   # +2 = "\n\n" séparateur
            continue

        # Segment court non-section (numéro de page, auteur, etc.) : on l'avale.
        if len(paragraph) < 20:
            cursor += len(paragraph) + 2
            continue

        para_start = text.find(paragraph, cursor)
        if para_start < 0:
            para_start = cursor
        cursor = para_start + len(paragraph)

        para_tokens = estimate_tokens(paragraph)

        # --- Paragraphe très long : vidage forcé puis découpe avec overlap ---
        if para_tokens > CHUNK_TARGET_TOKENS:
            flush()
            for piece in split_with_overlap(
                paragraph, CHUNK_TARGET_TOKENS, CHUNK_HARD_CAP_TOKENS
            ):
                current = piece
                current_len = para_tokens
                current_start = para_start
                flush()
            continue

        # --- Accumulation normale dans le buffer jusqu'à la cible ---
        if current and current_len + para_tokens > CHUNK_TARGET_TOKENS:
            flush()
        if not current:
            current_start = para_start
        current = current + (" " if current else "") + paragraph
        current_len += para_tokens + 1

    flush()
    return chunks, chunk_seq


def build_chunks_for_document(
    entry: dict,
    corpus_root: Path,
    docs_processed: int,
) -> tuple[list[Chunk], int]:
    """Extrait et découpe un document entier du manifest.

    Retourne (chunks, nb_pages_lues). Lève FileNotFoundError si le PDF manque.
    """
    pdf = corpus_root / entry["file_path"]
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf}")

    pages = extract_pages(pdf)
    doc_chunks: list[Chunk] = []
    pages_with_chunks = 0
    seq = 0
    for raw_page in pages:
        page_chunks, seq = build_chunks_for_page(raw_page, entry, seq)
        if page_chunks:
            pages_with_chunks += 1
        doc_chunks.extend(page_chunks)
    return doc_chunks, pages_with_chunks


# ---------------------------------------------------------------------------
# Sérialisation des artefacts
# ---------------------------------------------------------------------------

def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Convertit un Chunk en dictionnaire JSON (schéma stable pour J4/J5)."""
    return {
        "id": chunk.id,                          # uuid5 stable -> upsert Qdrant
        "reference": chunk.reference,            # lisible : doc-page-chunk
        "document_id": chunk.document_id,
        "theme": chunk.theme,
        "title": chunk.title,
        "page": chunk.page,
        "section": chunk.section,
        "text": chunk.text,                      # -> index BM25 (J5)
        "tokens_est": chunk.tokens_est,          # budget token J4/LLM
        "chars": chunk.chars,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def write_outputs(
    corpus_root: Path,
    chunks: list[Chunk],
    stats: dict[str, Any],
) -> None:
    """Écrit chunks.json + chunks_report.json + chunks_report.md."""
    payload = {
        "generator": "extract_chunks.py (JOUR 3)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "chars_per_token": CHARS_PER_TOKEN,
            "chunk_target_tokens": CHUNK_TARGET_TOKENS,
            "chunk_hard_cap_tokens": CHUNK_HARD_CAP_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "min_tokens_to_keep": MIN_TOKENS_TO_KEEP,
            "chunks_bounded_to_single_page": True,
        },
        "stats": stats,
        "chunks": [chunk_to_dict(c) for c in chunks],
    }
    (corpus_root / "chunks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Rapport de build des chunks (JOUR 3)",
        "",
        f"- Généré le : {payload['generated_at']}",
        f"- Corpus : `{corpus_root}`",
        "",
        "## Indicateurs",
        "",
        "| Indicateur | Valeur |",
        "| --- | --- |",
        f"| Documents traités | {stats['documents']} |",
        f"| Pages extraites | {stats['pages']} |",
        f"| Chunks produits | {stats['chunks']} |",
        f"| Chunks / document (moyenne) | {stats['chunks_per_doc']:.1f} |",
        f"| Caractères totaux | {stats['chars']:,} |",
        f"| Tokens estimés totaux | {stats['tokens_est']:,} |",
        f"| Tokens / chunk (médiane) | {stats['median_tokens']} |",
        f"| Tokens / chunk (p90) | {stats['p90_tokens']} |",
        f"| Chunks au-dessus du cap ({CHUNK_HARD_CAP_TOKENS}) | {stats['over_cap']} |",
        f"| Erreurs / documents ignorés | {stats['errors']} |",
        "",
    ]
    # Lignes par thème
    lines += ["| Thème | Documents | Chunks | Tokens est. |", "| --- | --- | --- | --- |"]
    for theme, per_theme in sorted(stats["per_theme"].items()):
        lines.append(
            f"| {theme} | {per_theme['documents']} | {per_theme['chunks']} | "
            f"{per_theme['tokens_est']:,} |"
        )
    lines.append("")
    if stats["errors"]:
        lines += ["## Erreurs", ""]
        for err in stats["errors"][:20]:
            lines.append(f"- {err}")
    (corpus_root / "chunks_report.md").write_text("\n".join(lines), encoding="utf-8")
    (corpus_root / "chunks_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraction, nettoyage et chunking du corpus (JOUR 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python extract_chunks.py                 # build complet du corpus\n"
            "  python extract_chunks.py --limit-docs 3  # test rapide sur 3 documents\n"
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


def compute_stats(
    chunks: list[Chunk],
    per_theme: dict[str, dict[str, int]],
    nb_docs: int,
    nb_pages: int,
    errors: list[str],
    docs_processed: int,
) -> dict[str, Any]:
    """Calcule les indicateurs du build (agrégats + distribution des chunks)."""
    tokens = sorted(c.tokens_est for c in chunks)

    def percentile(rank: float) -> int:
        if not tokens:
            return 0
        idx = min(len(tokens) - 1, int(rank * len(tokens)))
        return tokens[idx]

    return {
        "documents": docs_processed,
        "documents_requested": nb_docs,
        "pages": nb_pages,
        "chunks": len(chunks),
        "chunks_per_doc": len(chunks) / docs_processed if docs_processed else 0.0,
        "chars": sum(c.chars for c in chunks),
        "tokens_est": sum(tokens),
        "median_tokens": percentile(0.5),
        "p90_tokens": percentile(0.9),
        "over_cap": sum(1 for t in tokens if t > CHUNK_HARD_CAP_TOKENS),
        "errors": errors,
        "per_theme": per_theme,
    }


def main() -> None:
    args = parse_args()
    corpus_root, manifest_path = locate_corpus()
    entries = load_manifest(manifest_path)

    if args.limit_docs:
        entries = entries[: args.limit_docs]
        print(f"[JOUR 3] Mode test : {len(entries)} document(s) seulement")

    all_chunks: list[Chunk] = []
    errors: list[str] = []
    per_theme: dict[str, dict[str, int]] = {}
    nb_pages_total = 0

    for i, entry in enumerate(tqdm(entries, desc="Extraction & chunking", unit="doc")):
        theme = entry.get("theme", "?")
        per_theme.setdefault(theme, {"documents": 0, "chunks": 0, "tokens_est": 0})
        per_theme[theme]["documents"] += 1

        try:
            doc_chunks, pages_with_chunks = build_chunks_for_document(entry, corpus_root, i)
        except Exception as exc:  # noqa: BLE001 — un document ne bloque pas le corpus
            errors.append(f"{entry.get('id')}: {exc}")
            continue

        all_chunks.extend(doc_chunks)
        nb_pages_total += pages_with_chunks
        per_theme[theme]["chunks"] += len(doc_chunks)
        per_theme[theme]["tokens_est"] += sum(c.tokens_est for c in doc_chunks)

    stats = compute_stats(
        chunks=all_chunks,
        per_theme=per_theme,
        nb_docs=len(entries),
        nb_pages=nb_pages_total,
        errors=errors,
        docs_processed=len(entries) - len(errors),
    )
    write_outputs(corpus_root, all_chunks, stats)

    # --- Résumé console ---
    print("\n[JOUR 3] Extraction & chunking terminés")
    print(f"  · Documents traités   : {stats['documents']} ({stats['errors']} erreur(s))")
    print(f"  · Pages extraites     : {stats['pages']}")
    print(f"  · Chunks produits     : {stats['chunks']} "
          f"({stats['chunks_per_doc']:.1f} chunks/doc)")
    print(f"  · Tokens estimés      : {stats['tokens_est']:,} "
          f"(médiane {stats['median_tokens']} / chunk)")
    print(f"  · Chunks > cap        : {stats['over_cap']} "
          f"(cap = {CHUNK_HARD_CAP_TOKENS} tokens)")
    print("=> chunks.json, chunks_report.json, chunks_report.md générés")


if __name__ == "__main__":
    main()