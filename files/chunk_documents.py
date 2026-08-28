"""
Découpage du corpus en chunks optimisés RAG (JOUR 3, étape B).

Input :
    corpus/documents.json    (produit par extract_text.py)

Objectif :
    Transformer le texte extrait page par page en une collection de chunks
    autonomes, calibrés pour un système RAG hybride (dense + BM25 + reranker).

Paramètres (rationnel complet dans le message de livraison) :
    - TARGET_TOKENS = 412 : granularité "une affirmation + son développement".
      Choix calibré pour la borne de contexte des embeddings BGE (~512 tokens) :
      412 (contenu nouveau) + 100 (overlap) = 512 -> AUCUNE troncature du vecteur
      dense (J4), ni du reranker BGE (J6). Densité lexicale suffisante pour BM25.
      (L'ancien défaut 500 produisait des chunks jusqu'à ~600-996 tokens et était
       donc tronqué par BGE ; il reste accessible via --target-tokens 500.)
    - OVERLAP_TOKENS = 100 (24 %) : chaque frontière de chunk est présente dans
      les deux chunks adjacents -> aucune information n'est perdue exactement sur
      la frontière d'un découpage, pour la recherche dense comme pour BM25.
      L'overlap est coupé sur une FRONTIÈRE DE MOT (jamais en plein mot).

Propriétés des chunks émis :
    chunk_id     uuid5 déterministe -> reproductible entre exécutions (upsert Qdrant).
    document_id  identifiant arXiv du document source.
    theme        appartenance thématique (rag / agents / fine_tuning).
    page_start   première page du chunk.
    page_end     dernière page du chunk  (provenance exacte pour la vérification
                 de citations RoBERTa-MNLI).
    text         contenu (<= hard_cap tokens, frontières de phrases/mots).
    tokens_est   estimation (informative, non contractuelle).

Sortie :
    corpus/chunks.json            la collection de chunks (artefact canonique J4/J5)

Usage :
    python chunk_documents.py [--limit-docs N] [--target-tokens 412] [--overlap-tokens 100]
                              [--hard-cap N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# Estimation du nombre de tokens depuis les caractères (~4 caractères/token)
CHARS_PER_TOKEN = 4.0

# Tailles de chunks (valeurs par défaut, surchargeables en CLI)
# target=412 + overlap=100 => hard_cap=512 = borne max des embeddings BGE -> aucune
# troncature dense (J4) ni reranking (J6). 412 garde une bonne granularité
# ("une affirmation + son développement") tout en restant < 512 une fois l'overlap
# ajouté. Pour récupérer l'ancien comportement (BGE tronqué, non recommandé) :
#   --target-tokens 500  (le script alertera en conséquence).
DEFAULT_TARGET_TOKENS = 412
DEFAULT_OVERLAP_TOKENS = 100

# En dessous : chunk jugé trop pauvre pour être utile (bruit de page)
MIN_CHUNK_TOKENS = 100

# Limite de contexte des modèles d'embedding BGE (BAAI/bge-*-v1_5) utilisée en J4
# (recherche dense) et J6 (reranker BGE). Au-delà, sentence-transformers tronque :
# un chunk plus grand perd son extrémité (tokens 513+). On alerte quand le hard
# cap choisi dépasse cette limite (voir get_logger / main).
BGE_MAX_TOKENS = 512

# Nom du fichier d'entrée / sortie
INPUT_FILENAME = "documents.json"
OUTPUT_FILENAME = "chunks.json"

# Namespace uuid5 : IDs stables entre les exécutions (reproductibilité).
CHUNK_UUID_NAMESPACE = uuid.UUID("6fae14c2-9e8a-4b0b-9b9b-2b0d0b1a6c01")

# Frontières de phrases (anglais pour un corpus arXiv)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def get_logger() -> logging.Logger:
    """Configure et retourne un logger console avec formatage constant."""
    logger = logging.getLogger("chunk_documents")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def locate_corpus() -> tuple[Path, Path]:
    """Retourne (corpus_root, documents_path) via plusieurs ancrages."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "corpus" / INPUT_FILENAME,
        Path.cwd() / "corpus" / INPUT_FILENAME,
        script_dir / INPUT_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent, candidate
    raise SystemExit(
        f"{INPUT_FILENAME} introuvable. Lancez d'abord extract_text.py "
        "(produit corpus/{INPUT_FILENAME})."
    )


def load_documents(path: Path) -> list[dict]:
    """Charge documents.json (sortie de extract_text.py)."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["documents"]


def estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (ratio caractères/token)."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def to_units(page: dict) -> list[dict]:
    """Découpe le texte d'une page en unités (phrases) avec attribution de page."""
    page_no = page["page"]
    units: list[dict] = []
    for part in SENTENCE_SPLIT_RE.split(page["text"]):
        part = part.strip()
        if not part:
            continue
        units.append({
            "page": page_no,
            "text": part,
            "tokens": estimate_tokens(part),
        })
    return units


def split_long_unit(unit: dict, max_tokens: int) -> list[dict]:
    """Découpe binaire d'une unité trop longue (ex. long paragraphe sans ponctuation).

    Retourne des sous-unités chacune <= max_tokens, toutes rattachées à la même page.
    """
    pieces: list[dict] = []
    remaining = unit["text"]
    max_chars = int(max_tokens * CHARS_PER_TOKEN)   # ~2000 caractères pour 500 tokens
    while remaining:
        # On prend une tranche d'environ max_tokens tokens, en se coupant sur un espace.
        take = remaining[:max_chars]
        if len(remaining) > max_chars:
            cut = take.rfind(" ")
            if cut > 0:
                take = take[:cut]
        pieces.append({
            "page": unit["page"],
            "text": take.strip(),
            "tokens": estimate_tokens(take),
        })
        remaining = remaining[len(take):].strip()
    return [p for p in pieces if p["text"]]


def extract_overlap_tail(text: str, overlap_tokens: int) -> str:
    """Derniers ~overlap_tokens du texte, servant d'amorce au chunk suivant.

    La queue est tronquée à une FRONTIÈRE DE MOT (jamais en plein mot) : si la
    limite tombe au milieu d'un mot, on avance au mot suivant. Conséquences :
      - la queue commence toujours par un mot complet -> overlap lexicalment propre
        pour BM25 et cohérence pour le reranker BGE (J6) ;
      - la queue est bornée à ~overlap_tokens (taille <= overlap_tokens * CHARS_PER_TOKEN)
        -> le `hard_cap` de `chunk_document` est STRICT : aucun chunk ne le
        déborde (utile pour BGE dense/reranker, J4/J6).
    Le contenu tronqué (ex. URL longue) reste présent dans le corps du chunk
    d'où il provient.
    """
    tail_chars = int(overlap_tokens * CHARS_PER_TOKEN)
    if len(text) <= tail_chars:
        return text.lstrip(" .,;:!?")
    start = len(text) - tail_chars
    # Frontière tombe-t-elle au milieu d'un mot ? -> on avance au mot suivant.
    if not text[start].isspace():
        sp = text.find(" ", start)
        if sp != -1:
            start = sp + 1
    return text[start:].lstrip(" .,;:!?")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(
    doc: dict,
    target_tokens: int,
    overlap_tokens: int,
    hard_cap_tokens: int | None = None,
) -> list[dict]:
    """Découpe un document en chunks de ~target_tokens avec overlap.

    Algorithme : fenêtre glissante sur les unités (phrases). Dès que la fenêtre
    atteint la cible, on émet un chunk puis on conserve en tête de la fenêtre
    suivante la fin du chunk courant (~overlap_tokens, alignée sur une frontière
    de mot) -> continuité contextuelle.

    Le `hard_cap_tokens` (défaut target+overlap) est une BORNE SUPÉRIEURE
    stricte : on vide la fenêtre AVANT de dépasser cette borne, ce qui évite les
    chunks de 2x la cible (ex. 996 tokens). Pour ne jamais dépasser la limite de
    contexte des embeddings BGE (512), fixer hard_cap <= 512.
    """
    doc_id = doc["id"]

    # 1. Aplatir toutes les pages en unités (avec attribution de page), en
    #    découpant au préalable les unités trop massives.
    units: list[dict] = []
    for page in doc["pages"]:
        for unit in to_units(page):
            if unit["tokens"] > target_tokens:
                units.extend(split_long_unit(unit, target_tokens))
            else:
                units.append(unit)
    if not units:
        return []

    chunks: list[dict] = []
    buffer: list[dict] = []
    buffer_tokens = 0
    seq = 0
    # Borne supérieure stricte d'un chunk (target + 1 overlap au max).
    hard_cap = hard_cap_tokens or (target_tokens + overlap_tokens)

    def flush() -> None:
        nonlocal buffer, buffer_tokens, seq
        if not buffer or buffer_tokens < MIN_CHUNK_TOKENS:
            buffer = []
            buffer_tokens = 0
            return

        seq += 1
        text = " ".join(u["text"] for u in buffer)
        page_start = buffer[0]["page"]
        page_end = buffer[-1]["page"]
        ref = f"{doc_id}-c{seq:04d}"
        chunk_id = str(uuid.uuid5(CHUNK_UUID_NAMESPACE, ref + "|" + text[:200]))

        chunks.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "theme": doc.get("theme", ""),
            "page_start": page_start,
            "page_end": page_end,
            "tokens_est": buffer_tokens,
            "text": text,
        })

        # Overlap : on garde la QUEUE du chunk émis (~overlap tokens) comme
        # amorce du suivant. Coupe alignée sur une FRONTIÈRE DE MOT (jamais en
        # plein mot) -> overlap lexicalment propre pour BM25 et le reranker.
        tail = extract_overlap_tail(text, overlap_tokens)
        last_page = buffer[-1]["page"]
        buffer = [{"page": last_page, "text": tail, "tokens": estimate_tokens(tail)}]
        buffer_tokens = estimate_tokens(tail)

    # units_since_flush > 0 tant qu'on a ajouté du CONTENU RÉEL (au-delà de la
    # simple queue d'overlap) depuis le dernier flush. Servira à ne pas réémettre
    # un chunk ne contenant que l'overlap (duplication en queue de document).
    units_since_flush = 0
    for unit in units:
        # Hard cap : si l'unité courante repousserait la fenêtre au-dessus de
        # `hard_cap` ET qu'elle contient déjà du vrai contenu, on vide d'abord.
        # On n'aggit pas si la file n'est que la queue d'overlap (units_since_flush
        # == 0) pour ne pas créer de chunk purement redondant.
        if buffer_tokens + unit["tokens"] > hard_cap and units_since_flush > 0:
            flush()
            units_since_flush = 0
        buffer.append(unit)
        buffer_tokens += unit["tokens"]
        units_since_flush += 1
        if buffer_tokens >= target_tokens:
            flush()
            units_since_flush = 0

    # Contenu résiduel final : on ne réémet QUE s'il contient du vrai contenu
    # (au-delà de la queue d'overlap). On évite ainsi les chunks doublons-queue.
    if units_since_flush > 0 and buffer_tokens >= MIN_CHUNK_TOKENS:
        flush()

    return chunks


# ---------------------------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------------------------

def write_chunks(
    corpus_root: Path,
    chunks: list[dict],
    stats: dict[str, Any],
    target_tokens: int,
    overlap_tokens: int,
    hard_cap_tokens: int,
) -> None:
    """Écrit corpus/chunks.json (artefact canonique pour J4/J5)."""
    payload = {
        "generator": "chunk_documents.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "chars_per_token": CHARS_PER_TOKEN,
            "target_tokens": target_tokens,
            "overlap_tokens": overlap_tokens,
            "hard_cap_tokens": hard_cap_tokens,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "bge_max_tokens": BGE_MAX_TOKENS,
        },
        "stats": stats,
        "chunks": chunks,
    }
    (corpus_root / OUTPUT_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Découpe documents.json en chunks RAG (chunks.json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python chunk_documents.py                         # 412 tokens / overlap 100 (BGE-safe)\n"
            "  python chunk_documents.py --limit-docs 3          # test sur 3 documents\n"
            "  python chunk_documents.py --overlap-tokens 50     # overlap personnalisé\n"
            "  python chunk_documents.py --hard-cap 512          # cap explicite\n"
        ),
    )
    parser.add_argument(
        "--limit-docs", type=int, default=0, metavar="N",
        help="Traiter uniquement les N premiers documents (dev/test).",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS, metavar="N",
        help=f"Taille cible d'un chunk (défaut : {DEFAULT_TARGET_TOKENS}).",
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS, metavar="N",
        help=f"Overlap entre chunks voisins (défaut : {DEFAULT_OVERLAP_TOKENS}).",
    )
    parser.add_argument(
        "--hard-cap", type=int, default=0, metavar="N",
        help="Taille max d'un chunk en tokens (0 = target+overlap). "
             "Embeddings BGE (512) -> privilégier <= 512, ex. "
             "--target-tokens 412 --overlap-tokens 100.",
    )
    return parser.parse_args()


def main() -> None:
    log = get_logger()
    args = parse_args()

    if args.target_tokens <= 0 or args.overlap_tokens < 0:
        raise SystemExit("--target-tokens > 0 et --overlap-tokens >= 0 requis.")
    hard_cap_tokens = args.hard_cap or (args.target_tokens + args.overlap_tokens)

    corpus_root, documents_path = locate_corpus()
    documents = load_documents(documents_path)
    if args.limit_docs:
        documents = documents[: args.limit_docs]
        log.info("Mode test : %d document(s) sélectionné(s)", len(documents))

    log.info(
        "Chunking : cible=%d tokens, overlap=%d tokens, hard_cap=%d",
        args.target_tokens, args.overlap_tokens, hard_cap_tokens,
    )
    if hard_cap_tokens > BGE_MAX_TOKENS:
        log.warning(
            "hard_cap=%d > %d (BGE) : les chunks seront tronques a %d tokens "
            "au moment de l'embedding dense (J4) et du rerankage (J6). "
            "Utilisez --target-tokens 412 --overlap-tokens 100 (hard_cap=512).",
            hard_cap_tokens, BGE_MAX_TOKENS, BGE_MAX_TOKENS,
        )

    all_chunks: list[dict] = []
    per_theme: dict[str, dict[str, int]] = {}
    errors: list[str] = []

    for doc in tqdm(documents, desc="chunking", unit="doc", ncols=80):
        theme = doc.get("theme", "?")
        tinfo = per_theme.setdefault(
            theme, {"documents": 0, "chunks": 0, "tokens_est": 0}
        )
        per_theme[theme]["documents"] += 1
        try:
            doc_chunks = chunk_document(
                doc, args.target_tokens, args.overlap_tokens,
                hard_cap_tokens=hard_cap_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — isolation par document
            errors.append(f"{doc.get('id')}: {exc}")
            log.warning("ÉCHEC %s — %s", doc.get("id"), exc)
            continue
        all_chunks.extend(doc_chunks)
        per_theme[theme]["chunks"] += len(doc_chunks)
        per_theme[theme]["tokens_est"] += sum(c["tokens_est"] for c in doc_chunks)

    tokens_list = sorted(c["tokens_est"] for c in all_chunks)
    multi_page = sum(1 for c in all_chunks if c["page_start"] != c["page_end"])

    def _percentile(rank):
        return (tokens_list[min(len(tokens_list) - 1, int(rank * len(tokens_list)))]
                if tokens_list else 0)

    stats = {
        "documents": len(documents),
        "errors": len(errors),
        "chunks": len(all_chunks),
        "tokens": sum(tokens_list),
        "avg_tokens_per_chunk": round(sum(tokens_list) / len(tokens_list), 1) if tokens_list else 0.0,
        "min_tokens": tokens_list[0] if tokens_list else 0,
        "median_tokens": tokens_list[len(tokens_list) // 2] if tokens_list else 0,
        "p90_tokens": _percentile(0.9),
        "max_tokens": tokens_list[-1] if tokens_list else 0,
        "hard_cap_tokens": hard_cap_tokens,
        "chunks_over_cap": sum(1 for t in tokens_list if t > hard_cap_tokens),
        "chunks_over_bge": sum(1 for t in tokens_list if t > BGE_MAX_TOKENS),
        "multi_page_chunks": multi_page,
        "per_theme": per_theme,
    }
    write_chunks(
        corpus_root, all_chunks, stats,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        hard_cap_tokens=hard_cap_tokens,
    )

    # --- Résumé console ---
    log.info("Chunking terminé.")
    log.info("  Documents chunkés : %d", stats["documents"])
    log.info("  Chunks produits   : %d", stats["chunks"])
    log.info("  Tokens (total)    : %d", stats["tokens"])
    log.info("  Tokens / chunk    : %s (médiane) / min %d / max %d",
             stats["median_tokens"], stats["min_tokens"], stats["max_tokens"])
    log.info("  Chunks multi-pages: %d", stats["multi_page_chunks"])
    log.info("  Chunks > hard_cap (%d): %d", hard_cap_tokens, stats["chunks_over_cap"])
    log.info("  Chunks > BGE 512      : %d", stats["chunks_over_bge"])
    if errors:
        log.warning("  Échecs (%d) : %s", len(errors), "; ".join(errors))
    log.info("=> %s généré", corpus_root / OUTPUT_FILENAME)


if __name__ == "__main__":
    main()