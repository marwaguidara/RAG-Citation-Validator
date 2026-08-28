"""
Construction de l'index BM25 sur le corpus de chunks (JOUR 5, étape A).

Input :
    corpus/chunks.json    (produit par chunk_documents.py, validé par
                           validate_chunks.py)

Objectif :
    Tokeniser les chunks, construire l'index lexical BM25 (rank_bm25) puis le
    persister avec les métadonnées indispensables à la recherche.

Choix techniques :
    - Tokenizer déterministe sans dépendance externe : minuscules -> jetons
      alphanumériques (`[a-z0-9]+`) -> suppression des mots vides anglais
      (liste intégrée au script, désactivable via --no-stopwords).
    - La configuration EXACTE du tokenizer est sérialisée dans l'artefact :
      la recherche future doit tokeniser la requête à l'identique, sinon les
      scores BM25 n'ont aucun sens. C'est le point de cohérence critique.
    - Pas de stemming : volontaire. Le canal dense (BGE) couvre la variabilité
      morphologique ; ajouter un stemmer maison non testé ajouterait du risque.
    - L'ordre des documents est l'ordre trié par chunk_id => reproductible ;
      l'artefact embarque la liste `chunk_ids` qui mappe chaque position de
      l'index BM25 vers son chunk.

Artefact `corpus/bm25_index.pkl` (pickle, protocole par défaut) :
    {
      "format_version": 1,
      "bm25":            objet rank_bm25.BM25Okapi,
      "chunk_ids":       [chunk_id, ...]  (position i de BM25 => chunk_ids[i]),
      "tokenizer":       {"use_stopwords": bool},
      "params":          {"k1": float, "b": float},
    }
    NB : ne charger ce fichier que depuis une source de confiance (pickle).

Sortie :
    corpus/bm25_report.json   rapport traçable : empreinte SHA-256 de l'entrée,
    paramètres BM25/tokenizer, statistiques du corpus lexical, termes dominants,
    vérification post-construction (rechargement + requête témoin).

Code de sortie :
    0   index construit et vérifié
    1   échec (entrée invalide ou vérification négative)

Usage :
    python build_bm25_index.py
    python build_bm25_index.py --k1 1.2 --b 0.75 --no-stopwords
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
INDEX_FILENAME = "bm25_index.pkl"
REPORT_FILENAME = "bm25_report.json"

DEFAULT_K1 = 1.5             # saturation BM25 (défaut classique Okapi)
DEFAULT_B = 0.75             # normalisation par longueur de document
TOP_TERMS_LIMIT = 20         # termes les plus fréquents conservés au rapport
EXAMPLE_SAMPLE_SIZE = 3      # résultats de la requête témoin au rapport

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id", "document_id", "theme",
    "page_start", "page_end", "tokens_est", "text",
)

INDEX_FORMAT_VERSION = 1


class Status(StrEnum):
    """Verdict de la construction de l'index."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# Liste fermée de mots vides anglais (volontairement conservatrice : elle ne
# supprime aucun terme porteur de sens pour un corpus scientifique arXiv).
ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours",
})


def build_logger(as_json: bool) -> tuple[logging.Logger, "EventLogger"]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("build_bm25_index")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(message)s") if as_json
            else logging.Formatter(LOG_FORMAT_TEXT, LOG_DATE_FORMAT)
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger, EventLogger(logger, as_json=as_json)


class EventLogger:
    """Logger d'événements structurés (rendu texte clé=valeur ou JSON)."""

    def __init__(self, logger: logging.Logger, as_json: bool = False) -> None:
        self._logger = logger
        self._as_json = as_json

    @staticmethod
    def _render_text(event: str, fields: dict[str, Any]) -> str:
        parts = [f"event={event}"]
        for key, value in fields.items():
            text = str(value)
            rendered = f'"{text}"' if re.search(r"\s", text) else text
            parts.append(f"{key}={rendered}")
        return " ".join(parts)

    def _emit(self, level: int, event: str, fields: dict[str, Any]) -> None:
        if self._as_json:
            payload = {"level": logging.getLevelName(level), "event": event, **fields}
            self._logger.log(level, json.dumps(payload, ensure_ascii=False))
        else:
            self._logger.log(level, self._render_text(event, fields))

    def info(self, event: str, **fields: Any) -> None:
        """Émet un événement de niveau INFO."""
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Émet un événement de niveau WARNING."""
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        """Émet un événement de niveau ERROR."""
        self._emit(logging.ERROR, event, fields)


# ---------------------------------------------------------------------------
# Accès aux fichiers
# ---------------------------------------------------------------------------

def locate_corpus(filename: str) -> Path | None:
    """Localise un fichier sous corpus/ (ancrages script dir / CWD), sinon None."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "corpus" / filename,
        Path.cwd() / "corpus" / filename,
        script_dir / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def sha256_of(path: Path, chunk_size: int = 65_536) -> str:
    """Calcule l'empreinte SHA-256 hexadécimale d'un fichier (traçabilité)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Configuration du tokenizer (sérialisée dans l'artefact BM25)."""

    use_stopwords: bool = True
    pattern: str = TOKEN_PATTERN.pattern

    def tokenize(self, text: str) -> list[str]:
        """Tokenise un texte : minuscules -> alphanumérique -> sans mots vides.

        Cette fonction DOIT être utilisée à l'identique au moment de la
        recherche : la config est embarquée dans bm25_index.pkl pour l'imposer.
        """
        tokens = TOKEN_PATTERN.findall(text.lower())
        if self.use_stopwords:
            return [token for token in tokens if token not in ENGLISH_STOPWORDS]
        return tokens

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable pour l'artefact et le rapport."""
        return {
            "use_stopwords": self.use_stopwords,
            "stopwords_count": len(ENGLISH_STOPWORDS) if self.use_stopwords else 0,
            "pattern": self.pattern,
        }


def load_chunks(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Charge chunks.json avec garde-fous d'intégrité minimale.

    Garantit : champs obligatoires présents, chunk_id uniques, ordre trié par
    chunk_id (=> ordre documentaire de l'index BM25 reproductible).

    Returns:
        Tuple (chunks triés, métadonnées du payload source).
    """
    try:
        with input_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON invalide dans {input_path.name} : {exc}") from exc
    if not isinstance(payload.get("chunks"), list) or not payload["chunks"]:
        raise SystemExit(
            f"Format inattendu dans {input_path.name} : "
            "clé 'chunks' (liste non vide) requise."
        )
    chunks: list[dict[str, Any]] = payload["chunks"]

    incomplete = [
        str(c.get("chunk_id", "<sans id>")) for c in chunks
        if any(name not in c or c[name] is None for name in REQUIRED_CHUNK_FIELDS)
    ]
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} chunk(s) avec champ(s) obligatoire(s) manquant(s), "
            f"exemples : {incomplete[:5]}. Lancez validate_chunks.py."
        )
    seen: dict[str, int] = {}
    for chunk in chunks:
        seen[chunk["chunk_id"]] = seen.get(chunk["chunk_id"], 0) + 1
    duplicates = [chunk_id for chunk_id, count in seen.items() if count > 1]
    if duplicates:
        raise SystemExit(
            f"{len(duplicates)} chunk_id dupliqué(s), exemples : "
            f"{sorted(duplicates)[:5]}. Lancez validate_chunks.py."
        )
    return sorted(chunks, key=lambda c: str(c["chunk_id"])), payload


# ---------------------------------------------------------------------------
# Statistiques lexicales
# ---------------------------------------------------------------------------

def lexical_statistics(
    tokenized_corpus: list[list[str]],
) -> dict[str, Any]:
    """Statistiques du corpus tokenisé (volumes, longueurs, termes dominants).

    Args:
        tokenized_corpus: liste des listes de jetons, une par chunk.
    """
    term_frequencies: dict[str, int] = {}
    empty_docs = 0
    total_tokens = 0
    for tokens in tokenized_corpus:
        total_tokens += len(tokens)
        if not tokens:
            empty_docs += 1
        for token in tokens:
            term_frequencies[token] = term_frequencies.get(token, 0) + 1

    doc_lengths = sorted(len(tokens) for tokens in tokenized_corpus)
    count = len(doc_lengths)
    top_terms = sorted(
        term_frequencies.items(), key=lambda item: (-item[1], item[0])
    )[:TOP_TERMS_LIMIT]
    return {
        "documents": count,
        "total_tokens": total_tokens,
        "vocabulary_size": len(term_frequencies),
        "mean_tokens_per_chunk": round(total_tokens / count, 1) if count else 0.0,
        "min_tokens_per_chunk": doc_lengths[0] if doc_lengths else 0,
        "max_tokens_per_chunk": doc_lengths[-1] if doc_lengths else 0,
        "empty_token_documents": empty_docs,
        "top_terms": [{"term": term, "count": freq} for term, freq in top_terms],
    }


# ---------------------------------------------------------------------------
# Construction / persistance / vérification de l'index
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BM25Params:
    """Paramètres du ranking BM25 (tracés dans l'artefact et le rapport)."""

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B


def build_bm25_index(
    tokenized_corpus: list[list[str]], params: BM25Params
) -> BM25Okapi:
    """Construit l'index BM25Okapi sur le corpus tokenisé."""
    return BM25Okapi(tokenized_corpus, k1=params.k1, b=params.b)


def save_index(
    index_path: Path,
    bm25: BM25Okapi,
    chunk_ids: list[str],
    tokenizer_config: TokenizerConfig,
    params: BM25Params,
) -> dict[str, Any]:
    """Persiste l'index et les métadonnées critiques au format pickle.

    L'artefact embarque volontairement la configuration du tokenizer : la
    recherche doit tokeniser sa requête à l'identique pour des scores valides.

    Returns:
        Métadonnées du fichier écrit (taille, sha256).
    """
    artifact = {
        "format_version": INDEX_FORMAT_VERSION,
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "tokenizer": {"use_stopwords": tokenizer_config.use_stopwords},
        "params": {"k1": params.k1, "b": params.b},
    }
    with index_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return {
        "file": index_path.name,
        "size_bytes": index_path.stat().st_size,
        "sha256": sha256_of(index_path),
    }


def verify_index(
    index_path: Path,
    expected_chunk_ids: list[str],
    tokenizer_config: TokenizerConfig,
) -> tuple[bool, dict[str, Any]]:
    """Recharge l'artefact depuis le disque et vérifie qu'il est exploitable.

    Contrôles : format_version, type de l'objet BM25, ordre exact des
    chunk_ids, puis requête témoin (« retrieval augmented generation ») dont
    les scores doivent être finis et mappés vers des chunks connus.

    Returns:
        Tuple (succès global, détails de vérification).
    """
    import math
    import pickle

    with index_path.open("rb") as handle:
        artifact = pickle.load(handle)

    checks: dict[str, Any] = {}
    bm25_object = artifact.get("bm25")
    checks["format_version_ok"] = (
        artifact.get("format_version") == INDEX_FORMAT_VERSION
    )
    checks["bm25_type_ok"] = isinstance(bm25_object, BM25Okapi)
    stored_ids = artifact.get("chunk_ids", [])
    checks["chunk_ids_match"] = stored_ids == expected_chunk_ids
    checks["tokenizer_config_stored"] = isinstance(
        artifact.get("tokenizer"), dict
    ) and set(artifact["tokenizer"]) >= {"use_stopwords"}

    sample_query = "retrieval augmented generation"
    query_tokens = tokenizer_config.tokenize(sample_query)
    sample_results: list[dict[str, Any]] = []
    if checks["bm25_type_ok"]:
        scores = bm25_object.get_scores(query_tokens)
        checks["scores_length_ok"] = len(scores) == len(expected_chunk_ids)
        checks["scores_finite"] = all(math.isfinite(float(score)) for score in scores)
        top_indices = sorted(
            range(len(scores)), key=lambda position: -scores[position]
        )[:EXAMPLE_SAMPLE_SIZE]
        sample_results = [
            {
                "rank": rank + 1,
                "chunk_id": expected_chunk_ids[position],
                "score": round(float(scores[position]), 4),
            }
            for rank, position in enumerate(top_indices)
        ]
    else:
        checks["scores_length_ok"] = False
        checks["scores_finite"] = False

    success = all(bool(value) for value in checks.values())
    return success, {
        "sample_query": sample_query,
        "query_tokens": query_tokens,
        "checks": checks,
        "sample_query_top_results": sample_results,
    }


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def build_report(
    input_path: Path,
    source_payload: dict[str, Any],
    chunk_count: int,
    tokenizer_config: TokenizerConfig,
    params: BM25Params,
    lexical_stats: dict[str, Any],
    artifact_info: dict[str, Any],
    verification_success: bool,
    verification_details: dict[str, Any],
    timings_seconds: dict[str, float],
) -> dict[str, Any]:
    """Assemble le rapport de construction BM25 (sérialisable en JSON)."""
    from importlib.metadata import version

    empty_docs = int(lexical_stats["empty_token_documents"])
    status = (
        Status.FAIL.value if not verification_success
        else Status.WARNING.value if empty_docs > 0
        else Status.PASS.value
    )
    return {
        "generator": "build_bm25_index.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "input": {
            "file": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_of(input_path),
            "chunks": chunk_count,
            "generator": source_payload.get("generator"),
            "generated_at": source_payload.get("generated_at"),
        },
        "tokenizer": tokenizer_config.to_dict(),
        "params": {"variant": "BM25Okapi", "k1": params.k1, "b": params.b},
        "rank_bm25_version": version("rank-bm25"),
        "lexical_statistics": lexical_stats,
        "output": artifact_info,
        "verification": {
            "success": verification_success,
            **verification_details,
        },
        "durations_seconds": {
            key: round(value, 2) for key, value in timings_seconds.items()
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_report(corpus_dir: Path, report: dict[str, Any]) -> Path:
    """Écrit bm25_report.json sous corpus/ et retourne son chemin."""
    report_path = corpus_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report_path


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du constructeur BM25."""
    parser = argparse.ArgumentParser(
        description=(
            "Construit l'index BM25 sur chunks.json et produit "
            "bm25_index.pkl + bm25_report.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Codes de sortie : 0 = index vérifié · 1 = échec\n"
            "Exemples :\n"
            "  python build_bm25_index.py\n"
            "  python build_bm25_index.py --k1 1.2 --b 0.75\n"
            "  python build_bm25_index.py --no-stopwords"
        ),
    )
    parser.add_argument("--input", default=INPUT_FILENAME,
                        help=f"Fichier d'entrée sous corpus/ ({INPUT_FILENAME}).")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1, metavar="X",
                        help=f"Paramètre k1 de saturation ({DEFAULT_K1}).")
    parser.add_argument("--b", type=float, default=DEFAULT_B, metavar="X",
                        help=f"Normalisation par longueur b ({DEFAULT_B}).")
    parser.add_argument("--no-stopwords", action="store_true",
                        help="Désactive la suppression des mots vides anglais.")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Format des logs console (défaut : text).")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : tokenisation -> index BM25 -> persistance -> rapport."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.k1 <= 0 or not 0 <= args.b <= 1:
        raise SystemExit("--k1 doit être > 0 et --b dans [0 ; 1].")

    input_path = locate_corpus(args.input)
    if input_path is None:
        raise SystemExit(
            f"{args.input} introuvable sous corpus/. Lancez chunk_documents.py."
        )

    events.info("build_started", file=input_path.name)

    chunks, source_payload = load_chunks(input_path)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    tokenizer_config = TokenizerConfig(use_stopwords=not args.no_stopwords)
    params = BM25Params(k1=args.k1, b=args.b)

    started = time.perf_counter()
    tokenized_corpus: list[list[str]] = []
    for chunk in tqdm(chunks, desc="tokenization", unit="chunk", ncols=80):
        tokenized_corpus.append(tokenizer_config.tokenize(str(chunk["text"])))
    tokenize_seconds = time.perf_counter() - started

    lexical_stats = lexical_statistics(tokenized_corpus)
    events.info(
        "corpus_tokenized",
        documents=lexical_stats["documents"],
        total_tokens=lexical_stats["total_tokens"],
        vocabulary_size=lexical_stats["vocabulary_size"],
        empty_token_documents=lexical_stats["empty_token_documents"],
    )

    started = time.perf_counter()
    bm25 = build_bm25_index(tokenized_corpus, params)
    build_seconds = time.perf_counter() - started
    events.info("index_built", variant="BM25Okapi", k1=params.k1, b=params.b)

    index_path = input_path.parent / INDEX_FILENAME
    started = time.perf_counter()
    artifact_info = save_index(index_path, bm25, chunk_ids, tokenizer_config, params)
    save_seconds = time.perf_counter() - started
    events.info("index_saved", **artifact_info)

    verification_success, verification_details = verify_index(
        index_path, chunk_ids, tokenizer_config,
    )
    top_result = verification_details["sample_query_top_results"][0]
    log_event = (
        events.info if verification_success else events.error
    )
    log_event(
        "verification_completed",
        success=verification_success,
        sample_query=verification_details["sample_query"],
        top_chunk_id=top_result["chunk_id"],
        top_score=top_result["score"],
    )

    report = build_report(
        input_path=input_path,
        source_payload=source_payload,
        chunk_count=len(chunks),
        tokenizer_config=tokenizer_config,
        params=params,
        lexical_stats=lexical_stats,
        artifact_info=artifact_info,
        verification_success=verification_success,
        verification_details=verification_details,
        timings_seconds={
            "tokenization": tokenize_seconds,
            "index_build": build_seconds,
            "index_save": save_seconds,
        },
    )
    report_path = write_report(input_path.parent, report)

    status = Status(report["status"])
    events.info("build_completed", status=status.value, report=str(report_path))
    if status is Status.WARNING:
        events.warning(
            "empty_token_documents_present",
            count=int(lexical_stats["empty_token_documents"]),
        )

    sys.exit(EXIT_OK if status is not Status.FAIL else EXIT_FAIL)


if __name__ == "__main__":
    main()