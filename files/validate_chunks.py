"""
Gate de validation qualité du corpus de chunks avant indexation (JOUR 3).

Input :
    corpus/chunks.json     (produit par chunk_documents.py)
    corpus/documents.json  (référence optionnelle : renforce le contrôle de
                           couverture documentaire si présent)

Objectif :
    Dernier contrôle qualité avant l'indexation (J4). Le script applique des
    vérifications à sévérité graduée puis rend un verdict global :

        PASS     aucune anomalie détectée
        WARNING  écarts qualité détectés (non bloquants par défaut)
        FAIL     au moins une erreur d'intégrité bloque l'indexation

Vérifications (sévérité par défaut) :
    1. unicité des chunk_id                                           -> FAIL
    2. champs obligatoires présents                                   -> FAIL
    3. aucun chunk au texte vide                                      -> FAIL
    4. chunks dupliqués (exacts / quasi-doublons Jaccard)             -> WARNING
    5. cohérence des document_id (format + thème homogène)            -> WARNING
    6. cohérence des pages (page_start <= page_end, pages >= 1)       -> FAIL
    7. longueur dans la plage configurable [min_tokens, max_tokens]   -> WARNING
    8. couverture documentaire complète (pages + référence documents) -> FAIL/WARNING

Sorties :
    corpus/chunk_validation_report.json  rapport machine traçable : verdict
    global, empreinte SHA-256 de l'entrée, seuils, résultat détaillé de chaque
    vérification, statistiques globales / par thème / par document,
    liste d'anomalies, environnement d'exécution.
    corpus/chunk_validation_report.md    rapport lisible (à citer dans le README).

Code de sortie :
    0   PASS (ou WARNING accepté)
    1   FAIL
    2   WARNING avec --fail-on-warning

Usage :
    python validate_chunks.py
    python validate_chunks.py --min-tokens 120 --max-tokens 512
    python validate_chunks.py --fail-on-warning --log-format json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
REFERENCE_FILENAME = "documents.json"
REPORT_JSON_FILENAME = "chunk_validation_report.json"
REPORT_MD_FILENAME = "chunk_validation_report.md"

DEFAULT_MIN_TOKENS = 100      # aligné sur MIN_CHUNK_TOKENS du chunker
DEFAULT_MAX_TOKENS = 512      # aligné sur BGE_MAX_TOKENS (borne dense J4/J6)
DEFAULT_COVERAGE_MIN_PCT = 99.0   # couverture pages/doc minimale (WARNING sinon)
NEAR_DUP_JACCARD = 0.60           # similarité => quasi-doublon de contenu
SHINGLE_WORDS = 8                 # taille des shingles (mots) pour la redondance
EXAMPLE_LIMIT = 10            # nb max d'exemples conservés par vérification

# Champs obligatoires d'un chunk (contrat d'interface avec chunk_documents.py)
REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id", "document_id", "theme",
    "page_start", "page_end", "tokens_est", "text",
)


class Status(StrEnum):
    """Verdict d'une vérification ou du rapport global."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


_SEVERITY_RANK: dict[Status, int] = {
    Status.PASS: 0,
    Status.WARNING: 1,
    Status.FAIL: 2,
}


def worst_status(statuses: list[Status]) -> Status:
    """Retourne la sévérité la plus haute parmi les statuts fournis."""
    if not statuses:
        return Status.PASS
    return max(statuses, key=lambda status: _SEVERITY_RANK[status])


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Seuils de validation (tracés tels quels dans le rapport)."""

    min_tokens: int = DEFAULT_MIN_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    near_dup_jaccard: float = NEAR_DUP_JACCARD


@dataclass(slots=True)
class CheckResult:
    """Résultat normalisé d'une vérification unitaire."""

    name: str
    status: Status
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    examples: list[Any] = field(default_factory=list)


class EventLogger:
    """Logger d'événements structurés (rendu texte clé=valeur ou JSON).

    Wrapper fin autour de `logging.Logger` : chaque événement porte un nom et
    des champs typés, ce qui rend les runs grepables (texte) ou exploitables
    par une stack de logs (JSON), sans dépendance externe.
    """

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

    def log(self, level: int, event: str, **fields: Any) -> None:
        """Émet un événement à un niveau de logging explicite."""
        self._emit(level, event, fields)


def build_logger(as_json: bool) -> tuple[logging.Logger, EventLogger]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("validate_chunks")
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
# Chargement
# ---------------------------------------------------------------------------

def load_payload(input_path: Path) -> dict[str, Any]:
    """Charge chunks.json et vérifie la structure minimale du payload.

    Raises:
        SystemExit: si le JSON est illisible ou si `chunks` est absent/vide.
    """
    try:
        with input_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON invalide dans {input_path.name} : {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise SystemExit(
            f"Format inattendu dans {input_path.name} : clé 'chunks' (liste) absente."
        )
    if not payload["chunks"]:
        raise SystemExit(f"{input_path.name} ne contient aucun chunk à valider.")
    return payload


def load_reference() -> dict[str, int] | None:
    """Charge documents.json (référence de couverture), ou None s'il est absent.

    Returns:
        Mapping `{document_id: page_count}` trié par id, ou None si la
        référence n'est pas disponible ou illisible.
    """
    reference_path = locate_corpus(REFERENCE_FILENAME)
    if reference_path is None:
        return None
    try:
        with reference_path.open(encoding="utf-8") as handle:
            documents = json.load(handle)["documents"]
    except (json.JSONDecodeError, KeyError):
        return None
    return {
        document["id"]: document.get("page_count", 0)
        for document in sorted(documents, key=lambda item: item["id"])
    }


# ---------------------------------------------------------------------------
# Vérifications unitaires
# ---------------------------------------------------------------------------

def check_required_fields(chunks: list[dict[str, Any]]) -> CheckResult:
    """Vérifie la présence et le type attendu des champs obligatoires."""
    missing_counts: dict[tuple[str, ...], int] = defaultdict(int)
    offending_ids: list[str] = []
    for chunk in chunks:
        missing = [
            name for name in REQUIRED_CHUNK_FIELDS
            if name not in chunk or chunk[name] is None
        ]
        if missing:
            missing_counts[tuple(missing)] += 1
            if len(offending_ids) < EXAMPLE_LIMIT:
                offending_ids.append(str(chunk.get("chunk_id", "<sans id>")))
    total_missing = sum(missing_counts.values())
    status = Status.FAIL if total_missing else Status.PASS
    summary = (
        f"{total_missing} chunk(s) avec champ(s) obligatoire(s) manquant(s)"
        if total_missing else "tous les champs obligatoires sont présents"
    )
    return CheckResult(
        name="required_fields",
        status=status,
        summary=summary,
        metrics={
            "chunks_with_missing_fields": total_missing,
            "missing_field_patterns": {
                "+".join(pattern): count
                for pattern, count in sorted(missing_counts.items())
            },
        },
        examples=[{"chunk_id": chunk_id} for chunk_id in offending_ids],
    )


def check_unique_ids(chunks: list[dict[str, Any]]) -> CheckResult:
    """Vérifie l'unicité globale des chunk_id (pré-requis upsert Qdrant)."""
    seen: dict[str, int] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        seen[chunk_id] = seen.get(chunk_id, 0) + 1
    duplicates = [
        {"chunk_id": chunk_id, "occurrences": count}
        for chunk_id, count in sorted(seen.items()) if count > 1
    ]
    status = Status.FAIL if duplicates else Status.PASS
    summary = (
        f"{len(duplicates)} chunk_id dupliqué(s)"
        if duplicates else f"{len(seen)} chunk_id uniques"
    )
    return CheckResult(
        name="chunk_id_unique",
        status=status,
        summary=summary,
        metrics={
            "total_chunks": len(chunks),
            "unique_chunk_ids": len(seen),
            "duplicate_ids": len(duplicates),
        },
        examples=duplicates[:EXAMPLE_LIMIT],
    )


# ---------------------------------------------------------------------------
# Similarité de contenu (doublons)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalise un texte pour comparaison (casse, ponctuation, espaces)."""
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def word_shingles(text: str, width: int = SHINGLE_WORDS) -> frozenset[str]:
    """Construit l'ensemble des shingles de `width` mots d'un texte normalisé."""
    words = normalize_text(text).split()
    if not words:
        return frozenset()
    if len(words) < width:
        return frozenset({" ".join(words)})
    return frozenset(
        " ".join(words[i:i + width]) for i in range(len(words) - width + 1)
    )


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Coefficient de Jaccard |A∩B| / |A∪B| ; 0.0 si les deux ensembles sont vides."""
    if not first and not second:
        return 0.0
    union_size = len(first | second)
    return len(first & second) / union_size if union_size else 0.0


def check_empty_text(chunks: list[dict[str, Any]]) -> CheckResult:
    """Détecte les chunks au texte vide ou sans contenu exploitable."""
    empty_ids = [
        str(chunk.get("chunk_id", "<sans id>"))
        for chunk in chunks
        if not normalize_text(str(chunk.get("text", "")))
    ]
    status = Status.FAIL if empty_ids else Status.PASS
    summary = (
        f"{len(empty_ids)} chunk(s) au texte vide"
        if empty_ids else "aucun chunk au texte vide"
    )
    return CheckResult(
        name="empty_text",
        status=status,
        summary=summary,
        metrics={"empty_text_chunks": len(empty_ids)},
        examples=[{"chunk_id": chunk_id} for chunk_id in empty_ids[:EXAMPLE_LIMIT]],
    )


def check_content_duplicates(
    chunks: list[dict[str, Any]], near_dup_threshold: float
) -> CheckResult:
    """Détecte les doublons exacts et les quasi-doublons de contenu.

    Un quasi-doublon est une paire de chunks du même document dont la similarité
    de Jaccard sur shingles de `SHINGLE_WORDS` mots atteint `near_dup_threshold`.
    Sévérité WARNING : la redondance dégrade la qualité du retrieval mais ne
    casse pas l'indexation.
    """
    by_normalized: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        by_normalized[normalize_text(str(chunk.get("text", "")))].append(
            str(chunk["chunk_id"])
        )
    exact_groups = [
        sorted(ids) for _, ids in sorted(by_normalized.items()) if len(ids) > 1
    ]

    shingles_by_id = {c["chunk_id"]: word_shingles(str(c.get("text", ""))) for c in chunks}
    ids_by_doc: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        ids_by_doc[str(chunk["document_id"])].append(chunk["chunk_id"])

    near_pairs: list[dict[str, Any]] = []
    for document_id in sorted(ids_by_doc):
        ids = sorted(ids_by_doc[document_id])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                first = shingles_by_id[ids[i]]
                second = shingles_by_id[ids[j]]
                if not first or not second:
                    continue
                score = jaccard(first, second)
                if score >= near_dup_threshold:
                    near_pairs.append({
                        "a": ids[i], "b": ids[j],
                        "document_id": document_id,
                        "jaccard": round(score, 3),
                    })
    near_pairs.sort(key=lambda pair: (-pair["jaccard"], pair["a"], pair["b"]))

    status = Status.WARNING if exact_groups or near_pairs else Status.PASS
    summary_parts = [f"{len(exact_groups)} groupe(s) de doublons exacts"]
    if exact_groups:
        summary_parts.append(
            f"{sum(len(group) - 1 for group in exact_groups)} chunk(s) redondant(s)"
        )
    summary_parts.append(f"{len(near_pairs)} paire(s) quasi-doublon(s)")
    return CheckResult(
        name="content_duplicates",
        status=status,
        summary=" · ".join(summary_parts),
        metrics={
            "near_dup_jaccard_threshold": near_dup_threshold,
            "exact_duplicate_groups": len(exact_groups),
            "exact_redundant_chunks": sum(len(g) - 1 for g in exact_groups),
            "near_duplicate_pairs": len(near_pairs),
            "documents_with_near_duplicates": sorted(
                {pair["document_id"] for pair in near_pairs}
            ),
        },
        examples={
            "exact_groups": exact_groups[:5],
            "near_pairs": near_pairs[:EXAMPLE_LIMIT],
        },
    )


def check_document_consistency(
    chunks: list[dict[str, Any]], arxiv_id_pattern: re.Pattern[str]
) -> CheckResult:
    """Vérifie la cohérence des document_id : présence, format, thème homogène.

    Un même document doit porter un seul thème ; le format attendu est un
    identifiant arXiv avec version (ex. 2308.16118v2).
    """
    themes_by_doc: dict[str, set[str]] = defaultdict(set)
    invalid_format: list[dict[str, str]] = []
    empty_ids = 0
    for chunk in chunks:
        document_id = str(chunk.get("document_id", ""))
        if not document_id.strip():
            empty_ids += 1
            continue
        themes_by_doc[document_id].add(str(chunk.get("theme", "")))
        if not arxiv_id_pattern.match(document_id) and len(invalid_format) < EXAMPLE_LIMIT:
            invalid_format.append({
                "document_id": document_id,
                "chunk_id": str(chunk.get("chunk_id")),
            })
    mixed_theme_docs = {
        document_id: sorted(themes)
        for document_id, themes in sorted(themes_by_doc.items())
        if len(themes) > 1
    }
    problems = bool(empty_ids or invalid_format or mixed_theme_docs)
    return CheckResult(
        name="document_id_consistency",
        status=Status.WARNING if problems else Status.PASS,
        summary=(
            f"{len(mixed_theme_docs)} document(s) multi-thèmes · "
            f"{len(invalid_format)} id hors format · {empty_ids} id vide(s)"
            if problems else
            f"{len(themes_by_doc)} document_id cohérents (format + thème unique)"
        ),
        metrics={
            "documents": len(themes_by_doc),
            "empty_document_ids": empty_ids,
            "invalid_format_count": len(invalid_format),
            "mixed_theme_documents": mixed_theme_docs,
        },
        examples={"invalid_format": invalid_format},
    )


def check_page_coherence(chunks: list[dict[str, Any]]) -> CheckResult:
    """Vérifie que page_start <= page_end et que les pages sont >= 1."""
    violations: list[dict[str, Any]] = []
    for chunk in chunks:
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        coherent = (
            isinstance(page_start, int) and isinstance(page_end, int)
            and 1 <= page_start <= page_end
        )
        if not coherent and len(violations) < EXAMPLE_LIMIT:
            violations.append({
                "chunk_id": str(chunk.get("chunk_id")),
                "document_id": str(chunk.get("document_id")),
                "page_start": page_start,
                "page_end": page_end,
            })
    # Comptage complet (au-delà des exemples plafonnés).
    total = sum(
        1 for chunk in chunks
        if not (
            isinstance(chunk.get("page_start"), int)
            and isinstance(chunk.get("page_end"), int)
            and 1 <= chunk["page_start"] <= chunk["page_end"]
        )
    )
    status = Status.FAIL if total else Status.PASS
    summary = (
        f"{total} chunk(s) avec incohérence de pages"
        if total else f"pages cohérentes sur {len(chunks)} chunks"
    )
    return CheckResult(
        name="page_coherence",
        status=status,
        summary=summary,
        metrics={"incoherent_page_chunks": total},
        examples=violations,
    )


def check_length_range(
    chunks: list[dict[str, Any]], thresholds: Thresholds
) -> CheckResult:
    """Vérifie les longueurs : plage [min_tokens, max_tokens] et estimations > 0."""
    below_min = [
        {
            "chunk_id": str(c["chunk_id"]),
            "document_id": str(c["document_id"]),
            "tokens_est": c.get("tokens_est", 0),
        }
        for c in sorted(chunks, key=lambda item: item.get("tokens_est", 0))
        if c.get("tokens_est", 0) < thresholds.min_tokens
    ]
    above_max = [
        {
            "chunk_id": str(c["chunk_id"]),
            "document_id": str(c["document_id"]),
            "tokens_est": c.get("tokens_est", 0),
        }
        for c in sorted(
            chunks, key=lambda item: item.get("tokens_est", 0), reverse=True
        )
        if c.get("tokens_est", 0) > thresholds.max_tokens
    ]
    non_positive = [
        str(c["chunk_id"]) for c in chunks if c.get("tokens_est", 0) <= 0
    ]
    problems = bool(below_min or above_max or non_positive)
    return CheckResult(
        name="length_range",
        status=Status.WARNING if problems else Status.PASS,
        summary=(
            f"{len(below_min)} sous {thresholds.min_tokens} t · "
            f"{len(above_max)} au-dessus de {thresholds.max_tokens} t · "
            f"{len(non_positive)} estimation(s) <= 0"
        ),
        metrics={
            "min_tokens": thresholds.min_tokens,
            "max_tokens": thresholds.max_tokens,
            "below_min_count": len(below_min),
            "above_max_count": len(above_max),
            "non_positive_estimates": len(non_positive),
        },
        examples={
            "below_min": below_min[:EXAMPLE_LIMIT],
            "above_max": above_max[:EXAMPLE_LIMIT],
        },
    )


def percentile(sorted_values: list[int], quantile: float) -> int:
    """Percentile par rang (sans interpolation) sur une liste triée croissante."""
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(quantile * len(sorted_values)))
    return sorted_values[index]


def merge_page_intervals(
    chunks: list[dict[str, Any]],
) -> tuple[int, int]:
    """Fusionne les intervalles de pages d'un document.

    Returns:
        Tuple `(pages_couvertes, dernière_page)` — l'overlap entre chunks ne
        compte qu'une fois grâce à la fusion des intervalles.
    """
    intervals = sorted(
        (int(c["page_start"]), int(c["page_end"])) for c in chunks
    )
    merged_end = 0
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals:
        if start <= current_end + 1:          # contigu ou chevauchant
            current_end = max(current_end, end)
        else:                                  # trou de couverture
            covered += current_end - current_start + 1
            current_start, current_end = start, end
        merged_end = max(merged_end, end)
    covered += current_end - current_start + 1
    return covered, merged_end


def check_document_coverage(
    chunks_by_document: dict[str, list[dict[str, Any]]],
    reference: dict[str, int] | None,
    coverage_min_pct: float,
) -> CheckResult:
    """Vérifie la couverture documentaire complète.

    Contrôles effectués :
      - documents présents dans chunks.json mais absents de la référence
        (`documents.json`) -> FAIL si la référence est disponible ;
      - documents de la référence sans aucun chunk -> FAIL ;
      - trous de couverture au sein d'un document (pages manquantes entre la
        première et la dernière page chunkée) -> WARNING ;
      - couverture pages/doc sous `coverage_min_pct` -> WARNING ;
      - `last_page` != `page_count` de la référence (pages vides en fin de PDF,
        non bloquant) -> WARNING.
    """
    per_document: dict[str, dict[str, Any]] = {}
    holes_total = 0
    low_coverage_docs: list[dict[str, Any]] = []
    page_count_mismatches: list[dict[str, Any]] = []

    for document_id in sorted(chunks_by_document):
        doc_chunks = chunks_by_document[document_id]
        covered_pages, last_page = merge_page_intervals(doc_chunks)
        first_page = min(int(c["page_start"]) for c in doc_chunks)
        span_pages = last_page - first_page + 1
        hole_pages = max(0, span_pages - covered_pages)
        holes_total += hole_pages
        coverage_pct = round(100 * covered_pages / max(1, last_page), 1)
        entry = {
            "theme": str(doc_chunks[0].get("theme", "")),
            "chunks": len(doc_chunks),
            "first_page": first_page,
            "last_page": last_page,
            "pages_covered": covered_pages,
            "hole_pages": hole_pages,
            "coverage_pct": coverage_pct,
        }
        if reference is not None and document_id in reference:
            expected = reference[document_id]
            entry["reference_page_count"] = expected
            if last_page != expected:
                page_count_mismatches.append({
                    "document_id": document_id,
                    "last_chunked_page": last_page,
                    "reference_page_count": expected,
                })
        per_document[document_id] = entry
        if coverage_pct < coverage_min_pct or hole_pages > 0:
            low_coverage_docs.append({
                "document_id": document_id,
                "coverage_pct": coverage_pct,
                "hole_pages": hole_pages,
            })

    missing_in_chunks: list[str] = []
    unexpected_documents: list[str] = []
    if reference is not None:
        missing_in_chunks = sorted(set(reference) - set(per_document))
        unexpected_documents = sorted(set(per_document) - set(reference))

    has_integrity_issue = bool(missing_in_chunks or not chunks_by_document)
    has_quality_issues = bool(
        holes_total or low_coverage_docs or page_count_mismatches
        or unexpected_documents
    )
    status = (
        Status.FAIL if has_integrity_issue
        else Status.WARNING if has_quality_issues
        else Status.PASS
    )
    summary_parts = [f"{len(per_document)} document(s)"]
    if reference is not None:
        summary_parts.append(f"référence: {len(reference)} document(s)")
        summary_parts.append(f"{len(missing_in_chunks)} manquant(s)")
    summary_parts.append(f"{holes_total} page(s) manquante(s) (trous)")
    return CheckResult(
        name="document_coverage",
        status=status,
        summary=" · ".join(summary_parts),
        metrics={
            "documents": len(per_document),
            "reference_used": reference is not None,
            "missing_documents_vs_reference": missing_in_chunks,
            "unexpected_documents_vs_reference": unexpected_documents,
            "total_hole_pages": holes_total,
            "coverage_min_pct": coverage_min_pct,
            "low_coverage_documents": sorted(
                low_coverage_docs, key=lambda item: item["coverage_pct"]
            )[:EXAMPLE_LIMIT],
            "page_count_mismatches": page_count_mismatches[:EXAMPLE_LIMIT],
            "overall_coverage_pct": round(
                100
                * sum(entry["pages_covered"] for entry in per_document.values())
                / max(1, sum(max(1, entry["last_page"]) for entry in per_document.values())),
                1,
            ),
        },
        examples={"per_document": per_document},
    )


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def compute_statistics(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcule les statistiques globales, par thème et par document.

    Toutes les itérations sont triées pour garantir un rapport reproductible.
    """
    tokens = sorted(int(c.get("tokens_est", 0)) for c in chunks)
    spans = [
        int(c["page_end"]) - int(c["page_start"]) + 1 for c in chunks
    ]
    multi_page = sum(1 for span in spans if span > 1)
    global_stats: dict[str, Any] = {
        "chunks": len(chunks),
        "documents": len({str(c["document_id"]) for c in chunks}),
        "themes": len({str(c.get("theme", "")) for c in chunks}),
        "tokens_est_total": sum(tokens),
        "tokens_est_min": tokens[0] if tokens else 0,
        "tokens_est_max": tokens[-1] if tokens else 0,
        "tokens_est_mean": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "tokens_est_median": percentile(tokens, 0.50),
        "tokens_est_p05": percentile(tokens, 0.05),
        "tokens_est_p25": percentile(tokens, 0.25),
        "tokens_est_p75": percentile(tokens, 0.75),
        "tokens_est_p90": percentile(tokens, 0.90),
        "tokens_est_p95": percentile(tokens, 0.95),
        "pages_span_mean": round(sum(spans) / len(spans), 2) if spans else 0.0,
        "multi_page_chunks": multi_page,
        "multi_page_pct": round(100 * multi_page / len(spans), 1) if spans else 0.0,
    }

    themes_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"documents": set(), "chunks": 0, "tokens_est": 0}
    )
    documents_acc: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        theme = str(chunk.get("theme", ""))
        document_id = str(chunk["document_id"])
        token_count = int(chunk.get("tokens_est", 0))

        theme_entry = themes_acc[theme]
        theme_entry["documents"].add(document_id)
        theme_entry["chunks"] += 1
        theme_entry["tokens_est"] += token_count

        doc_entry = documents_acc.setdefault(document_id, {
            "theme": theme, "chunks": 0, "tokens_est": 0,
            "first_page": int(chunk["page_start"]),
            "last_page": int(chunk["page_end"]),
        })
        doc_entry["chunks"] += 1
        doc_entry["tokens_est"] += token_count
        doc_entry["first_page"] = min(doc_entry["first_page"], int(chunk["page_start"]))
        doc_entry["last_page"] = max(doc_entry["last_page"], int(chunk["page_end"]))

    per_theme = {
        theme: {
            "documents": len(entry["documents"]),
            "chunks": entry["chunks"],
            "tokens_est": entry["tokens_est"],
            "mean_tokens_per_chunk": round(
                entry["tokens_est"] / entry["chunks"], 1
            ) if entry["chunks"] else 0.0,
        }
        for theme, entry in sorted(themes_acc.items())
    }
    per_document = {
        document_id: {
            **entry,
            "mean_tokens_per_chunk": round(
                entry["tokens_est"] / entry["chunks"], 1
            ) if entry["chunks"] else 0.0,
        }
        for document_id, entry in sorted(documents_acc.items())
    }
    return {
        "global": global_stats,
        "per_theme": per_theme,
        "per_document": per_document,
    }


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def build_report(
    payload: dict[str, Any],
    input_path: Path,
    reference: dict[str, int] | None,
    thresholds: Thresholds,
    coverage_min_pct: float,
    checks: list[CheckResult],
) -> dict[str, Any]:
    """Assemble le rapport de validation complet (sérialisable en JSON)."""
    return {
        "generator": "validate_chunks.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": worst_status([check.status for check in checks]).value,
        "input": {
            "file": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_of(input_path),
            "generator": payload.get("generator"),
            "generated_at": payload.get("generated_at"),
            "config": payload.get("config"),
        },
        "reference": {
            "used": reference is not None,
            "file": REFERENCE_FILENAME if reference is not None else None,
            "documents": len(reference) if reference is not None else 0,
        },
        "thresholds": asdict(thresholds) | {"coverage_min_pct": coverage_min_pct},
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "checks": [asdict(check) for check in checks],
        "anomalies": [
            {
                "check": check.name,
                "severity": check.status.value,
                "message": check.summary,
            }
            for check in checks if check.status is not Status.PASS
        ],
        "statistics": compute_statistics(payload["chunks"]),
    }


# ---------------------------------------------------------------------------
# Rendu Markdown
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Construit une table Markdown simple à partir d'en-têtes et de lignes."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _format_config(config: dict[str, Any] | None) -> str:
    """Formate la config du chunking pour une ligne Markdown compacte."""
    if not config:
        return "non renseignée"
    return ", ".join(f"{key}={value}" for key, value in config.items())


_STATUS_BADGE: dict[Status, str] = {
    Status.PASS: "✅ PASS",
    Status.WARNING: "⚠️ WARNING",
    Status.FAIL: "❌ FAIL",
}


def render_markdown(report: dict[str, Any]) -> str:
    """Rend le rapport de validation au format Markdown lisible (README)."""
    status = Status(report["status"])
    global_stats = report["statistics"]["global"]
    lines: list[str] = []

    lines.append(f"# Validation des chunks (JOUR 3) — {status.value}")
    lines.append("")
    lines.append(f"- Généré le : `{report['generated_at']}`")
    lines.append(
        f"- Source : `{report['input']['file']}` "
        f"(sha256 `{report['input']['sha256'][:16]}…`, "
        f"{report['input']['size_bytes']:,} octets)"
    )
    lines.append(
        f"- Chunking : `{report['input'].get('generator')}` — config : "
        f"{_format_config(report['input'].get('config'))}"
    )
    reference = report["reference"]
    if reference["used"]:
        lines.append(
            f"- Référence couverture : `{reference['file']}` "
            f"({reference['documents']} documents)"
        )
    lines.append(f"- Seuils : {_format_config(report['thresholds'])}")
    lines.append("")

    lines.append("## Résultat des vérifications")
    lines.append("")
    lines.append(_md_table(
        ["Vérification", "Statut", "Détail"],
        [
            [f"`{check['name']}`", _STATUS_BADGE[Status(check["status"])],
             check["summary"]]
            for check in report["checks"]
        ],
    ))

    if report["anomalies"]:
        lines.append("")
        lines.append("## Anomalies détectées")
        lines.append("")
        lines.append(_md_table(
            ["Vérification", "Sévérité", "Message"],
            [
                [f"`{item['check']}`", item["severity"], item["message"]]
                for item in report["anomalies"]
            ],
        ))

    lines.append("")
    lines.append("## Statistiques globales")
    lines.append("")
    lines.append(_md_table(
        ["Indicateur", "Valeur"],
        [[key, value] for key, value in global_stats.items()],
    ))

    theme_rows = [
        [theme, entry["documents"], entry["chunks"],
         f"{entry['tokens_est']:,}", entry["mean_tokens_per_chunk"]]
        for theme, entry in report["statistics"]["per_theme"].items()
    ]
    lines.append("")
    lines.append("## Statistiques par thème")
    lines.append("")
    lines.append(_md_table(
        ["Thème", "Documents", "Chunks", "Tokens est.", "Tokens/chunk"], theme_rows,
    ))

    doc_rows = [
        [document_id, entry["theme"], entry["chunks"],
         entry["mean_tokens_per_chunk"],
         f"{entry['first_page']}–{entry['last_page']}"]
        for document_id, entry in report["statistics"]["per_document"].items()
    ]
    lines.append("")
    lines.append("## Statistiques par document")
    lines.append("")
    lines.append(_md_table(
        ["Document", "Thème", "Chunks", "Tokens/chunk", "Pages"], doc_rows,
    ))
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    corpus_dir: Path, report: dict[str, Any], markdown: str
) -> tuple[Path, Path]:
    """Écrit le rapport JSON et le rapport Markdown sous corpus/.

    Returns:
        Tuple des deux chemins écrits (json, markdown).
    """
    json_path = corpus_dir / REPORT_JSON_FILENAME
    md_path = corpus_dir / REPORT_MD_FILENAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_WARNING_ESCALATED = 2


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du validateur."""
    parser = argparse.ArgumentParser(
        description=(
            "Gate qualité du corpus de chunks : verdict PASS / WARNING / FAIL "
            "+ rapports traçables (chunk_validation_report.json/.md)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Codes de sortie : 0 = PASS/WARNING · 1 = FAIL · "
            "2 = WARNING avec --fail-on-warning\n"
            "Exemples :\n"
            "  python validate_chunks.py\n"
            "  python validate_chunks.py --min-tokens 120 --max-tokens 512\n"
            "  python validate_chunks.py --fail-on-warning --log-format json"
        ),
    )
    parser.add_argument("--input", default=INPUT_FILENAME,
                        help=f"Fichier d'entrée sous corpus/ ({INPUT_FILENAME}).")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                        metavar="N",
                        help=f"Longueur minimale d'un chunk ({DEFAULT_MIN_TOKENS}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        metavar="N",
                        help=f"Longueur maximale d'un chunk ({DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--near-dup-jaccard", type=float, default=NEAR_DUP_JACCARD,
                        metavar="X",
                        help=(f"Similarité Jaccard => quasi-doublon "
                              f"({NEAR_DUP_JACCARD})."))
    parser.add_argument("--coverage-min-pct", type=float,
                        default=DEFAULT_COVERAGE_MIN_PCT, metavar="X",
                        help=(f"Couverture pages/doc minimale en %% "
                              f"({DEFAULT_COVERAGE_MIN_PCT})."))
    parser.add_argument("--fail-on-warning", action="store_true",
                        help="Considère un statut WARNING comme un échec (CI).")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Format des logs console (défaut : text).")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : exécute les vérifications puis écrit les rapports."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.min_tokens <= 0 or args.max_tokens <= 0:
        raise SystemExit("--min-tokens et --max-tokens doivent être > 0.")
    if args.min_tokens > args.max_tokens:
        raise SystemExit("--min-tokens doit être <= --max-tokens.")
    if not 0.0 < args.near_dup_jaccard <= 1.0:
        raise SystemExit("--near-dup-jaccard doit être dans ]0 ; 1].")

    input_path = locate_corpus(args.input)
    if input_path is None:
        raise SystemExit(
            f"{args.input} introuvable sous corpus/. Lancez chunk_documents.py."
        )
    payload = load_payload(input_path)
    chunks: list[dict[str, Any]] = payload["chunks"]
    reference = load_reference()
    thresholds = Thresholds(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        near_dup_jaccard=args.near_dup_jaccard,
    )

    events.info("validation_started", file=input_path.name, chunks=len(chunks))

    arxiv_id_pattern = re.compile(r"^\d{4}\.\d{4,5}v\d+$")
    chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[str(chunk["document_id"])].append(chunk)

    checks: list[CheckResult] = [
        check_unique_ids(chunks),
        check_required_fields(chunks),
        check_empty_text(chunks),
        check_content_duplicates(chunks, thresholds.near_dup_jaccard),
        check_document_consistency(chunks, arxiv_id_pattern),
        check_page_coherence(chunks),
        check_length_range(chunks, thresholds),
        check_document_coverage(chunks_by_document, reference, args.coverage_min_pct),
    ]

    report = build_report(
        payload=payload,
        input_path=input_path,
        reference=reference,
        thresholds=thresholds,
        coverage_min_pct=args.coverage_min_pct,
        checks=checks,
    )
    json_path, md_path = write_outputs(
        input_path.parent, report, render_markdown(report),
    )

    for check in checks:
        level = (
            logging.INFO if check.status is Status.PASS
            else logging.WARNING if check.status is Status.WARNING
            else logging.ERROR
        )
        events.log(level, f"check_{check.name}",
                   status=check.status.value, detail=check.summary)

    status = Status(report["status"])
    events.info(
        "validation_completed",
        status=status.value,
        checks_total=len(checks),
        warnings=sum(1 for c in checks if c.status is Status.WARNING),
        failures=sum(1 for c in checks if c.status is Status.FAIL),
        report_json=str(json_path),
        report_md=str(md_path),
    )

    if status is Status.FAIL:
        sys.exit(EXIT_FAIL)
    if status is Status.WARNING and args.fail_on_warning:
        sys.exit(EXIT_WARNING_ESCALATED)
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()