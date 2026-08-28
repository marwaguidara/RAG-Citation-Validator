"""
Évaluation de la qualité du corpus de chunks (JOUR 3, étape C).

Input :
    corpus/chunks.json    (produit par chunk_documents.py)

Objectif :
    Produire un rapport qualité objectif et reproductible du corpus de chunks,
    en amont de l'indexation (J4+). Le rapport mesure :
      - volumes : nombre total de chunks, par thème, par document ;
      - tailles : moyenne, médiane, percentiles, histogramme ;
      - granularité : distribution du nombre de pages par chunk ;
      - anomalies : chunks trop petits / trop grands ;
      - redondance : doublons exacts et quasi-doublons (shingles 8 mots,
        similarité de Jaccard intra-document) ;
      - couverture : pages réellement couvertes par document / thème.

Sorties (écrites sous corpus/) :
    chunk_stats.json    rapport machine (artefact de suivi J4→J7)
    chunk_stats.md      rapport lisible

Choix :
    - Aucune dépendance externe hors stdlib : reproductible et léger.
    - Seuils configurables en CLI, tracés dans le rapport (`thresholds`).
    - Sorties triées de façon déterministe (aucune itération sur un set nu).

Usage :
    python evaluate_chunks.py
    python evaluate_chunks.py --small-tokens 150 --large-tokens 700
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# Fichiers d'entrée / sortie (ancrés sous corpus/)
INPUT_FILENAME = "chunks.json"
OUTPUT_JSON_FILENAME = "chunk_stats.json"
OUTPUT_MD_FILENAME = "chunk_stats.md"

# Seuils qualité (surchargeables en CLI, tracés dans le rapport)
DEFAULT_SMALL_TOKENS = 150   # en dessous : chunk jugé trop petit
DEFAULT_LARGE_TOKENS = 700   # au-dessus : chunk jugé trop grand
DEFAULT_NEAR_DUP_JACCARD = 0.60  # similarité => quasi-doublon
SHINGLE_WORDS = 8            # taille des shingles (mots) pour la redondance
DOCS_MIN_CHUNKS_WARN = 5     # document avec moins de chunks => couverture faible

# Bornes de l'histogramme des tailles (en tokens, pas de 100)
SIZE_BIN_WIDTH = 100


def get_logger() -> logging.Logger:
    """Configure et retourne un logger console avec formatage constant."""
    logger = logging.getLogger("evaluate_chunks")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def locate_corpus(filename: str) -> Path:
    """Retourne le chemin du fichier corpus demandé via plusieurs ancrages.

    Cherche successivement `<script>/corpus/<filename>`, `<cwd>/corpus/<filename>`
    puis `<script>/<filename>` (même convention que chunk_documents.py).
    Lève SystemExit si introuvable.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "corpus" / filename,
        Path.cwd() / "corpus" / filename,
        script_dir / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"{filename} introuvable. Lancez d'abord chunk_documents.py "
        f"(produit corpus/{INPUT_FILENAME})."
    )


def load_chunks(path: Path) -> dict[str, Any]:
    """Charge chunks.json et retourne le payload complet (dict)."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "chunks" not in payload:
        raise SystemExit(f"Format inattendu dans {path.name} : clé 'chunks' absente.")
    return payload


def percentile(sorted_values: list[int], quantile: float) -> int:
    """Percentile par rang (sans interpolation) sur une liste déjà triée.

    Args:
        sorted_values: valeurs triées croissantes (non vide).
        quantile: quantile voulu dans [0.0, 1.0].
    """
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(quantile * len(sorted_values)))
    return sorted_values[index]


def normalize_text(text: str) -> str:
    """Normalise un texte pour comparaison (casse + ponctuation + espaces)."""
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def word_shingles(text: str, width: int = SHINGLE_WORDS) -> frozenset[str]:
    """Construit l'ensemble des shingles de `width` mots d'un texte normalisé."""
    words = normalize_text(text).split()
    if len(words) < width:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i:i + width]) for i in range(len(words) - width + 1))


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Coefficient de Jaccard |A∩B| / |A∪B| ; 0.0 si les deux sont vides."""
    if not first and not second:
        return 0.0
    union_size = len(first | second)
    return len(first & second) / union_size if union_size else 0.0


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def size_distribution(tokens: list[int]) -> dict[str, Any]:
    """Statistiques de taille des chunks (moyenne, médiane, percentiles, bins).

    Args:
        tokens: liste des `tokens_est` (un par chunk).
    """
    ordered = sorted(tokens)
    count = len(ordered)
    if count == 0:
        return {"count": 0}
    histogram: dict[str, int] = {}
    for value in ordered:
        low = (value // SIZE_BIN_WIDTH) * SIZE_BIN_WIDTH
        key = f"{low}-{low + SIZE_BIN_WIDTH - 1}"
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "count": count,
        "min": ordered[0],
        "p05": percentile(ordered, 0.05),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / count, 1),
        "histogram_100t_bins": dict(sorted(
            histogram.items(), key=lambda item: int(item[0].split("-")[0])
        )),
    }


def theme_breakdown(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Agrège les métriques par thème (documents, chunks, tokens)."""
    grouped: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        entry = grouped.setdefault(chunk.get("theme", "?"), {
            "documents": set(), "chunks": 0, "tokens_est": 0,
        })
        entry["documents"].add(chunk["document_id"])
        entry["chunks"] += 1
        entry["tokens_est"] += chunk.get("tokens_est", 0)
    result: dict[str, dict[str, Any]] = {}
    for theme in sorted(grouped):
        entry = grouped[theme]
        docs = len(entry["documents"])
        n_chunks = entry["chunks"]
        result[theme] = {
            "documents": docs,
            "chunks": n_chunks,
            "tokens_est": entry["tokens_est"],
            "mean_chunks_per_document": round(n_chunks / docs, 1) if docs else 0.0,
            "mean_tokens_per_chunk": (
                round(entry["tokens_est"] / n_chunks, 1) if n_chunks else 0.0
            ),
        }
    return result


def page_span_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution du nombre de pages couvertes par chunk."""
    spans = [max(1, c["page_end"] - c["page_start"] + 1) for c in chunks]
    counter = Counter(spans)
    multi_page = sum(1 for span in spans if span > 1)
    count = len(spans)
    return {
        "mean_span_pages": round(sum(spans) / count, 2) if count else 0.0,
        "max_span_pages": max(spans) if spans else 0,
        "multi_page_chunks": multi_page,
        "multi_page_pct": round(100 * multi_page / count, 1) if count else 0.0,
        "single_page_chunks": count - multi_page,
        "span_distribution": {str(s): counter[s] for s in sorted(counter)},
    }


def flag_outliers(
    chunks: list[dict[str, Any]],
    small_threshold: int,
    large_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Identifie les chunks trop petits (< seuil) et trop grands (> seuil).

    Returns:
        Tuple (petits triés croissant, grands triés décroissant), réduits aux
        champs utiles pour le rapport.
    """
    def describe(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "theme": chunk.get("theme", ""),
            "tokens_est": chunk.get("tokens_est", 0),
            "span_pages": max(1, chunk["page_end"] - chunk["page_start"] + 1),
            "preview": normalize_text(chunk["text"])[:80],
        }

    small = sorted(
        (describe(c) for c in chunks if c.get("tokens_est", 0) < small_threshold),
        key=lambda item: item["tokens_est"],
    )
    large = sorted(
        (describe(c) for c in chunks if c.get("tokens_est", 0) > large_threshold),
        key=lambda item: item["tokens_est"], reverse=True,
    )
    return small, large


def duplication_analysis(
    chunks: list[dict[str, Any]],
    near_dup_threshold: float,
) -> dict[str, Any]:
    """Détecte les doublons exacts et les quasi-doublons intra-document.

    Un quasi-doublon est une paire de chunks du même document dont la similarité
    de Jaccard sur shingles de `SHINGLE_WORDS` mots atteint `near_dup_threshold`.
    """
    by_normalized: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        by_normalized[normalize_text(chunk["text"])].append(chunk["chunk_id"])
    exact_groups = sorted(sorted(ids) for ids in by_normalized.values() if len(ids) > 1)
    exact_redundant = sum(len(group) - 1 for group in exact_groups)

    shingles_by_id = {c["chunk_id"]: word_shingles(c["text"]) for c in chunks}
    ids_by_doc: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        ids_by_doc[chunk["document_id"]].append(chunk["chunk_id"])

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
    involved = {pair[key] for pair in near_pairs for key in ("a", "b")}

    excessive = bool(exact_redundant) or len(near_pairs) > max(5, int(0.02 * len(chunks)))
    return {
        "near_dup_jaccard_threshold": near_dup_threshold,
        "shingle_words": SHINGLE_WORDS,
        "exact_duplicate_groups": len(exact_groups),
        "exact_redundant_chunks": exact_redundant,
        "exact_examples": exact_groups[:5],
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_chunks_involved": len(involved),
        "near_examples": near_pairs[:10],
        "excessive_duplication": excessive,
    }


def coverage_analysis(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Mesure la couverture documentaire : pages réellement couvertes par doc.

    Les intervalles de pages des chunks sont fusionnés (l'overlap ne compte donc
    qu'une fois). Le dénominateur est `last_page` du document (pages numérotées
    à partir de 1), seul disponible depuis chunks.json.
    """
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["document_id"]].append(chunk)

    per_document: dict[str, dict[str, Any]] = {}
    total_covered = 0
    total_pages = 0
    for document_id in sorted(chunks_by_doc):
        doc_chunks = sorted(
            chunks_by_doc[document_id], key=lambda c: c["page_start"]
        )
        merged: list[list[int]] = []
        for chunk in doc_chunks:
            start = chunk["page_start"]
            end = max(chunk["page_start"], chunk["page_end"])
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        pages_covered = sum(end - start + 1 for start, end in merged)
        first_page = merged[0][0]
        last_page = merged[-1][1]
        total_covered += pages_covered
        total_pages += last_page
        tokens = sum(c.get("tokens_est", 0) for c in doc_chunks)
        per_document[document_id] = {
            "theme": doc_chunks[0].get("theme", ""),
            "chunks": len(doc_chunks),
            "tokens_est": tokens,
            "mean_tokens_per_chunk": round(tokens / len(doc_chunks), 1),
            "first_page": first_page,
            "last_page": last_page,
            "pages_covered": pages_covered,
            "coverage_pct": round(100 * pages_covered / max(1, last_page), 1),
        }

    chunk_counts = [entry["chunks"] for entry in per_document.values()]
    thin_documents = {
        doc_id: entry["chunks"] for doc_id, entry in per_document.items()
        if entry["chunks"] < DOCS_MIN_CHUNKS_WARN
    }
    return {
        "documents": len(per_document),
        "mean_chunks_per_document": (
            round(sum(chunk_counts) / len(chunk_counts), 1) if chunk_counts else 0.0
        ),
        "min_chunks_per_document": min(chunk_counts) if chunk_counts else 0,
        "max_chunks_per_document": max(chunk_counts) if chunk_counts else 0,
        "overall_coverage_pct": (
            round(100 * total_covered / max(1, total_pages), 1)
        ),
        "docs_below_min_chunks": DOCS_MIN_CHUNKS_WARN,
        "thin_documents": dict(sorted(thin_documents.items())),
        "per_document": per_document,
    }


def build_report(
    payload: dict[str, Any],
    input_path: Path,
    small_threshold: int,
    large_threshold: int,
    near_dup_threshold: float,
) -> dict[str, Any]:
    """Assemble le rapport qualité complet (structure sérialisée en JSON)."""
    chunks: list[dict[str, Any]] = payload["chunks"]
    tokens_list = [c.get("tokens_est", 0) for c in chunks]
    small_chunks, large_chunks = flag_outliers(chunks, small_threshold, large_threshold)

    return {
        "generator": "evaluate_chunks.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "file": input_path.name,
            "generator": payload.get("generator"),
            "generated_at": payload.get("generated_at"),
            "config": payload.get("config"),
        },
        "thresholds": {
            "too_small_tokens": small_threshold,
            "too_large_tokens": large_threshold,
            "near_dup_jaccard": near_dup_threshold,
            "shingle_words": SHINGLE_WORDS,
            "size_bin_width": SIZE_BIN_WIDTH,
            "docs_min_chunks_warn": DOCS_MIN_CHUNKS_WARN,
        },
        "totals": {
            "chunks": len(chunks),
            "documents": len({c["document_id"] for c in chunks}),
            "themes": len({c.get("theme", "?") for c in chunks}),
            "tokens_est": sum(tokens_list),
            "empty_text_chunks": sum(
                1 for c in chunks if not normalize_text(c["text"])
            ),
        },
        "theme_breakdown": theme_breakdown(chunks),
        "size_distribution": size_distribution(tokens_list),
        "pages_per_chunk": page_span_stats(chunks),
        "outliers": {
            "too_small": {
                "threshold": small_threshold,
                "count": len(small_chunks),
                "chunks": small_chunks,
            },
            "too_large": {
                "threshold": large_threshold,
                "count": len(large_chunks),
                "chunks": large_chunks,
            },
        },
        "duplication": duplication_analysis(chunks, near_dup_threshold),
        "coverage": coverage_analysis(chunks),
    }


# ---------------------------------------------------------------------------
# Rendu Markdown
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Construit une table Markdown simple à partir d'en-têtes et de lignes."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """Rend le rapport qualité au format Markdown lisible."""
    totals = report["totals"]
    size = report["size_distribution"]
    pages = report["pages_per_chunk"]
    dup = report["duplication"]
    coverage = report["coverage"]
    thresholds = report["thresholds"]
    lines: list[str] = []

    lines.append("# Rapport qualité des chunks (JOUR 3)")
    lines.append("")
    lines.append(f"- Généré le : {report['generated_at']}")
    lines.append(
        f"- Source : `{report['input']['file']}` "
        f"(generator={report['input'].get('generator')}, "
        f"généré le {report['input'].get('generated_at')})"
    )
    config = report["input"].get("config") or {}
    if config:
        rendered_config = ", ".join(f"{key}={value}" for key, value in config.items())
        lines.append(f"- Config du chunking : {rendered_config}")
    lines.append(
        f"- Seuils : petits <{thresholds['too_small_tokens']} t · "
        f"grands >{thresholds['too_large_tokens']} t · "
        f"quasi-doublon Jaccard ≥{thresholds['near_dup_jaccard']}"
    )
    lines.append("")

    lines.append("## Volumes")
    lines.append("")
    lines.append(_md_table(
        ["Indicateur", "Valeur"],
        [
            ["Chunks totaux", totals["chunks"]],
            ["Documents couverts", totals["documents"]],
            ["Thèmes", totals["themes"]],
            ["Tokens estimés (total)", f"{totals['tokens_est']:,}"],
            ["Chunks au texte vide", totals["empty_text_chunks"]],
        ],
    ))
    lines.append("")
    theme_rows = [
        [theme, data["documents"], data["chunks"], f"{data['tokens_est']:,}",
         data["mean_chunks_per_document"], data["mean_tokens_per_chunk"]]
        for theme, data in report["theme_breakdown"].items()
    ]
    lines.append(_md_table(
        ["Thème", "Documents", "Chunks", "Tokens est.", "Chunks/doc", "Tokens/chunk"],
        theme_rows,
    ))

    lines.append("")
    lines.append("## Taille des chunks (tokens estimés)")
    lines.append("")
    size_rows = [[key, size[key]] for key in
                 ("min", "p05", "p10", "p25", "median", "mean",
                  "p75", "p90", "p95", "p99", "max")]
    lines.append(_md_table(["Métrique", "Valeur"], size_rows))
    lines.append("")
    lines.append("Distribution (bins de 100 tokens) :")
    lines.append("")
    bin_rows = [
        [bin_label, count]
        for bin_label, count in size["histogram_100t_bins"].items()
    ]
    lines.append(_md_table(["Bin (tokens)", "Chunks"], bin_rows))

    lines.append("")
    lines.append("## Pages par chunk")
    lines.append("")
    lines.append(_md_table(
        ["Indicateur", "Valeur"],
        [
            ["Span moyen (pages)", pages["mean_span_pages"]],
            ["Span max (pages)", pages["max_span_pages"]],
            ["Chunks mono-page", pages["single_page_chunks"]],
            ["Chunks multi-pages",
             f"{pages['multi_page_chunks']} ({pages['multi_page_pct']} %)"],
        ],
    ))
    lines.append("")
    span_rows = [[span, count] for span, count in pages["span_distribution"].items()]
    lines.append(_md_table(["Pages / chunk", "Chunks"], span_rows))

    for label, key in (("trop petits", "too_small"), ("trop grands", "too_large")):
        outlier = report["outliers"][key]
        lines.append("")
        lines.append(f"## Chunks {label} ({outlier['count']})")
        lines.append("")
        outlier_rows = [
            [c["chunk_id"][:8], c["document_id"], c["theme"],
             c["tokens_est"], c["span_pages"], c["preview"]]
            for c in outlier["chunks"][:15]
        ]
        if outlier_rows:
            lines.append(_md_table(
                ["Chunk", "Document", "Thème", "Tokens", "Pages", "Aperçu"],
                outlier_rows,
            ))
            if len(outlier["chunks"]) > 15:
                lines.append("")
                lines.append("*(liste tronquée à 15)*")
        else:
            lines.append("_Aucun._")

    lines.append("")
    dup_verdict = "OUI" if dup["excessive_duplication"] else "non"
    lines.append("## Redondance / duplication")
    lines.append("")
    lines.append(_md_table(
        ["Indicateur", "Valeur"],
        [
            ["Doublons exacts (groupes)", dup["exact_duplicate_groups"]],
            ["Chunks redondants (exacts)", dup["exact_redundant_chunks"]],
            ["Paires quasi-doublons (même document)", dup["near_duplicate_pairs"]],
            ["Chunks impliqués (quasi-doublons)",
             dup["near_duplicate_chunks_involved"]],
            ["Duplication excessive ?", dup_verdict],
        ],
    ))

    lines.append("")
    lines.append("## Couverture documentaire")
    lines.append("")
    lines.append(_md_table(
        ["Indicateur", "Valeur"],
        [
            ["Documents", coverage["documents"]],
            ["Chunks / document (moyen)", coverage["mean_chunks_per_document"]],
            ["Chunks / document (min – max)",
             f"{coverage['min_chunks_per_document']} – "
             f"{coverage['max_chunks_per_document']}"],
            ["Couverture pages (globale)", f"{coverage['overall_coverage_pct']} %"],
        ],
    ))
    if coverage["thin_documents"]:
        lines.append("")
        thin_rows = [
            [doc_id, count] for doc_id, count in coverage["thin_documents"].items()
        ]
        lines.append(
            f"Documents sous le seuil de {coverage['docs_below_min_chunks']} chunks :"
        )
        lines.append("")
        lines.append(_md_table(["Document", "Chunks"], thin_rows))
    lines.append("")
    per_doc_rows = [
        [doc_id, entry["theme"], entry["chunks"], entry["tokens_est"],
         f"{entry['first_page']}–{entry['last_page']}",
         entry["pages_covered"], f"{entry['coverage_pct']} %"]
        for doc_id, entry in coverage["per_document"].items()
    ]
    lines.append(_md_table(
        ["Document", "Thème", "Chunks", "Tokens est.", "Pages",
         "Pages couvertes", "Couverture"],
        per_doc_rows,
    ))
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    corpus_dir: Path, report: dict[str, Any], markdown: str
) -> tuple[Path, Path]:
    """Écrit `chunk_stats.json` et `chunk_stats.md` sous corpus/.

    Returns:
        Tuple des deux chemins écrits (json, markdown).
    """
    json_path = corpus_dir / OUTPUT_JSON_FILENAME
    md_path = corpus_dir / OUTPUT_MD_FILENAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI de l'évaluateur."""
    parser = argparse.ArgumentParser(
        description="Rapport qualité du corpus de chunks (chunk_stats.json/.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python evaluate_chunks.py                       # seuils par défaut\n"
            "  python evaluate_chunks.py --small-tokens 120    # petit chunk < 120 t\n"
            "  python evaluate_chunks.py --near-dup-jaccard 0.5 # redondance plus stricte\n"
        ),
    )
    parser.add_argument(
        "--input", default=INPUT_FILENAME,
        help=f"Fichier d'entrée sous corpus/ (défaut : {INPUT_FILENAME}).",
    )
    parser.add_argument(
        "--small-tokens", type=int, default=DEFAULT_SMALL_TOKENS, metavar="N",
        help=f"Seuil 'chunk trop petit' en tokens (défaut : {DEFAULT_SMALL_TOKENS}).",
    )
    parser.add_argument(
        "--large-tokens", type=int, default=DEFAULT_LARGE_TOKENS, metavar="N",
        help=f"Seuil 'chunk trop grand' en tokens (défaut : {DEFAULT_LARGE_TOKENS}).",
    )
    parser.add_argument(
        "--near-dup-jaccard", type=float, default=DEFAULT_NEAR_DUP_JACCARD,
        metavar="X",
        help=(f"Similarité Jaccard => quasi-doublon "
              f"(défaut : {DEFAULT_NEAR_DUP_JACCARD})."),
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : charge chunks.json, calcule les métriques, écrit les rapports."""
    log = get_logger()
    args = parse_args()

    if args.small_tokens <= 0 or args.large_tokens <= 0:
        raise SystemExit("--small-tokens et --large-tokens doivent être > 0.")
    if not 0.0 < args.near_dup_jaccard <= 1.0:
        raise SystemExit("--near-dup-jaccard doit être dans ]0 ; 1].")

    input_path = locate_corpus(args.input)
    payload = load_chunks(input_path)
    chunks: list[dict[str, Any]] = payload["chunks"]
    if not chunks:
        raise SystemExit(f"{input_path.name} ne contient aucun chunk.")

    log.info("Évaluation du corpus : %s (%d chunks)", input_path, len(chunks))
    report = build_report(
        payload=payload,
        input_path=input_path,
        small_threshold=args.small_tokens,
        large_threshold=args.large_tokens,
        near_dup_threshold=args.near_dup_jaccard,
    )
    markdown = render_markdown(report)
    json_path, md_path = write_outputs(input_path.parent, report, markdown)

    # --- Résumé console ---
    totals = report["totals"]
    size = report["size_distribution"]
    pages = report["pages_per_chunk"]
    dup = report["duplication"]
    coverage = report["coverage"]

    log.info("Chunks           : %d (%d documents, %d thèmes)",
             totals["chunks"], totals["documents"], totals["themes"])
    log.info("Taille / chunk   : moyenne %s · médiane %s · min %s · max %s",
             size["mean"], size["median"], size["min"], size["max"])
    log.info("Pages / chunk    : moyen %s · multi-pages %s (%s %%)",
             pages["mean_span_pages"], pages["multi_page_chunks"],
             pages["multi_page_pct"])
    log.info("Trop petits (<%d) : %d · Trop grands (>%d) : %d",
             args.small_tokens, len(report["outliers"]["too_small"]["chunks"]),
             args.large_tokens, len(report["outliers"]["too_large"]["chunks"]))
    log.info("Duplication      : %d doublon(s) exact(s) · %d paire(s) quasi-identique(s)",
             dup["exact_redundant_chunks"], dup["near_duplicate_pairs"])
    log.info("Couverture       : %.1f %% des pages · %s chunks/doc en moyenne",
             coverage["overall_coverage_pct"], coverage["mean_chunks_per_document"])

    if dup["excessive_duplication"]:
        log.warning("Duplication excessive détectée : vérifiez le chunking/overlap.")
    if coverage["thin_documents"]:
        log.warning("%d document(s) avec < %d chunks : %s",
                    len(coverage["thin_documents"]),
                    coverage["docs_below_min_chunks"],
                    ", ".join(coverage["thin_documents"]))
    if totals["empty_text_chunks"]:
        log.warning("%d chunk(s) au texte vide.", totals["empty_text_chunks"])

    log.info("=> %s", json_path)
    log.info("=> %s", md_path)


if __name__ == "__main__":
    main()
