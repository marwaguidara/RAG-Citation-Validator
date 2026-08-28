"""
Reranking cross-encoder des résultats hybrides (JOUR 6).

Inputs :
    résultats de hybrid_search.search()   (pool hybride, ~top-20)
    corpus/chunks.json                    (texte complet : requis par le
                                           cross-encoder ; le preview hybride
                                           de 160 caractères ne suffit pas)

Objectif :
    Re-scorer les candidats hybrides avec le cross-encoder BAAI/bge-reranker-base
    sur les paires (query, chunk_text), puis retourner un classement final.

Pipeline :
    1. réception du pool hybride (~top-20 résultats SearchResult)
    2. construction des paires (query, chunk_text)
    3. scores du cross-encoder (inférence batchée, torch.no_grad)
    4. re-classement par score décroissant (ties => chunk_id)
    5. retour du top-K final (défaut 5) + métriques

Mesures :
    - latence reranker (inférence seule) et latence totale
    - taille du pool initial / taille du pool final
    - mouvement de classement (Δrang moyen avant/après)

Sortie :
    corpus/reranker_report.json   rapport traçable par requête : classement
    AVANT/APRÈS, scores, latences, tailles de pools, readiness pour l'étape
    de génération de réponses.

Contraintes respectées : aucun LLM génératif, aucune API HTTP, aucune UI,
pas de vérification NLI. Ce module s'arrête au classement reranké — entrée
directe de la génération de réponses (étape suivante).

Usage programme :
    from rerank_results import CrossEncoderReranker, rerank_results
    reranker = CrossEncoderReranker()
    response = rerank_results(query, hybrid_results, text_by_id, reranker)

Usage CLI :
    python rerank_results.py                      # 3 requêtes de test intégrées
    python rerank_results.py --pool-size 30 --top-k 8 "How does ReAct work?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Types partagés avec l'étape hybride (aucune duplication de contrat).
from hybrid_search import HybridResponse, SearchResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
REPORT_FILENAME = "reranker_report.json"

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_POOL_SIZE = 20         # candidats hybrides re-scorés (top-20)
DEFAULT_TOP_K = 5              # résultats finaux après reranking
DEFAULT_BATCH_SIZE = 16        # paires par lot d'inférence
MAX_SEQ_LENGTH = 512           # borne du cross-encoder (query + passage)
EXAMPLE_LIMIT = 10             # nb max d'anomalies conservées au rapport

DEFAULT_TEST_QUERIES: tuple[str, ...] = (
    "What is retrieval augmented generation?",
    "How does ReAct work?",
    "What is LoRA fine tuning?",
)


class Status(StrEnum):
    """Verdict de la session de reranking."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


def build_logger(as_json: bool) -> tuple[logging.Logger, "EventLogger"]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("rerank_results")
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


def load_chunk_text_index(
    input_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Charge chunks.json : mapping chunk_id -> chunk (texte complet inclus)."""
    try:
        with input_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON invalide dans {input_path.name} : {exc}") from exc
    chunks = payload.get("chunks", [])
    if not chunks:
        raise SystemExit(f"{input_path.name} ne contient aucun chunk.")
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    return by_id, payload


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Wrapper d'inférence pour BAAI/bge-reranker-base (cross-encoder).

    Le modèle score des paires (query, passage) : plus le logit est élevé,
    plus le passage est pertinent pour la requête.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "auto",
        max_seq_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self.max_seq_length = max_seq_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def score_pairs(
        self, query: str, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[float]:
        """Score chaque paire (query, text) ; retourne les logits bruts.

        Args:
            query: requête en langage naturel (identique pour toutes les paires).
            texts: passages à scorer.
            batch_size: paires par lot d'inférence.
        """
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = self._tokenizer(
                [[query, text] for text in batch_texts],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
        return scores


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RerankedResult:
    """Un chunk du classement final après cross-encoder."""

    final_rank: int
    chunk_id: str
    document_id: str
    theme: str
    page_start: int
    page_end: int
    hybrid_rank: int           # rang dans le pool hybride AVANT reranking
    hybrid_score: float        # score RRF hybride
    reranker_score: float      # logit brut du cross-encoder
    channels: list[str]
    text_preview: str = ""


@dataclass(slots=True)
class RerankMetrics:
    """Métriques d'une exécution de reranking."""

    pool_initial_size: int
    pool_final_size: int
    pairs_scored: int
    reranker_ms: float
    total_ms: float
    mean_abs_rank_change: float   # |Δrang| moyen avant/après (0..pool)


@dataclass(slots=True)
class RerankResponse:
    """Réponse complète d'un reranking."""

    query: str
    results: list[RerankedResult]
    metrics: RerankMetrics


def _resolve_reranker(
    reranker: CrossEncoderReranker | None,
    model_name: str,
    device: str,
) -> CrossEncoderReranker:
    """Retourne le reranker fourni ou instancie le défaut paresseusement."""
    if reranker is not None:
        return reranker
    return CrossEncoderReranker(model_name=model_name, device=device)


def rerank_results(
    query: str,
    hybrid_results: list[SearchResult],
    chunk_index: dict[str, dict[str, Any]],
    reranker: CrossEncoderReranker | None = None,
    top_k: int = DEFAULT_TOP_K,
    pool_size: int = DEFAULT_POOL_SIZE,
    model_name: str = "BAAI/bge-reranker-base",
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RerankResponse:
    """Re-score le pool hybride avec le cross-encoder et retourne le top-K final.

    Args:
        query: requête en langage naturel.
        hybrid_results: résultats du canal hybride (RRF), meilleur en premier.
        chunk_index: mapping chunk_id -> chunk (texte complet depuis chunks.json).
        reranker: instance optionnelle réutilisable ; sinon créée à la demande.
        top_k: taille du classement final.
        pool_size: nombre maximum de candidats hybrides re-scorés.
        model_name: nom du cross-encoder si instanciation interne.
        device: device d'inférence ('auto' => CUDA si disponible).
        batch_size: paires par lot d'inférence.

    Returns:
        RerankResponse : top-K final + métriques (pools, latence, Δrang moyen).
    """
    started_total = time.perf_counter()
    engine = _resolve_reranker(reranker, model_name, device)

    # 1. Pool initial : top-N hybride, uniquement les chunks dont on a le texte.
    pool = [
        result for result in hybrid_results[:pool_size]
        if str(result.chunk_id) in chunk_index
    ]
    pool_initial_size = len(pool)
    if pool_initial_size == 0:
        raise SystemExit("Pool hybride vide ou sans texte disponible pour le reranking.")

    # 2. Paires (query, chunk_text) puis 3. scores du cross-encoder.
    texts = [str(chunk_index[str(result.chunk_id)]["text"]) for result in pool]
    started = time.perf_counter()
    reranker_scores = engine.score_pairs(query, texts, batch_size=batch_size)
    reranker_ms = (time.perf_counter() - started) * 1000

    # 4. Re-classement déterministe : score décroissant, ties => chunk_id.
    scored_pool = sorted(
        zip(pool, reranker_scores),
        key=lambda pair: (-pair[1], str(pair[0].chunk_id)),
    )

    # 5. Top-K final avec métriques de mouvement de classement.
    rank_changes: list[int] = []
    reranked_results: list[RerankedResult] = []
    for final_rank, (hybrid_result, reranker_score) in enumerate(
        scored_pool[:top_k], start=1
    ):
        rank_changes.append(abs(hybrid_result.rank - final_rank))
        source_chunk = chunk_index[str(hybrid_result.chunk_id)]
        reranked_results.append(RerankedResult(
            final_rank=final_rank,
            chunk_id=str(hybrid_result.chunk_id),
            document_id=str(source_chunk["document_id"]),
            theme=str(source_chunk["theme"]),
            page_start=int(source_chunk["page_start"]),
            page_end=int(source_chunk["page_end"]),
            hybrid_rank=hybrid_result.rank,
            hybrid_score=hybrid_result.rrf_score,
            reranker_score=round(float(reranker_score), 4),
            channels=list(hybrid_result.channels),
            text_preview=str(source_chunk.get("text", ""))[:160].replace("\n", " "),
        ))

    total_ms = (time.perf_counter() - started_total) * 1000
    metrics = RerankMetrics(
        pool_initial_size=pool_initial_size,
        pool_final_size=len(reranked_results),
        pairs_scored=len(reranker_scores),
        reranker_ms=round(reranker_ms, 2),
        total_ms=round(total_ms, 2),
        mean_abs_rank_change=(
            round(sum(rank_changes) / len(rank_changes), 2) if rank_changes else 0.0
        ),
    )
    return RerankResponse(query=query, results=reranked_results, metrics=metrics)


# ---------------------------------------------------------------------------
# Rapport de session
# ---------------------------------------------------------------------------

def aggregate_metrics(responses: list[RerankResponse]) -> dict[str, Any]:
    """Agrège les métriques de toutes les requêtes rerankées."""
    count = len(responses)
    if not count:
        return {"queries": 0}
    return {
        "queries": count,
        "mean_reranker_ms": round(
            sum(r.metrics.reranker_ms for r in responses) / count, 2
        ),
        "max_reranker_ms": max(r.metrics.reranker_ms for r in responses),
        "mean_total_ms": round(
            sum(r.metrics.total_ms for r in responses) / count, 2
        ),
        "mean_abs_rank_change": round(
            sum(r.metrics.mean_abs_rank_change for r in responses) / count, 2
        ),
        "pairs_scored_total": sum(r.metrics.pairs_scored for r in responses),
    }


def assess_generation_readiness(
    responses: list[RerankResponse],
    chunk_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Vérifie que le classement final est prêt pour la génération de réponses.

    Contrôles : chaque requête retourne exactement `top_k` résultats ; chaque
    résultat pointe vers un texte complet exploitable ; les champs de citation
    (document_id, pages) sont présents.
    """
    problems: list[str] = []
    full_text_results = 0
    for response in responses:
        for result in response.results:
            source = chunk_index.get(result.chunk_id)
            if source is None:
                problems.append(f"{result.chunk_id} absent de chunks.json")
                continue
            if len(str(source.get("text", ""))) < 100:
                problems.append(f"{result.chunk_id} : texte trop court")
            else:
                full_text_results += 1
            if not result.document_id or result.page_start <= 0:
                problems.append(f"{result.chunk_id} : champs de citation incomplets")
    return {
        "ready_for_generation": bool(responses) and not problems,
        "results_with_full_text": full_text_results,
        "problems": problems[:EXAMPLE_LIMIT],
    }


def build_session_report(
    model_name: str,
    device: str,
    pool_size: int,
    top_k: int,
    batch_size: int,
    hybrid_responses: list["HybridResponse"],
    responses: list[RerankResponse],
    chunks_sha256: str | None,
    chunk_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble reranker_report.json (sérialisable en JSON)."""
    from importlib.metadata import version

    readiness = assess_generation_readiness(responses, chunk_index)
    status = (
        Status.PASS.value
        if len(responses) == len(hybrid_responses)
        and readiness["ready_for_generation"]
        else Status.FAIL.value
    )
    return {
        "generator": "rerank_results.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "config": {
            "model": model_name,
            "device": device,
            "pool_size": pool_size,
            "top_k": top_k,
            "batch_size": batch_size,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
        "input_traceability": {
            "chunks_file": INPUT_FILENAME,
            "chunks_sha256": chunks_sha256,
        },
        "versions": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "transformers_version": version("transformers"),
            "torch_version": torch.__version__,
        },
        "aggregate_metrics": aggregate_metrics(responses),
        "generation_readiness": readiness,
        "queries": [
            {
                "query": reranked.query,
                "metrics": asdict(reranked.metrics),
                "before_ranking": [
                    {
                        "hybrid_rank": hybrid_result.rank,
                        "chunk_id": hybrid_result.chunk_id,
                        "rrf_score": hybrid_result.rrf_score,
                    }
                    for hybrid_result in hybrid_response.results
                ],
                "after_ranking": [asdict(result) for result in reranked.results],
            }
            for hybrid_response, reranked in zip(hybrid_responses, responses)
        ],
    }


def queries_count(hybrid_responses: list["HybridResponse"]) -> int:
    """Nombre de requêtes attendues (utilitaire de lisibilité du statut)."""
    return len(hybrid_responses)


def write_session_report(corpus_dir: Path, report: dict[str, Any]) -> Path:
    """Écrit reranker_report.json sous corpus/ et retourne son chemin."""
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
    """Parse les arguments CLI de la session de reranking."""
    parser = argparse.ArgumentParser(
        description=(
            "Reranking cross-encoder (BAAI/bge-reranker-base) du pool hybride "
            "(rapport : reranker_report.json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sans requête fournie, exécute les 3 requêtes de test du projet.\n"
            "Exemples :\n"
            '  python rerank_results.py "How does ReAct work?"\n'
            '  python rerank_results.py --pool-size 30 --top-k 8'
        ),
    )
    parser.add_argument("queries", nargs="*", default=None,
                        help="Requêtes à exécuter (défaut : 3 requêtes de test).")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                        metavar="N",
                        help=(f"Candidats hybrides re-scorés ({DEFAULT_POOL_SIZE})."))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, metavar="N",
                        help=f"Résultats finaux après reranking ({DEFAULT_TOP_K}).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"Cross-encoder ({DEFAULT_MODEL_NAME}).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device d'inférence.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        metavar="N",
                        help=f"Paires par lot d'inférence ({DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--hybrid-fetch-k", type=int, default=50, metavar="N",
                        help="Profondeur de recherche hybride en amont (50).")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Format des logs console (défaut : text).")
    return parser.parse_args()


def _console_safe(text: str) -> str:
    """Rend un texte affichable sur la console courante (Windows cp1252 inclus)."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_before_after(
    hybrid_response: HybridResponse, rerank_response: RerankResponse
) -> None:
    """Affiche le classement AVANT (RRF) et APRÈS (cross-encoder)."""
    metrics = rerank_response.metrics
    print(_console_safe(f"\nQUERY: {rerank_response.query}"))
    print(f"  AVANT  (hybride RRF) — pool initial {metrics.pool_initial_size} candidats :")
    for result in hybrid_response.results[:5]:
        print(
            f"    #{result.rank} {result.chunk_id[:8]} {result.document_id} "
            f"p.{result.page_start}-{result.page_end} | rrf={result.rrf_score:.4f}"
        )
    print(f"  APRÈS  (cross-encoder) — top final {metrics.pool_final_size} :")
    for result in rerank_response.results:
        moved = (
            f"(hybride #{result.hybrid_rank} -> #{result.final_rank})"
            if result.hybrid_rank != result.final_rank else "(stable)"
        )
        print(
            f"    #{result.final_rank} {result.chunk_id[:8]} "
            f"{result.document_id} p.{result.page_start}-{result.page_end} | "
            f"reranker={result.reranker_score:.4f} rrf={result.hybrid_score:.4f} "
            f"{moved}"
        )
    print(
        f"  métriques -> latence reranker {metrics.reranker_ms} ms · total "
        f"{metrics.total_ms} ms · paires scorées {metrics.pairs_scored} · "
        f"{_console_safe('|Δrang|')} moyen {metrics.mean_abs_rank_change}"
    )


def main() -> None:
    """Point d'entrée : hybride top-N -> cross-encoder -> top-K final + rapport."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.pool_size <= 0 or args.top_k <= 0 or args.batch_size <= 0:
        raise SystemExit("--pool-size, --top-k et --batch-size doivent être > 0.")
    if args.pool_size < args.top_k:
        raise SystemExit("--pool-size doit être >= --top-k.")

    queries = list(args.queries) if args.queries else list(DEFAULT_TEST_QUERIES)
    chunks_path = locate_corpus(INPUT_FILENAME)
    if chunks_path is None:
        raise SystemExit(f"{INPUT_FILENAME} introuvable sous corpus/.")
    chunks_sha256 = sha256_of(chunks_path)
    chunk_index, _payload = load_chunk_text_index(chunks_path)

    # Étape amont : recherche hybride (top `pool_size` candidats par requête).
    from hybrid_search import EngineConfig, get_engine

    events.info("hybrid_search_started", queries=len(queries))
    engine = get_engine(EngineConfig(fetch_k=max(args.pool_size, args.top_k)))
    try:
        hybrid_responses = [
            engine.search(query=query, top_k=args.pool_size) for query in queries
        ]
    finally:
        engine.close()

    # Reranking cross-encoder.
    events.info("reranker_loading", model=args.model, device=args.device)
    reranker = CrossEncoderReranker(model_name=args.model, device=args.device)
    responses = [
        rerank_results(
            query=query,
            hybrid_results=hybrid_response.results,
            chunk_index=chunk_index,
            reranker=reranker,
            top_k=args.top_k,
            pool_size=args.pool_size,
            batch_size=args.batch_size,
        )
        for query, hybrid_response in zip(queries, hybrid_responses)
    ]

    for hybrid_response, rerank_response in zip(hybrid_responses, responses):
        _print_before_after(hybrid_response, rerank_response)

    report = build_session_report(
        model_name=args.model,
        device=reranker.device,
        pool_size=args.pool_size,
        top_k=args.top_k,
        batch_size=args.batch_size,
        hybrid_responses=hybrid_responses,
        responses=responses,
        chunks_sha256=chunks_sha256,
        chunk_index=chunk_index,
    )
    report_path = write_session_report(chunks_path.parent, report)

    aggregate = report["aggregate_metrics"]
    readiness = report["generation_readiness"]
    events.info(
        "session_completed",
        status=report["status"],
        mean_reranker_ms=aggregate.get("mean_reranker_ms"),
        max_reranker_ms=aggregate.get("max_reranker_ms"),
        pairs_scored_total=aggregate.get("pairs_scored_total"),
    )
    events.info(
        "generation_readiness",
        ready=readiness["ready_for_generation"],
        results_with_full_text=readiness["results_with_full_text"],
        problems=len(readiness["problems"]),
    )
    print(f"\n=> rapport : {report_path}")
    sys.exit(EXIT_OK if report["status"] == Status.PASS.value else EXIT_FAIL)


if __name__ == "__main__":
    main()