"""
Framework d'évaluation comparative final du pipeline RAG Citation Validator.

Compare quatre configurations du pipeline sur un jeu de questions annotées
(``annotation_template.json``, ground truth réel pointant vers des chunk_id
UUID existant dans ``corpus/chunks.json``) :

    A  Dense Retrieval uniquement              (BGE / Qdrant)
    B  Dense Retrieval + BM25                  (fusion RRF)
    C  Dense + BM25 + BGE Reranker             (cross-encoder)
    D  C + Generation + Citation Verification  (RoBERTa-MNLI)

Métriques calculées à partir des sorties réelles du pipeline :
    Retrieval   : Recall@3 / Recall@5 / Recall@10, MRR (chunk level)
    Generation  : Faithfulness (support NLI max par affirmation, config D)
    Vérification: Citation Accuracy, Mean Support Score,
                  Supported / Weak Support / Unsupported Rates (config D)
    Performance : Average Latency, P95 Latency (par configuration)

Sorties (dans ``--output-dir``, défaut ``corpus/evaluation/``) :
    evaluation_report.json   rapport machine complet et traçable
    evaluation_report.md     rapport lisible (gains, meilleure config,
                             trade-offs précision / latence)
    comparison_plot.png      graphiques comparatifs (matplotlib)

Traçabilité et reproductibilité :
    - SHA-256 de ``chunks.json`` et du fichier d'annotations enregistrés ;
    - versions Python / plate-forme / librairies enregistrées ;
    - aucune métrique simulée : si une métrique ne peut pas être calculée
      (ex. Faithfulness pour A/B/C), elle vaut ``null`` accompagnée d'une
      raison explicite dans le rapport.

Usage :
    python evaluate_pipeline.py
    python evaluate_pipeline.py --limit 4 --configs A,B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # rendu sans serveur graphique
import matplotlib.pyplot as plt  # noqa: E402

from citation_verifier import MNLICitationVerifier, verify_citations  # noqa: E402
from generate_answer import TemplateProvider, generate_answer  # noqa: E402
from hybrid_search import EngineConfig, HybridSearchEngine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CORPUS_DIR: Path = SCRIPT_DIR / "corpus"

CHUNKS_PATH: Path = CORPUS_DIR / "chunks.json"
ANNOTATIONS_PATH: Path = SCRIPT_DIR / "annotation_template.json"
DEFAULT_OUTPUT_DIR: Path = CORPUS_DIR / "evaluation"

OUTPUT_JSON = "evaluation_report.json"
OUTPUT_MD = "evaluation_report.md"
OUTPUT_PNG = "comparison_plot.png"

# Les quatre configurations comparées, dans l'ordre du rapport.
CONFIG_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
CONFIG_LABELS: dict[str, str] = {
    "A": "A · Dense",
    "B": "B · Hybrid (Dense+BM25)",
    "C": "C · Hybrid + Rerank",
    "D": "D · Hybrid + Rerank + Verification",
}

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "A": "Recherche dense seule (BAAI/bge-small-en-v1.5 via Qdrant, cosinus).",
    "B": "Fusion Reciprocal Rank Fusion des canaux dense et BM25.",
    "C": "Pool hybride top-20 re-scoré par BAAI/bge-reranker-base.",
    "D": ("Pipeline complet : C + génération extractive avec citations "
          "+ vérification NLI roberta-large-mnli."),
}

# Profondeurs d'évaluation.
TOP_K_RETRIEVAL = 10        # profondeur max pour Recall@10
RERANK_POOL_SIZE = 20       # candidats hybrides re-scorés (cf. pipeline Top20)
RERANK_TOP_K = 10           # sorties du reranker conservées pour Recall@10
GENERATION_SOURCES = 5      # sources injectées dans la réponse (cf. pipeline Top5)
SUPPORT_LOW_THRESHOLD = 0.40  # seuil « Weak Support » de citation_verifier.py


def build_logger(log_format: str = "text") -> logging.Logger:
    """Configure et retourne le logger structuré du framework.

    Args:
        log_format: ``text`` (console lisible) ou ``json`` (une ligne JSON
            par événement, pour ingestion par un outil externe).

    Returns:
        Logger configuré, sans propagation vers le root logger.
    """
    logger = logging.getLogger("evaluate_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter
    if log_format == "json":
        formatter = _JsonEventFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S"
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class _JsonEventFormatter(logging.Formatter):
    """Formateur JSON-lines : chaque log devient un objet sérialisable."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **payload: Any,
) -> None:
    """Émet un événement de log, enrichi d'un payload structuré en mode JSON."""
    logger.log(level, event, extra={"payload": payload})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedQuestion:
    """Question annotée dont le ground truth a été résolu dans le corpus."""

    qid: str
    question: str
    theme: str
    difficulty: str
    relevant_chunks: frozenset[str]

    @property
    def n_relevant(self) -> int:
        """Nombre de chunks pertinents annotés."""
        return len(self.relevant_chunks)


@dataclass
class RetrievalMetrics:
    """Métriques de retrieval agrégées pour une configuration."""

    recall_at_3: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None


@dataclass
class VerificationStats:
    """Statistiques de vérification NLI des citations (config D).

    faithfulness suit la convention type RAGAS : une affirmation est
    fidèle si au moins une de ses citations l'entaille ; on moyenne donc,
    par affirmation, le support NLI maximum obtenu parmi ses citations.
    """

    citations_verified: int = 0
    mean_support_score: float | None = None
    citation_accuracy: float | None = None
    supported_rate: float | None = None
    weak_support_rate: float | None = None
    unsupported_rate: float | None = None
    faithfulness: float | None = None


@dataclass
class ConfigResult:
    """Résultat agrégé d'une configuration sur l'ensemble des questions."""

    key: str
    label: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    verification: VerificationStats | None = None
    not_computed: dict[str, str] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Traçabilité et chargement du ground truth
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """Retourne l'empreinte SHA-256 hexadécimale d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotations(
    annotations_path: Path,
    known_chunk_ids: set[str],
    logger: logging.Logger,
) -> tuple[list[ResolvedQuestion], list[dict[str, str]]]:
    """Charge et résout les annotations contre le corpus réel.

    Chaque question doit déclarer des ``relevant_chunks`` existants dans
    ``chunks.json``.  Les questions vides ou non résolubles sont écartées
    avec une raison explicite (traçées dans le rapport) — aucune donnée
    n'est inventée.

    Args:
        annotations_path: chemin de ``annotation_template.json``.
        known_chunk_ids: ensemble des chunk_id réellement indexés.
        logger: logger structuré.

    Returns:
        Tuple (questions résolues, questions écartées avec raison).
    """
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    resolved: list[ResolvedQuestion] = []
    skipped: list[dict[str, str]] = []

    for raw in payload.get("questions", []):
        qid = str(raw.get("id", "?"))
        question = str(raw.get("question", "")).strip()
        raw_chunks = raw.get("relevant_chunks") or []

        if not question or not raw_chunks:
            skipped.append({
                "qid": qid,
                "reason": "slot vide (non encore annoté)",
            })
            continue

        chunk_ids = [str(cid) for cid in raw_chunks]
        unknown = [cid for cid in chunk_ids if cid not in known_chunk_ids]
        if unknown:
            skipped.append({
                "qid": qid,
                "reason": (
                    f"{len(unknown)}/{len(chunk_ids)} chunk_id inconnus du "
                    f"corpus (ex. {unknown[0][:13]}...) — annotation à "
                    "mettre à jour"
                ),
            })
            log_event(logger, "annotation_skipped", level=logging.WARNING,
                      qid=qid, reason="unknown_chunk_ids")
            continue

        resolved.append(ResolvedQuestion(
            qid=qid,
            question=question,
            theme=str(raw.get("theme", "n/a")),
            difficulty=str(raw.get("difficulty", "n/a")),
            relevant_chunks=frozenset(chunk_ids),
        ))

    log_event(logger, "annotations_loaded",
              total=len(payload.get("questions", [])),
              resolved=len(resolved), skipped=len(skipped))
    return resolved, skipped


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: frozenset[str] | set[str],
    k: int,
) -> float:
    """Calcule Recall@k au niveau du chunk.

    Args:
        retrieved_ids: chunk_id récupérés, rang 1 en premier.
        relevant_ids: ground truth (chunk_id pertinents).
        k: profondeur d'évaluation.

    Returns:
        Fraction des chunks pertinents présents dans le top-k.
    """
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: frozenset[str] | set[str],
) -> float:
    """Rang réciproque du premier chunk pertinent (1/rang, 0 si absent)."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def percentile_95(values: list[float]) -> float:
    """95e percentile empirique d'un échantillon de latences (ms)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def mean_or_none(values: list[float]) -> float | None:
    """Moyenne d'une liste, ou None si vide (métrique non calculable)."""
    return round(statistics.mean(values), 4) if values else None


# ---------------------------------------------------------------------------
# Exécution des configurations
# ---------------------------------------------------------------------------

def build_extractive_answer(top_chunks: list[dict[str, Any]]) -> str:
    """Construit une réponse extractive avec citations ``[N]``.

    Chaque source du top-5 contribue une phrase concise extraite de son
    texte et citée via le marqueur ``[N]`` correspondant.  Ce générateur
    déterministe hors-ligne remplace le TemplateProvider (dont la phrase
    générique n'est pas vérifiable par NLI) ; il rend Faithfulness /
    Citation Accuracy réellement mesurables sans serveur LLM.  Limite
    assumée et documentée : la réponse est extractive, non paraphrasée.

    Args:
        top_chunks: chunks ordonnés par pertinence (top-1 en premier).

    Returns:
        Réponse multi-phrases, chaque phrase se terminant par ``[N]``.
    """
    import re as _re

    sentences: list[str] = []
    for idx, chunk in enumerate(top_chunks[:GENERATION_SOURCES], start=1):
        text = str(chunk.get("text", ""))
        best_sent = ""
        for sent in _re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if not 60 <= len(sent) <= 220:
                continue
            if "arXiv:" in sent or "http" in sent:
                continue
            if sent.startswith(("1 ", "2 ", "3 ", "Figure", "Table", "Abstract")):
                continue
            if sum(ch.isalpha() for ch in sent) / max(len(sent), 1) < 0.55:
                continue
            best_sent = sent
            break
        if best_sent:
            sentences.append(f"{best_sent} [{idx}].")
    if not sentences:
        return (
            "Based on the retrieved passages, an answer is synthesized "
            "from the provided sources. [1] [2] [3] [4] [5]."
        )
    return " ".join(sentences[:GENERATION_SOURCES])


def run_dense_only(hybrid_response: Any) -> tuple[list[str], float]:
    """Config A : retrieval dense seul, dérivé d'une recherche hybride.

    Les résultats du canal dense sont re-triés par score dense.  La
    latence retenue est la composante réellement consommée par A :
    encodage de requête + recherche Qdrant (BM25 et fusion RRF sont
    retranchées du temps total mesuré).

    Args:
        hybrid_response: réponse complète de HybridSearchEngine.search().

    Returns:
        Tuple (chunk_id récupérés, latence ms de la config A).
    """
    dense_only = sorted(
        (r for r in hybrid_response.results if "dense" in r.channels),
        key=lambda r: -(r.score_dense or 0.0),
    )
    chunk_ids = [str(r.chunk_id) for r in dense_only[:TOP_K_RETRIEVAL]]
    lat = hybrid_response.latencies_ms
    return chunk_ids, max(lat.total_ms - lat.bm25_ms - lat.fusion_ms, 0.0)


def run_hybrid(
    engine: HybridSearchEngine,
    query: str,
) -> tuple[Any, list[str], float]:
    """Config B : recherche hybride dense + BM25 fusionnée par RRF.

    Args:
        engine: moteur hybride initialisé.
        query: question en langage naturel.

    Returns:
        Tuple (réponse brute du moteur, chunk_id top-10, latence totale ms).
    """
    response = engine.search(query, top_k=RERANK_POOL_SIZE)
    chunk_ids = [str(r.chunk_id) for r in response.results[:TOP_K_RETRIEVAL]]
    return response, chunk_ids, response.latencies_ms.total_ms


def run_rerank_config(
    engine: HybridSearchEngine,
    reranker: CrossEncoderReranker,
    chunk_index: dict[str, dict[str, Any]],
    query: str,
) -> tuple[list[str], float]:
    """Config C : pool hybride top-20 re-scoré par le cross-encoder BGE.

    Args:
        engine: moteur hybride initialisé.
        reranker: reranker BGE initialisé.
        chunk_index: mapping chunk_id -> chunk (texte complet).
        query: question en langage naturel.

    Returns:
        Tuple (chunk_id rerankés top-10, latence totale ms recherche+rerank).
    """
    hybrid_response = engine.search(query, top_k=RERANK_POOL_SIZE)
    reranked = rerank_results(
        query=query,
        hybrid_results=hybrid_response.results,
        chunk_index=chunk_index,
        reranker=reranker,
        top_k=RERANK_TOP_K,
        pool_size=RERANK_POOL_SIZE,
    )
    chunk_ids = [str(r.chunk_id) for r in reranked.results[:TOP_K_RETRIEVAL]]
    total_ms = hybrid_response.latencies_ms.total_ms + reranked.metrics.reranker_ms
    return chunk_ids, round(total_ms, 2)


def summarize_verification(verification_results: list[Any]) -> VerificationStats:
    """Agrège les sorties brutes de verify_citations en statistiques.

    Args:
        verification_results: liste de CitationVerificationResult.

    Returns:
        Accuracy, taux par verdict, support moyen et faithfulness
        (support NLI max par affirmation, convention RAGAS).
    """
    if not verification_results:
        return VerificationStats(citations_verified=0)

    supports = [r.support_score for r in verification_results]
    n = len(supports)
    n_supported = sum(1 for r in verification_results if r.verdict == "Supported")
    n_weak = sum(1 for r in verification_results if r.verdict == "Weak Support")
    n_unsupported = sum(
        1 for r in verification_results if r.verdict == "Unsupported"
    )

    by_claim: dict[str, list[float]] = {}
    for result in verification_results:
        by_claim.setdefault(result.claim_text, []).append(result.support_score)
    per_claim_max = [max(scores) for scores in by_claim.values()]

    return VerificationStats(
        citations_verified=n,
        mean_support_score=round(statistics.mean(supports), 4),
        citation_accuracy=round(n_supported / n, 4),
        supported_rate=round(n_supported / n, 4),
        weak_support_rate=round(n_weak / n, 4),
        unsupported_rate=round(n_unsupported / n, 4),
        faithfulness=(
            round(statistics.mean(per_claim_max), 4) if per_claim_max else None
        ),
    )


def run_full_pipeline(
    engine: HybridSearchEngine,
    reranker: CrossEncoderReranker,
    verifier: MNLICitationVerifier,
    chunk_index: dict[str, dict[str, Any]],
    query: str,
) -> tuple[list[str], float, str, VerificationStats]:
    """Config D : pipeline complet jusqu'à la vérification des citations.

    Étapes : recherche hybride (top-20) -> rerank BGE -> réponse avec
    citations sur le top-5 -> vérification NLI de chaque paire
    (affirmation, passage source).

    Args:
        engine: moteur hybride initialisé.
        reranker: reranker BGE initialisé.
        verifier: vérificateur MNLI initialisé.
        chunk_index: mapping chunk_id -> chunk.
        query: question en langage naturel.

    Returns:
        Tuple (chunk_id top-10, latence totale ms, réponse générée,
        statistiques de vérification NLI).
    """
    t_start = time.perf_counter()

    hybrid_response = engine.search(query, top_k=RERANK_POOL_SIZE)
    reranked = rerank_results(
        query=query,
        hybrid_results=hybrid_response.results,
        chunk_index=chunk_index,
        reranker=reranker,
        top_k=RERANK_TOP_K,
        pool_size=RERANK_POOL_SIZE,
    )
    top_chunks = [
        chunk_index[str(r.chunk_id)]
        for r in reranked.results[:GENERATION_SOURCES]
        if str(r.chunk_id) in chunk_index
    ]
    # Génération par le module amélioré (politique claims ancrés + refus).
    # Les sources passées à verify_citations sont exactement le même top-5
    # reranké, donc les marqueurs [N] de la réponse y correspondent.
    reranked_top = [
        r for r in reranked.results[:GENERATION_SOURCES]
        if str(r.chunk_id) in chunk_index
    ]
    gen = generate_answer(
        query=query,
        reranked_results=reranked_top,
        chunk_index=chunk_index,
        provider=TemplateProvider(),
    )
    answer = gen.answer

    sources = [
        {
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "page_start": int(r.page_start),
            "page_end": int(r.page_end),
        }
        for r in reranked_top
    ]
    verification_results, _segments = verify_citations(
        query=query,
        answer=answer,
        sources=sources,
        chunk_index=chunk_index,
        verifier=verifier,
        batch_size=8,
    )
    stats = summarize_verification(list(verification_results))
    chunk_ids = [str(r.chunk_id) for r in reranked.results[:TOP_K_RETRIEVAL]]
    total_ms = (time.perf_counter() - t_start) * 1000.0
    return chunk_ids, round(total_ms, 2), answer, stats


# ---------------------------------------------------------------------------
# Boucle d'évaluation
# ---------------------------------------------------------------------------

def aggregate_retrieval(per_question: list[dict[str, Any]]) -> RetrievalMetrics:
    """Agrège les métriques de retrieval par question d'une configuration.

    Args:
        per_question: blocs détaillés produits pour chaque question.

    Returns:
        Moyennes Recall@3/5/10 et MRR (None si aucune question).
    """
    def _mean(key: str) -> float | None:
        vals = [q[key] for q in per_question if q.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    return RetrievalMetrics(
        recall_at_3=_mean("recall_at_3"),
        recall_at_5=_mean("recall_at_5"),
        recall_at_10=_mean("recall_at_10"),
        mrr=_mean("mrr"),
    )


def evaluate_configuration(
    key: str,
    questions: list[ResolvedQuestion],
    engine: HybridSearchEngine,
    reranker: CrossEncoderReranker,
    verifier: MNLICitationVerifier,
    chunk_index: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> ConfigResult:
    """Évalue une configuration sur toutes les questions résolues.

    Une seule recherche hybride par question alimente A (canal dense
    isolé) et B (fusion RRF) ; C et D ré-exécutent la recherche avec le
    pool de reranking top-20.

    Args:
        key: lettre de configuration (A/B/C/D).
        questions: questions annotées résolues.
        engine: moteur hybride initialisé.
        reranker: reranker BGE initialisé.
        verifier: vérificateur MNLI initialisé (config D).
        chunk_index: mapping chunk_id -> chunk.
        logger: logger structuré.

    Returns:
        Résultat agrégé de la configuration.

    Raises:
        ValueError: si la clé de configuration est inconnue.
    """
    if key not in CONFIG_ORDER:
        raise ValueError(f"Configuration inconnue : {key!r}")

    result = ConfigResult(key=key, label=CONFIG_LABELS[key])
    if key in ("A", "B", "C"):
        result.not_computed["faithfulness"] = (
            "étape de génération absente de cette configuration"
        )
        result.not_computed["citation_accuracy"] = (
            "étape de vérification NLI absente de cette configuration"
        )

    latencies: list[float] = []
    per_query_verifications: list[VerificationStats] = []

    for i, rq in enumerate(questions, start=1):
        log_event(logger, "query_start",
                  config=key, qid=rq.qid, index=i, total=len(questions))
        vstats = VerificationStats(citations_verified=0)

        if key == "A":
            response = engine.search(rq.question, top_k=RERANK_POOL_SIZE)
            retrieved, latency_ms = run_dense_only(response)
            answer: str | None = None
        elif key == "B":
            _resp, retrieved, latency_ms = run_hybrid(engine, rq.question)
            answer = None
        elif key == "C":
            retrieved, latency_ms = run_rerank_config(
                engine, reranker, chunk_index, rq.question
            )
            answer = None
        else:  # D
            retrieved, latency_ms, answer, vstats = run_full_pipeline(
                engine, reranker, verifier, chunk_index, rq.question
            )
            if vstats.citations_verified:
                per_query_verifications.append(vstats)

        result.per_question.append({
            "qid": rq.qid,
            "question": rq.question,
            "n_relevant": rq.n_relevant,
            "retrieved_chunk_ids": retrieved,
            "recall_at_3": round(recall_at_k(retrieved, rq.relevant_chunks, 3), 4),
            "recall_at_5": round(recall_at_k(retrieved, rq.relevant_chunks, 5), 4),
            "recall_at_10": round(recall_at_k(retrieved, rq.relevant_chunks, 10), 4),
            "mrr": round(reciprocal_rank(retrieved, rq.relevant_chunks), 4),
            "latency_ms": round(latency_ms, 2),
            **({"answer": answer} if answer is not None else {}),
            **({"verification": asdict(vstats)} if key == "D" else {}),
        })
        latencies.append(latency_ms)
        log_event(logger, "query_done", config=key, qid=rq.qid,
                  recall_at_5=result.per_question[-1]["recall_at_5"],
                  latency_ms=round(latency_ms, 2))

    result.retrieval = aggregate_retrieval(result.per_question)
    result.avg_latency_ms = (
        round(statistics.mean(latencies), 2) if latencies else None
    )
    result.p95_latency_ms = (
        round(percentile_95(latencies), 2) if latencies else None
    )

    if key == "D" and per_query_verifications:
        wsum = float(sum(s.citations_verified for s in per_query_verifications))

        def _weighted(attr: str) -> float:
            return round(sum(
                getattr(s, attr) * s.citations_verified
                for s in per_query_verifications
            ) / wsum, 4)

        faith_values = [
            s.faithfulness for s in per_query_verifications
            if s.faithfulness is not None
        ]
        result.verification = VerificationStats(
            citations_verified=int(wsum),
            mean_support_score=_weighted("mean_support_score"),
            citation_accuracy=_weighted("citation_accuracy"),
            supported_rate=_weighted("supported_rate"),
            weak_support_rate=_weighted("weak_support_rate"),
            unsupported_rate=_weighted("unsupported_rate"),
            faithfulness=(
                round(statistics.mean(faith_values), 4) if faith_values else None
            ),
        )
    elif key == "D":
        result.not_computed["citation_verification"] = (
            "aucune paire (affirmation, source) vérifiée sur ce jeu"
        )

    return result


# ---------------------------------------------------------------------------
# Orchestration multi-configurations
# ---------------------------------------------------------------------------

def run_evaluation(
    configs: list[str],
    questions: list[ResolvedQuestion],
    engine: HybridSearchEngine,
    reranker: CrossEncoderReranker,
    verifier: MNLICitationVerifier,
    chunk_index: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, ConfigResult]:
    """Exécute toutes les configurations demandées dans l'ordre A→D.

    Args:
        configs: sous-ensemble de clés A/B/C/D à évaluer.
        questions: questions annotées résolues.
        engine: moteur hybride initialisé.
        reranker: reranker BGE initialisé.
        verifier: vérificateur MNLI initialisé.
        chunk_index: mapping chunk_id -> chunk.
        logger: logger structuré.

    Returns:
        Mapping clé de configuration -> résultat agrégé.
    """
    results: dict[str, ConfigResult] = {}
    for key in [c for c in CONFIG_ORDER if c in configs]:
        log_event(logger, "config_started", config=key,
                  label=CONFIG_LABELS[key], n_questions=len(questions))
        t0 = time.perf_counter()
        results[key] = evaluate_configuration(
            key, questions, engine, reranker, verifier, chunk_index, logger
        )
        log_event(logger, "config_completed", config=key,
                  wall_time_s=round(time.perf_counter() - t0, 1))
    return results


# ---------------------------------------------------------------------------
# Analyse : gains, meilleure configuration, trade-offs
# ---------------------------------------------------------------------------

QUALITY_METRICS: tuple[str, ...] = (
    "recall_at_3", "recall_at_5", "recall_at_10", "mrr",
    "faithfulness", "citation_accuracy",
)


def _metric_value(result: ConfigResult, metric: str) -> float | None:
    """Extrait la valeur d'une métrique depuis un résultat de configuration."""
    if metric in ("faithfulness", "citation_accuracy"):
        if result.verification is None:
            return None
        return getattr(result.verification, metric)
    return getattr(result.retrieval, metric)


def compute_gains(results: dict[str, ConfigResult]) -> list[dict[str, Any]]:
    """Calcule les gains marginaux entre configurations successives.

    Args:
        results: résultats par clé de configuration (A→D).

    Returns:
        Analyses B−A (BM25), C−B (Reranker), D−C (NLI) : delta absolu et
        relatif de chaque métrique disponible + coût latence.
    """
    steps = [
        ("A", "B", "gain_bm25", "Ajout du canal BM25 (fusion RRF)"),
        ("B", "C", "gain_reranker", "Ajout du reranker cross-encoder BGE"),
        ("C", "D", "gain_nli",
         "Ajout génération + vérification NLI des citations"),
    ]
    gains: list[dict[str, Any]] = []
    for src_key, dst_key, gain_key, description in steps:
        if src_key not in results or dst_key not in results:
            continue
        src, dst = results[src_key], results[dst_key]
        metrics_delta: dict[str, Any] = {}
        for metric in QUALITY_METRICS:
            v_src = _metric_value(src, metric)
            v_dst = _metric_value(dst, metric)
            if v_src is None and v_dst is None:
                continue
            metrics_delta[metric] = {
                "from": v_src,
                "to": v_dst,
                "delta_abs": (
                    round(v_dst - v_src, 4)
                    if v_src is not None and v_dst is not None else None
                ),
                "delta_pct": (
                    round((v_dst - v_src) / v_src * 100, 1)
                    if v_src not in (None, 0) and v_dst is not None else None
                ),
            }

        latency_entry: dict[str, Any] | None = None
        if src.avg_latency_ms is not None and dst.avg_latency_ms is not None:
            latency_entry = {
                "from_ms": src.avg_latency_ms,
                "to_ms": dst.avg_latency_ms,
                "delta_ms": round(dst.avg_latency_ms - src.avg_latency_ms, 2),
            }
        gains.append({
            "step": gain_key,
            "description": description,
            "from_config": src.label,
            "to_config": dst.label,
            "metrics": metrics_delta,
            "latency": latency_entry,
        })
    return gains


def find_best_configuration(results: dict[str, ConfigResult]) -> dict[str, Any]:
    """Identifie la meilleure configuration par métrique et globalement.

    Args:
        results: résultats par clé de configuration.

    Returns:
        Gagnant par métrique de qualité, verdict global (métriques les plus
        souvent remportées) et configuration la plus rapide.
    """
    per_metric: dict[str, str | None] = {}
    for metric in QUALITY_METRICS:
        candidates = [
            (_metric_value(res, metric), key)
            for key, res in results.items()
            if _metric_value(res, metric) is not None
        ]
        per_metric[metric] = max(candidates)[1] if candidates else None

    wins = [k for k in per_metric.values() if k]
    overall = max(set(wins), key=wins.count) if wins else None

    lat_candidates = [
        (res.avg_latency_ms, key) for key, res in results.items()
        if res.avg_latency_ms is not None
    ]
    fastest = min(lat_candidates)[1] if lat_candidates else None

    return {
        "best_per_quality_metric": per_metric,
        "overall_best_quality": CONFIG_LABELS.get(overall or "") or None,
        "fastest_config": CONFIG_LABELS.get(fastest or "") or None,
    }


def analyze_tradeoff(results: dict[str, ConfigResult]) -> dict[str, Any]:
    """Analyse précision / latence entre configurations successives.

    Calcule pour chaque étape le coût en latence par point de Recall@5
    gagné, lorsque le gain est mesurable.

    Args:
        results: résultats par clé de configuration.

    Returns:
        Trade-offs par étape (bm25 / reranker / nli).
    """
    tradeoffs: dict[str, Any] = {}
    for src_key, dst_key, step in (("A", "B", "bm25"), ("B", "C", "reranker"),
                                   ("C", "D", "nli")):
        if src_key not in results or dst_key not in results:
            continue
        src, dst = results[src_key], results[dst_key]
        if (src.retrieval.recall_at_5 is None
                or dst.retrieval.recall_at_5 is None
                or src.avg_latency_ms is None
                or dst.avg_latency_ms is None):
            continue
        delta_recall_pp = (
            (dst.retrieval.recall_at_5 - src.retrieval.recall_at_5) * 100
        )
        delta_latency = dst.avg_latency_ms - src.avg_latency_ms
        cost_per_point: float | str = (
            round(delta_latency / delta_recall_pp, 1)
            if delta_recall_pp > 0 else "pas de gain de recall (coût pur)"
        )
        tradeoffs[step] = {
            "delta_recall_at_5_points": round(delta_recall_pp, 2),
            "delta_avg_latency_ms": round(delta_latency, 2),
            "latency_ms_per_recall_point": cost_per_point,
        }
    return tradeoffs


def _safe_version(module_name: str) -> str | None:
    """Retourne la version d'une librairie, ou None si introuvable."""
    import importlib.metadata as _meta
    for candidate in (module_name, module_name.replace("-", "_")):
        try:
            return _meta.version(candidate)
        except Exception:
            continue
    return None


def build_report_dict(
    results: dict[str, ConfigResult],
    resolved: list[ResolvedQuestion],
    skipped: list[dict[str, str]],
    judged_chunks: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble le rapport JSON final (configurations / metrics / summary).

    Args:
        results: résultats agrégés par configuration.
        resolved: questions réellement évaluées.
        skipped: questions écartées avec raison explicite.
        judged_chunks: nombre de chunks indexés dans le corpus.
        args: arguments CLI (trace de la commande exacte).

    Returns:
        Dict sérialisable en JSON.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sentence_transformers": _safe_version("sentence-transformers"),
        "transformers": _safe_version("transformers"),
        "torch": _safe_version("torch"),
        "rank_bm25": _safe_version("rank-bm25"),
        "qdrant_client": _safe_version("qdrant-client"),
    }

    configurations: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for key in CONFIG_ORDER:
        res = results.get(key)
        if res is None:
            continue
        configurations[key] = {
            "label": res.label,
            "description": CONFIG_DESCRIPTIONS[key],
            "n_questions": len(res.per_question),
            "per_question": res.per_question,
            "verification_aggregate": (
                asdict(res.verification) if res.verification else None
            ),
            "not_computed": dict(res.not_computed),
        }
        m: dict[str, Any] = {
            "recall_at_3": res.retrieval.recall_at_3,
            "recall_at_5": res.retrieval.recall_at_5,
            "recall_at_10": res.retrieval.recall_at_10,
            "mrr": res.retrieval.mrr,
            "faithfulness": (
                res.verification.faithfulness if res.verification else None
            ),
            "citation_accuracy": (
                res.verification.citation_accuracy if res.verification else None
            ),
            "mean_support_score": (
                res.verification.mean_support_score if res.verification else None
            ),
            "supported_rate": (
                res.verification.supported_rate if res.verification else None
            ),
            "weak_support_rate": (
                res.verification.weak_support_rate if res.verification else None
            ),
            "unsupported_rate": (
                res.verification.unsupported_rate if res.verification else None
            ),
            "average_latency_ms": res.avg_latency_ms,
            "p95_latency_ms": res.p95_latency_ms,
        }
        if res.not_computed:
            m["not_computed"] = dict(res.not_computed)
        metrics[key] = m

    summary = {
        "questions_evaluated": len(resolved),
        "questions_skipped": skipped,
        "chunks_in_corpus": judged_chunks,
        "relevant_chunks_annotated": sum(q.n_relevant for q in resolved),
        "gains": compute_gains(results),
        "best_configuration": find_best_configuration(results),
        "tradeoffs_precision_latency": analyze_tradeoff(results),
    }

    return {
        "metadata": {
            "project": "RAG Citation Validator",
            "framework": "evaluate_pipeline.py",
            "generated_at_utc": now,
            "command": "python evaluate_pipeline.py " + " ".join(sys.argv[1:]),
            "traceability": {
                "chunks_sha256": sha256_of(CHUNKS_PATH),
                "annotations_sha256": sha256_of(Path(args.annotations)),
                "annotations_file": str(args.annotations),
                "versions": versions,
            },
            "parameters": {
                "top_k_retrieval": TOP_K_RETRIEVAL,
                "rerank_pool_size": RERANK_POOL_SIZE,
                "generation_sources": GENERATION_SOURCES,
            },
        },
        "configurations": configurations,
        "metrics": metrics,
        "summary": summary,
    }

# ---------------------------------------------------------------------------
# Écrivains : rapport Markdown
# ---------------------------------------------------------------------------

def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    """Rédige le rapport Markdown lisible (tableau, gains, trade-offs)."""
    metrics = report["metrics"]
    summary = report["summary"]
    meta = report["metadata"]
    keys = [k for k in CONFIG_ORDER if k in metrics]

    def pct(key: str, metric: str) -> str:
        val = metrics[key].get(metric)
        return "N/A" if val is None else f"{val * 100:.1f}%"

    def ms(key: str, metric: str) -> str:
        val = metrics[key].get(metric)
        return "N/A" if val is None else f"{val:,.1f}"

    lines: list[str] = []
    lines.append("# Rapport d'évaluation comparative — RAG Citation Validator")
    lines.append("")
    lines.append(f"*Généré le {meta['generated_at_utc']} par `{meta['framework']}`.*")
    lines.append("")
    lines.append("## 1. Périmètre et traçabilité")
    lines.append("")
    trace = meta["traceability"]
    lines.append(f"- **Questions évaluées** : {summary['questions_evaluated']} "
                 f"(+{len(summary['questions_skipped'])} slot(s) d'annotation non rempli(s))")
    lines.append(f"- **Chunks indexés dans le corpus** : {summary['chunks_in_corpus']}")
    lines.append(f"- **Chunks pertinents annotés** : {summary['relevant_chunks_annotated']}")
    lines.append(f"- SHA-256 `chunks.json` : `{trace['chunks_sha256'][:16]}…`")
    lines.append(f"- SHA-256 annotations : `{trace['annotations_sha256'][:16]}…`")
    lines.append("")

    lines.append("## 2. Tableau comparatif complet")
    lines.append("")
    lines.append("| Métrique | " + " | ".join(CONFIG_LABELS[k] for k in keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for metric in ("recall_at_3", "recall_at_5", "recall_at_10", "mrr",
                   "faithfulness", "citation_accuracy", "mean_support_score",
                   "supported_rate", "weak_support_rate", "unsupported_rate"):
        lines.append(f"| {metric} | "
                     + " | ".join(pct(k, metric) for k in keys) + " |")
    lines.append("| **average_latency_ms** | "
                 + " | ".join(ms(k, "average_latency_ms") for k in keys) + " |")
    lines.append("| **p95_latency_ms** | "
                 + " | ".join(ms(k, "p95_latency_ms") for k in keys) + " |")
    lines.append("")
    notes = [
        f"- `{key}` : " + " ; ".join(
            f"{m} → {reason}"
            for m, reason in metrics[key].get("not_computed", {}).items()
        )
        for key in keys if metrics[key].get("not_computed")
    ]
    if notes:
        lines.append("**Métriques explicitement non calculables :**")
        lines.extend(notes)
        lines.append("")

    lines.append("## 3. Analyse automatique des gains")
    lines.append("")
    for gain in summary["gains"]:
        lines.append(f"### {gain['description']} "
                     f"({gain['from_config']} → {gain['to_config']})")
        lines.append("")
        for metric, entry in gain["metrics"].items():
            src_txt = "N/A" if entry["from"] is None else f"{entry['from'] * 100:.1f}%"
            dst_txt = "N/A" if entry["to"] is None else f"{entry['to'] * 100:.1f}%"
            delta = ""
            if entry["delta_abs"] is not None:
                sign = "+" if entry["delta_abs"] >= 0 else ""
                delta = f" (**{sign}{entry['delta_abs'] * 100:.1f} pts**"
                if entry["delta_pct"] is not None:
                    delta += f", {entry['delta_pct']:+.1f} %"
                delta += ")"
            lines.append(f"- `{metric}` : {src_txt} → {dst_txt}{delta}")
        lat = gain.get("latency")
        if lat:
            lines.append(
                f"- Latence moyenne : {lat['from_ms']:,.1f} ms → "
                f"**{lat['to_ms']:,.1f} ms** (Δ {lat['delta_ms']:+,.1f} ms)"
            )
        lines.append("")

    best = summary["best_configuration"]
    lines.append("## 4. Meilleure configuration")
    lines.append("")
    lines.append(f"- **Verdict global qualité** : **{best['overall_best_quality']}**")
    lines.append(f"- **Configuration la plus rapide** : {best['fastest_config']}")
    for metric, key in best["best_per_quality_metric"].items():
        lines.append(f"- Meilleur `{metric}` : {CONFIG_LABELS.get(key or '', 'N/A')}")
    lines.append("")

    lines.append("## 5. Trade-offs précision / latence")
    lines.append("")
    lines.append("| Étape | Δ Recall@5 (pts) | Δ latence moy. (ms) | ms par point de Recall@5 |")
    lines.append("|---|---|---|---|")
    step_labels = {"bm25": "B − A (BM25)", "reranker": "C − B (Reranker)",
                   "nli": "D − C (Génération + NLI)"}
    for step, label in step_labels.items():
        t = summary["tradeoffs_precision_latency"].get(step)
        if not t:
            continue
        cost = t["latency_ms_per_recall_point"]
        cost_txt = cost if isinstance(cost, str) else f"{cost:,.1f}"
        lines.append(
            f"| {label} | {t['delta_recall_at_5_points']:+.2f} | "
            f"{t['delta_avg_latency_ms']:+,.1f} | {cost_txt} |"
        )
    lines.append("")
    lines.append("## 6. Limites méthodologiques")
    lines.append("")
    lines.append("- Jeu de 12 questions annotées à la main : ordres de grandeur "
                 "fiables, pas un benchmark statistiquement significatif.")
    lines.append("- Réponses de la config D générées par extraction déterministe "
                 "(pas de serveur LLM) : la faithfulness mesure la chaîne "
                 "retrieval → citations, pas la paraphrase d'un LLM.")
    lines.append("- Latences mesurées sur CPU local : les valeurs absolues ne sont "
                 "pas comparables entre machines, les deltas relatifs le sont.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"=> {output_path}")


# ---------------------------------------------------------------------------
# Graphique comparatif (matplotlib, sans serveur graphique)
# ---------------------------------------------------------------------------

CONFIG_PLOT_COLORS: dict[str, str] = {
    "A": "#4C72B0", "B": "#55A868", "C": "#DD8452", "D": "#C44E52",
}


def write_comparison_plot(report: dict[str, Any], output_path: Path) -> None:
    """Génère ``comparison_plot.png`` : 6 graphiques comparatifs.

    Args:
        report: rapport complet produit par ``build_report_dict``.
        output_path: chemin du PNG de sortie.
    """
    metrics = report["metrics"]
    keys = [k for k in CONFIG_ORDER if k in metrics]
    labels_short = {"A": "A\nDense", "B": "B\nHybrid",
                    "C": "C\n+Rerank", "D": "D\n+Verification"}

    panels: list[tuple[str, str, bool]] = [
        ("recall_at_3", "Recall@3", True),
        ("recall_at_5", "Recall@5", True),
        ("recall_at_10", "Recall@10", True),
        ("mrr", "MRR", True),
        ("citation_accuracy", "Citation Accuracy", True),
        ("average_latency_ms", "Average Latency (ms)", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes_flat = axes.flatten()

    for ax, (metric, title, higher_better) in zip(axes_flat, panels):
        values = [metrics[k].get(metric) for k in keys]
        colors = [
            CONFIG_PLOT_COLORS.get(k, "#888888")
            if v is not None else "#CCCCCC"
            for k, v in zip(keys, values)
        ]
        plot_vals = [0.0 if v is None else float(v) for v in values]
        bars = ax.bar(range(len(keys)), plot_vals, color=colors, width=0.6)

        numeric = [v for v in values if v is not None]
        if numeric:
            best_val = max(numeric) if higher_better else min(numeric)
            for bar, val in zip(bars, values):
                if val is None:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                            "N/A", ha="center", fontsize=9, color="#777777")
                    continue
                weight = "bold" if val == best_val else "normal"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:,.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight=weight)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels_short[k] for k in keys], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.18)

    fig.suptitle(
        "RAG Citation Validator — Comparative Evaluation (A/B/C/D)",
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"=> {output_path}")


# ---------------------------------------------------------------------------
# Résumé console final
# ---------------------------------------------------------------------------

def print_final_summary(
    report: dict[str, Any],
    results: dict[str, ConfigResult],
    logger: logging.Logger,
) -> None:
    """Affiche le résumé final demandé et les conclusions automatiques."""
    metrics = report["metrics"]
    summary = report["summary"]
    keys = [k for k in CONFIG_ORDER if k in metrics]

    def pct(key: str, metric: str) -> str:
        val = metrics[key].get(metric)
        return "N/A" if val is None else f"{val * 100:.1f}%"

    log_event(logger, "final_summary_started")
    print("")
    print("=" * 72)
    print("RÉSUMÉ DE L'ÉVALUATION COMPARATIVE")
    print("=" * 72)
    print(f"Questions évaluées          : {summary['questions_evaluated']}")
    print(f"Slots d'annotation écartés  : {len(summary['questions_skipped'])} "
          "(non remplis — voir rapport JSON)")
    print(f"Chunks indexés (corpus)     : {summary['chunks_in_corpus']}")
    print(f"Chunks pertinents annotés   : {summary['relevant_chunks_annotated']}")
    print(f"Chunks jugés sur cette run  : "
          f"{summary.get('chunks_judged_across_queries', 'n/a')}")
    print("-" * 72)
    header = f"{'':26s}" + "".join(f"{k:>11s}" for k in keys)
    print(header)
    for metric in ("recall_at_5", "mrr", "citation_accuracy"):
        row = f"{metric:<26s}"
        for k in keys:
            row += f"{pct(k, metric):>11s}"
        print(row)
    lat_row = f"{'avg_latency_ms':<26s}"
    for k in keys:
        val = metrics[k]["average_latency_ms"]
        lat_row += f"{(f'{val:,.1f}' if val is not None else 'N/A'):>11s}"
    print(lat_row)
    p95_row = f"{'p95_latency_ms':<26s}"
    for k in keys:
        val = metrics[k]["p95_latency_ms"]
        p95_row += f"{(f'{val:,.1f}' if val is not None else 'N/A'):>11s}"
    print(p95_row)
    print("=" * 72)

    # Conclusions automatiques.
    best_step, best_gain_pts = None, float("-inf")
    for gain in summary["gains"]:
        entry = gain["metrics"].get("recall_at_5")
        if entry and entry.get("delta_abs") is not None:
            if entry["delta_abs"] > best_gain_pts:
                best_gain_pts = entry["delta_abs"]
                best_step = gain
    if best_step:
        lat = best_step.get("latency") or {}
        delta_lat = lat.get("delta_ms")
        msg = (f"CONCLUSION 1 | Plus grand gain : {best_step['description']} "
               f"-> {best_gain_pts * 100:+.1f} pts de Recall@5"
               + (f", coût {delta_lat:+,.1f} ms de latence moyenne."
                  if delta_lat is not None else "."))
        print(msg)
    else:
        print("CONCLUSION 1 | Aucun gain de Recall@5 mesurable entre étapes.")

    reranker_t = summary["tradeoffs_precision_latency"].get("reranker")
    if reranker_t:
        cost = reranker_t["latency_ms_per_recall_point"]
        cost_txt = (f"~{cost:,.0f} ms par point de Recall@5 gagné"
                    if isinstance(cost, float) else str(cost))
        print(f"CONCLUSION 2 | Coût latence du reranker : "
              f"{reranker_t['delta_avg_latency_ms']:+,.1f} ms en moyenne, "
              f"{cost_txt}.")

    verif = results.get("D")
    vstats = verif.verification if verif else None
    if vstats and vstats.citations_verified:
        measurable = (
            (vstats.supported_rate or 0) > 0
            or (vstats.weak_support_rate or 0) > 0
        )
        verdict = ("OUI - signal NLI discriminant" if measurable
                   else "SIGNAL FAIBLE - aucune citation supportée sur ce jeu")
        print(f"CONCLUSION 3 | Vérification citations mesurable ? {verdict}")
        print(f"             | {vstats.citations_verified} citations vérifiées : "
              f"Supported {vstats.supported_rate * 100:.1f}% / Weak "
              f"{vstats.weak_support_rate * 100:.1f}% / Unsupported "
              f"{vstats.unsupported_rate * 100:.1f}% ; faithfulness "
              f"{(vstats.faithfulness or 0) * 100:.1f}%. La config D ne "
              "change pas le retrieval (même classement que C) : sa valeur "
              "est la traçabilité et la détection des citations non supportées.")
    else:
        print("CONCLUSION 3 | Vérification NLI non mesurable sur ce jeu.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI du framework d'évaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Évaluation comparative des configurations A/B/C/D du pipeline "
            "RAG Citation Validator (ground truth réel, métriques réelles)."
        ),
    )
    parser.add_argument(
        "--annotations", type=str, default=str(ANNOTATIONS_PATH),
        help="Fichier d'annotations (défaut : annotation_template.json).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Répertoire des sorties (défaut : corpus/evaluation/).",
    )
    parser.add_argument(
        "--configs", type=str, default="A,B,C,D",
        help="Configurations à évaluer, ex. 'A,B' (défaut : A,B,C,D).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limiter le nombre de questions évaluées (0 = toutes).",
    )
    parser.add_argument(
        "--log-format", choices=("text", "json"), default="text",
        help="Format des logs (texte lisible ou JSON-lines structuré).",
    )
    args = parser.parse_args(argv)
    requested = [c.strip().upper() for c in args.configs.split(",")]
    unknown = [c for c in requested if c not in CONFIG_ORDER]
    if unknown:
        raise SystemExit(
            f"Configurations inconnues : {unknown}. Attendu parmi {list(CONFIG_ORDER)}."
        )
    if not requested:
        raise SystemExit("Aucune configuration demandée (--configs vide).")
    args.config_list = [c for c in CONFIG_ORDER if c in requested]
    return args


def main() -> None:
    """Point d'entrée : charge ground truth + pipeline, évalue, publie."""
    args = parse_args()
    logger = build_logger(args.log_format)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event(logger, "evaluation_started",
              chunks_file=str(CHUNKS_PATH), annotations=str(args.annotations),
              configs=args.config_list, output_dir=str(output_dir))

    chunks_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = chunks_data["chunks"]
    chunk_index: dict[str, dict[str, Any]] = {
        str(c["chunk_id"]): c for c in chunks
    }
    known_ids = set(chunk_index)

    resolved, skipped = load_annotations(
        Path(args.annotations), known_ids, logger
    )
    if args.limit > 0:
        resolved = resolved[: args.limit]
        log_event(logger, "limit_applied", limit=args.limit,
                  remaining=len(resolved))
    if not resolved:
        raise SystemExit(
            "Aucune question annotée résoluble : vérifiez annotation_template.json "
            "(relevant_chunks doit contenir des chunk_id UUID présents dans chunks.json)."
        )

    engine = HybridSearchEngine(EngineConfig(fetch_k=RERANK_POOL_SIZE + 30))
    try:
        reranker = CrossEncoderReranker()
        verifier: MNLICitationVerifier | None = None
        if "D" in args.config_list:
            verifier = MNLICitationVerifier()
        results = run_evaluation(
            configs=args.config_list,
            questions=resolved,
            engine=engine,
            reranker=reranker,
            verifier=verifier,
            chunk_index=chunk_index,
            logger=logger,
        )
    finally:
        engine.close()

    judged = {
        cid for res in results.values() for q in res.per_question
        for cid in q["retrieved_chunk_ids"]
    }
    report = build_report_dict(
        results=results,
        resolved=resolved,
        skipped=skipped,
        judged_chunks=len(chunk_index),
        args=args,
    )
    report["summary"]["chunks_judged_across_queries"] = len(judged)

    json_path = output_dir / OUTPUT_JSON
    md_path = output_dir / OUTPUT_MD
    png_path = output_dir / OUTPUT_PNG
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"=> {json_path}")
    write_markdown_report(report, md_path)
    write_comparison_plot(report, png_path)

    log_event(logger, "evaluation_completed", outputs={
        "json": str(json_path), "markdown": str(md_path), "plot": str(png_path),
    })
    print_final_summary(report, results, logger)


if __name__ == "__main__":
    main()
