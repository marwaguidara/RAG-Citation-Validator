"""
Recherche hybride dense + BM25 avec fusion Reciprocal Rank Fusion (JOUR 5, étape B).

Inputs :
    corpus/chunks.json     (métadonnées canoniques : source de vérité)
    corpus/bm25_index.pkl  (artefact produit par build_bm25_index.py)
    Collection Qdrant      (produite par build_dense_index.py)

Objectif :
    Fournir la fonction principale `search(query, top_k=10)` qui :
      1. encode la requête avec BAAI/bge-small-en-v1.5 (+ préfixe de requête BGE,
         appliqué UNIQUEMENT côté requête, jamais sur les documents) ;
      2. interroge Qdrant (recherche dense, similarité cosinus) ;
      3. interroge l'index BM25 (scores Okapi) ;
      4. fusionne les deux classements par Reciprocal Rank Fusion ;
      5. retourne le classement fusionné avec les scores par canal et les
         latences détaillées (dense / BM25 / fusion / totale).

Cohérence des canaux (points critiques hérités des étapes précédentes) :
    - La tokenisation de la requête est IMPORTÉE de build_bm25_index.py
      (source unique de vérité) et alignée sur la config stockée dans le
      pickle : mêmes jetons => scores valides.
    - Les point IDs Qdrant sont les chunk_id uuid5 : chaque résultat dense est
      réconciliable avec le canal lexical sans ambiguïté.

Contraintes respectées : pas de reranking, pas de génération de réponse,
pas d'API HTTP. Ce module s'arrête au classement hybride fusionné — entrée
directe du reranker BGE (JOUR 7).

Usage programme :
    from hybrid_search import search
    response = search("What is retrieval augmented generation?", top_k=5)
    response.results[0].chunk_id ; response.latencies_ms.total_ms

Usage CLI :
    python hybrid_search.py                       # 3 requêtes de test intégrées
    python hybrid_search.py --top-k 5 --fetch-k 100 "How does ReAct work?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Source unique de vérité pour la tokenisation BM25 (index ET requêtes).
from build_bm25_index import ENGLISH_STOPWORDS, TOKEN_PATTERN

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
BM25_INDEX_FILENAME = "bm25_index.pkl"
REPORT_FILENAME = "hybrid_search_report.json"

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION_NAME = "chunks_dense"
DEFAULT_TOP_K = 10
DEFAULT_FETCH_K = 50           # profondeur de candidats par canal avant fusion
RRF_K = 60                     # constante standard de la Reciprocal Rank Fusion
BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

DEFAULT_TEST_QUERIES: tuple[str, ...] = (
    "What is retrieval augmented generation?",
    "How does ReAct work?",
    "What is LoRA fine tuning?",
)


class Status(StrEnum):
    """Verdict de la session de recherche."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


def build_logger(as_json: bool) -> tuple[logging.Logger, "EventLogger"]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("hybrid_search")
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


def tokenize_for_bm25(text: str, use_stopwords: bool = True) -> list[str]:
    """Tokenise une requête à l'identique du tokenizer de build_bm25_index.py.

    La liste de stopwords et le pattern sont IMPORTÉS de build_bm25_index.py :
    source unique de vérité entre l'indexation et la recherche.
    """
    tokens = TOKEN_PATTERN.findall(text.lower())
    if use_stopwords:
        return [token for token in tokens if token not in ENGLISH_STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SearchResult:
    """Un chunk du classement fusionné, avec ses scores par canal."""

    rank: int
    chunk_id: str
    document_id: str
    theme: str
    page_start: int
    page_end: int
    score_dense: float | None   # None si absent du canal dense
    score_bm25: float | None    # None si absent du canal lexical
    rrf_score: float
    channels: list[str]         # ["dense", "bm25"] : canaux qui l'ont retourné
    text_preview: str = ""      # 160 premiers caractères (aide à la lecture)


@dataclass(slots=True)
class LatencyBreakdown:
    """Latences détaillées d'une recherche hybride, en millisecondes."""

    query_encoding_ms: float
    dense_ms: float
    bm25_ms: float
    fusion_ms: float
    total_ms: float


@dataclass(slots=True)
class HybridResponse:
    """Réponse complète d'une recherche hybride."""

    query: str
    results: list[SearchResult]
    latencies_ms: LatencyBreakdown


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Configuration d'exécution du moteur hybride (tracée dans le rapport)."""

    model_name: str = DEFAULT_MODEL_NAME
    collection_name: str = DEFAULT_COLLECTION_NAME
    device: str = "auto"
    qdrant_url: str | None = None
    qdrant_path: str | None = None
    fetch_k: int = DEFAULT_FETCH_K
    rrf_k: int = RRF_K
    corpus_sha256: str | None = None
    bm25_sha256: str | None = None


# ---------------------------------------------------------------------------
# Chargement des artefacts
# ---------------------------------------------------------------------------

def load_chunk_metadata(
    input_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Charge chunks.json comme source de vérité des métadonnées.

    Returns:
        Tuple (mapping chunk_id -> chunk, métadonnées du payload source).
    """
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    chunks: list[dict[str, Any]] = payload.get("chunks", [])
    if not chunks:
        raise SystemExit(f"{input_path.name} ne contient aucun chunk.")
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise SystemExit("chunk_id dupliqués dans chunks.json : lancez validate_chunks.py.")
    return by_id, payload


def load_bm25_artifact(index_path: Path) -> tuple[Any, list[str], bool]:
    """Charge l'artefact BM25 et valide son format minimal.

    Returns:
        Tuple (objet BM25Okapi, chunk_ids dans l'ordre de l'index,
        suppression de stopwords active ?).
    """
    with index_path.open("rb") as handle:
        # NB : pickle => ne charger que des artefacts produits localement.
        artifact = pickle.load(handle)
    if artifact.get("format_version") != 1 or "bm25" not in artifact:
        raise SystemExit(f"Artefact {index_path.name} invalide ou non supporté.")
    tokenizer_cfg = artifact.get("tokenizer", {})
    return (
        artifact["bm25"],
        list(artifact.get("chunk_ids", [])),
        bool(tokenizer_cfg.get("use_stopwords", True)),
    )


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]], rrf_k: int
) -> dict[str, float]:
    """Fusionne plusieurs classements par Reciprocal Rank Fusion.

    score_rrf(d) = somme sur les canaux de 1 / (rrf_k + rang(d))

    Args:
        ranked_lists: mapping canal -> chunk_ids classés (meilleur en premier).
        rrf_k: constante d'amortissement (60 = valeur standard de la littérature).
    """
    fused: dict[str, float] = {}
    for channel in sorted(ranked_lists):
        for position, chunk_id in enumerate(ranked_lists[channel], start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + position)
    return fused


# ---------------------------------------------------------------------------
# Moteur hybride
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """Moteur de recherche hybride dense (Qdrant) + lexical (BM25).

    Charge une seule fois les artefacts coûteux (modèle BGE, client Qdrant,
    index BM25 pickle, métadonnées chunks.json) puis sert des recherches.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        corpus_path = locate_corpus(INPUT_FILENAME)
        if corpus_path is None:
            raise SystemExit(f"{INPUT_FILENAME} introuvable sous corpus/.")
        bm25_path = locate_corpus(BM25_INDEX_FILENAME)
        if bm25_path is None:
            raise SystemExit(
                f"{BM25_INDEX_FILENAME} introuvable. Lancez build_bm25_index.py."
            )

        self._chunk_by_id, _payload = load_chunk_metadata(corpus_path)
        self._bm25, bm25_chunk_ids, self._bm25_use_stopwords = (
            load_bm25_artifact(bm25_path)
        )
        # Position i de l'index BM25 => chunk_ids[i] : mapping inverse pour
        # résoudre un rang BM25 vers son chunk_id.
        self._bm25_position_to_id = {
            position: chunk_id for position, chunk_id in enumerate(bm25_chunk_ids)
        }
        missing = set(bm25_chunk_ids) - set(self._chunk_by_id)
        if missing:
            raise SystemExit(
                f"{len(missing)} chunk(s) de l'index BM25 absents de chunks.json."
            )

        device = config.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover — torch est requis par ST
                device = "cpu"
        self._model = SentenceTransformer(config.model_name, device=device)
        if config.qdrant_url:
            self._client = QdrantClient(url=config.qdrant_url)
        else:
            local_path = Path(
                config.qdrant_path
                or str(Path(__file__).resolve().parent.parent / "vector_store" / "qdrant")
            )
            self._client = QdrantClient(path=local_path)

    def close(self) -> None:
        """Libère le client Qdrant (verrous du mode local)."""
        self._client.close()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> HybridResponse:
        """Exécute le pipeline hybride complet pour une requête.

        Pipeline : encodage requête (préfixe BGE) -> recherche dense -> recherche
        BM25 -> fusion RRF -> classement final déterministe.

        Args:
            query: question en langage naturel.
            top_k: nombre de résultats fusionnés à retourner.
        """
        started_total = time.perf_counter()

        started = time.perf_counter()
        query_vector = self._model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        encoding_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dense_hits = self._client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector.tolist(),
            limit=self.config.fetch_k,
            with_payload=True,
            with_vectors=False,
        ).points
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        query_tokens = tokenize_for_bm25(query, self._bm25_use_stopwords)
        bm25_scores = self._bm25.get_scores(query_tokens)
        ranked_positions = sorted(
            range(len(bm25_scores)), key=lambda pos: -bm25_scores[pos]
        )[: self.config.fetch_k]
        bm25_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dense_scores: dict[str, float] = {
            str(point.id): float(point.score) for point in dense_hits
        }
        bm25_by_id: dict[str, float] = {
            self._bm25_position_to_id[position]: float(bm25_scores[position])
            for position in ranked_positions
        }
        rrf_scores = reciprocal_rank_fusion(
            {
                "dense": list(dense_scores.keys()),
                "bm25": [self._bm25_position_to_id[p] for p in ranked_positions],
            },
            self.config.rrf_k,
        )
        ordered_ids = sorted(
            rrf_scores, key=lambda chunk_id: (-rrf_scores[chunk_id], chunk_id)
        )[:top_k]

        results = [
            self._build_result(
                rank + 1, chunk_id, dense_scores, bm25_by_id, rrf_scores[chunk_id],
            )
            for rank, chunk_id in enumerate(ordered_ids)
        ]
        fusion_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - started_total) * 1000
        return HybridResponse(
            query=query,
            results=results,
            latencies_ms=LatencyBreakdown(
                query_encoding_ms=round(encoding_ms, 2),
                dense_ms=round(dense_ms, 2),
                bm25_ms=round(bm25_ms, 2),
                fusion_ms=round(fusion_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )

    def _build_result(
        self,
        rank: int,
        chunk_id: str,
        dense_scores: dict[str, float],
        bm25_scores: dict[str, float],
        rrf_score: float,
    ) -> SearchResult:
        """Construit un SearchResult enrichi depuis la source de vérité."""
        chunk = self._chunk_by_id[chunk_id]
        channels = [
            channel for channel, scores in (
                ("dense", dense_scores), ("bm25", bm25_scores),
            )
            if chunk_id in scores
        ]
        return SearchResult(
            rank=rank,
            chunk_id=chunk_id,
            document_id=str(chunk["document_id"]),
            theme=str(chunk["theme"]),
            page_start=int(chunk["page_start"]),
            page_end=int(chunk["page_end"]),
            score_dense=dense_scores.get(chunk_id),
            score_bm25=bm25_scores.get(chunk_id),
            rrf_score=round(rrf_score, 6),
            channels=channels,
            text_preview=str(chunk.get("text", ""))[:160].replace("\n", " "),
        )


_engine: HybridSearchEngine | None = None


def get_engine(config: EngineConfig | None = None) -> HybridSearchEngine:
    """Retourne l'instance partagée du moteur (initialisation paresseuse)."""
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine(config or EngineConfig())
    return _engine


def search(query: str, top_k: int = DEFAULT_TOP_K) -> HybridResponse:
    """Fonction principale : recherche hybride dense + BM25 fusionnée par RRF.

    Args:
        query: question en langage naturel.
        top_k: nombre de résultats fusionnés à retourner.

    Returns:
        HybridResponse : résultats triés (chunk_id, document_id, theme, pages,
        score_dense, score_bm25, score_rrf) + latences détaillées.
    """
    return get_engine().search(query=query, top_k=top_k)


# ---------------------------------------------------------------------------
# Rapport de session
# ---------------------------------------------------------------------------

def aggregate_latencies(responses: list[HybridResponse]) -> dict[str, float]:
    """Agrège les latences de toutes les requêtes (moyenne et maximum)."""
    totals = sorted(r.latencies_ms.total_ms for r in responses)
    count = len(totals)
    if not count:
        return {"queries": 0}
    return {
        "queries": count,
        "mean_total_ms": round(sum(totals) / count, 2),
        "max_total_ms": totals[-1],
        "mean_dense_ms": round(
            sum(r.latencies_ms.dense_ms for r in responses) / count, 2
        ),
        "mean_bm25_ms": round(
            sum(r.latencies_ms.bm25_ms for r in responses) / count, 2
        ),
        "mean_fusion_ms": round(
            sum(r.latencies_ms.fusion_ms for r in responses) / count, 2
        ),
        "mean_query_encoding_ms": round(
            sum(r.latencies_ms.query_encoding_ms for r in responses) / count, 2
        ),
    }


def assess_reranker_readiness(
    responses: list[HybridResponse], fetch_k: int
) -> dict[str, Any]:
    """Vérifie que la sortie hybride est prête pour le reranker BGE (JOUR 7).

    Contrôles : chaque requête retourne des résultats ; le texte est disponible
    pour tous les résultats (requis par le cross-encoder) ; les deux canaux ont
    produit suffisamment de candidats.
    """
    all_results = [result for response in responses for result in response.results]
    min_dense = min(
        (sum(1 for r in resp.results if r.score_dense is not None) for resp in responses),
        default=0,
    )
    min_bm25 = min(
        (sum(1 for r in resp.results if r.score_bm25 is not None) for resp in responses),
        default=0,
    )
    text_available = all(bool(result.text_preview) for result in all_results)
    return {
        "ready_for_reranking": bool(all_results) and text_available,
        "all_queries_returned_results": bool(responses) and all(
            len(response.results) > 0 for response in responses
        ),
        "text_available_for_all_results": text_available,
        "min_dense_candidates": min_dense,
        "min_bm25_candidates": min_bm25,
        "results_total": len(all_results),
    }


def build_session_report(
    config: EngineConfig,
    queries: list[str],
    responses: list[HybridResponse],
    chunks_sha256: str | None,
) -> dict[str, Any]:
    """Assemble hybrid_search_report.json (sérialisable en JSON)."""
    from importlib.metadata import version

    status = Status.PASS.value if (
        len(responses) == len(queries)
        and queries
        and all(response.results for response in responses)
    ) else Status.FAIL.value
    return {
        "generator": "hybrid_search.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "config": {
            **asdict(config),
            "bge_query_prefix_applied_to_queries_only": True,
            "tokenizer_source": "build_bm25_index.py",
        },
        "indexes": {
            "dense_collection": config.collection_name,
            "bm25_index_sha256": config.bm25_sha256,
            "chunks_sha256": chunks_sha256,
        },
        "versions": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sentence_transformers_version": version("sentence-transformers"),
        },
        "queries": [
            {
                "query": response.query,
                "latencies_ms": asdict(response.latencies_ms),
                "results": [asdict(result) for result in response.results],
            }
            for response in responses
        ],
        "aggregate_latencies_ms": aggregate_latencies(responses),
        "reranker_readiness": assess_reranker_readiness(responses, config.fetch_k),
    }


def write_session_report(corpus_dir: Path, report: dict[str, Any]) -> Path:
    """Écrit hybrid_search_report.json sous corpus/ et retourne son chemin."""
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
    """Parse les arguments CLI de la session de recherche hybride."""
    parser = argparse.ArgumentParser(
        description=(
            "Recherche hybride dense + BM25 fusionnée par RRF "
            "(rapport : hybrid_search_report.json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sans requête fournie, exécute les 3 requêtes de test du projet.\n"
            "Exemples :\n"
            '  python hybrid_search.py "How does ReAct work?"\n'
            '  python hybrid_search.py --top-k 5 "What is LoRA fine tuning?"'
        ),
    )
    parser.add_argument("queries", nargs="*", default=None,
                        help="Requêtes à exécuter (défaut : 3 requêtes de test).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, metavar="N",
                        help=f"Résultats fusionnés retournés ({DEFAULT_TOP_K}).")
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K, metavar="N",
                        help=(f"Profondeur de candidats par canal avant fusion "
                              f"({DEFAULT_FETCH_K})."))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"Modèle d'embedding ({DEFAULT_MODEL_NAME}).")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME,
                        help=f"Collection Qdrant ({DEFAULT_COLLECTION_NAME}).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device d'encodage des requêtes.")
    parser.add_argument("--qdrant-url", default=None, metavar="URL",
                        help="URL d'un serveur Qdrant (sinon stockage local).")
    parser.add_argument("--qdrant-path", default=None, metavar="DIR",
                        help="Répertoire du Qdrant local.")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Format des logs console (défaut : text).")
    return parser.parse_args()


def _console_safe(text: str) -> str:
    """Rend un texte affichable sur la console courante (Windows cp1252 inclus).

    Les caractères non représentables sont remplacés au lieu de lever une
    UnicodeEncodeError ; le rapport JSON reste lui en UTF-8 intégral.
    """
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_results(response: HybridResponse) -> None:
    """Affiche lisiblement les résultats d'une requête dans la console."""
    print(_console_safe(f"\nQUERY: {response.query}"))
    print(
        f"  latences ms -> encodage {response.latencies_ms.query_encoding_ms}"
        f" | dense {response.latencies_ms.dense_ms}"
        f" | bm25 {response.latencies_ms.bm25_ms}"
        f" | fusion {response.latencies_ms.fusion_ms}"
        f" | TOTAL {response.latencies_ms.total_ms}"
    )
    for result in response.results[:3]:
        dense = "-" if result.score_dense is None else f"{result.score_dense:.4f}"
        sparse = "-" if result.score_bm25 is None else f"{result.score_bm25:.2f}"
        print(
            f"  #{result.rank} [{'+'.join(result.channels)}] "
            f"{result.chunk_id[:8]} {result.document_id} "
            f"p.{result.page_start}-{result.page_end} | rrf={result.rrf_score:.4f} "
            f"dense={dense} bm25={sparse}"
        )
        print(f"      {_console_safe(result.text_preview[:110])}")


def main() -> None:
    """Point d'entrée : exécute les requêtes et écrit le rapport de session."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.top_k <= 0 or args.fetch_k <= 0:
        raise SystemExit("--top-k et --fetch-k doivent être > 0.")
    if args.fetch_k < args.top_k:
        raise SystemExit("--fetch-k doit être >= --top-k (candidats par canal).")

    queries = list(args.queries) if args.queries else list(DEFAULT_TEST_QUERIES)
    corpus_path = locate_corpus(INPUT_FILENAME)
    if corpus_path is None:
        raise SystemExit(f"{INPUT_FILENAME} introuvable sous corpus/.")
    chunks_sha256 = sha256_of(corpus_path)
    bm25_path = locate_corpus(BM25_INDEX_FILENAME)
    if bm25_path is None:
        raise SystemExit(
            f"{BM25_INDEX_FILENAME} introuvable. Lancez build_bm25_index.py."
        )
    bm25_sha256 = sha256_of(bm25_path)

    config = EngineConfig(
        model_name=args.model,
        collection_name=args.collection,
        device=args.device,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        fetch_k=args.fetch_k,
        corpus_sha256=chunks_sha256,
        bm25_sha256=bm25_sha256,
    )

    events.info(
        "session_started",
        queries=len(queries),
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        rrf_k=RRF_K,
        collection=config.collection_name,
    )
    engine = get_engine(config)
    try:
        responses = [engine.search(query=query, top_k=args.top_k) for query in queries]
    finally:
        engine.close()

    for response in responses:
        _print_results(response)

    report = build_session_report(config, queries, responses, chunks_sha256)
    report_path = write_session_report(corpus_path.parent, report)

    aggregate = report["aggregate_latencies_ms"]
    readiness = report["reranker_readiness"]
    events.info(
        "session_completed",
        status=report["status"],
        queries=aggregate.get("queries", 0),
        mean_total_ms=aggregate.get("mean_total_ms"),
        max_total_ms=aggregate.get("max_total_ms"),
    )
    events.info(
        "reranker_readiness",
        ready=readiness["ready_for_reranking"],
        text_available=readiness["text_available_for_all_results"],
        min_dense_candidates=readiness["min_dense_candidates"],
        min_bm25_candidates=readiness["min_bm25_candidates"],
    )
    print(f"\n=> rapport : {report_path}")
    sys.exit(EXIT_OK if report["status"] == Status.PASS.value else EXIT_FAIL)


if __name__ == "__main__":
    main()