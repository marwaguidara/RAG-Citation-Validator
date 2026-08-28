"""
Indexation dense des chunks dans Qdrant (JOUR 4, étape A).

Input :
    corpus/chunks.json    (produit par chunk_documents.py, validé par
                           validate_chunks.py)

Objectif :
    Charger le modèle d'embedding BAAI/bge-small-en-v1.5, vectoriser tous les
    chunks puis indexer vecteurs + métadonnées dans une collection Qdrant.

Choix techniques :
    - Embeddings NORMALISÉS (L2) + distance COSINE : recommandation officielle
      BGE ; le score cosinus est alors identique au produit scalaire.
    - Les documents sont encodés SANS préfixe : le préfixe « Represent this
      sentence for searching relevant passages: » est réservé aux requêtes,
      au moment de la recherche (étape suivante du pipeline).
    - Point IDs = chunk_id (uuid5 déterministes de chunk_documents.py) :
      relancer l'indexation est idempotent (upserts stables).
    - Qdrant local persistant par défaut (aucun serveur requis) ; bascule vers
      un serveur HTTP via --qdrant-url.

Métadonnées stockées par point (payload) :
    chunk_id, document_id, theme, page_start, page_end
    (+ tokens_est et text par défaut, nécessaires à la génération J7 ;
     --payload-minimal pour restreindre aux cinq champs ci-dessus.)

Sortie :
    corpus/dense_index_report.json   rapport traçable : empreinte SHA-256 de
    l'entrée, modèle, dimensions, device, cible Qdrant, volumes indexés,
    vérification post-indexation, environnement d'exécution.

Code de sortie :
    0   indexation complète et vérifiée (PASS/WARNING)
    1   échec (fichier invalide, erreur d'upsert, compte de points incorrect)

Usage :
    python build_dense_index.py
    python build_dense_index.py --qdrant-url http://localhost:6333
    python build_dense_index.py --batch-size 128 --device cuda
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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
REPORT_FILENAME = "dense_index_report.json"
REFERENCE_FILENAME = "documents.json"

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION_NAME = "chunks_dense"
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_UPLOAD_BATCH_SIZE = 200
VECTOR_STORE_DIRNAME = "vector_store"

REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id", "document_id", "theme",
    "page_start", "page_end", "tokens_est", "text",
)
MINIMAL_PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id", "document_id", "theme", "page_start", "page_end",
)

HNSW_M = 16                # paramètre HNSW explicite => configuration traçable
HNSW_EF_CONSTRUCT = 100
EXAMPLE_SAMPLE_SIZE = 5    # nb de points revérifiés après indexation


class Status(StrEnum):
    """Verdict de l'indexation."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


def build_logger(as_json: bool) -> tuple[logging.Logger, "EventLogger"]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("build_dense_index")
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


def _sentence_transformers_version() -> str:
    """Retourne la version de sentence-transformers installée (traçabilité)."""
    from importlib.metadata import version

    return version("sentence-transformers")


# ---------------------------------------------------------------------------
# Chunks & modèle
# ---------------------------------------------------------------------------

def load_chunks(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Charge chunks.json avec garde-fous d'intégrité minimale.

    Garantit : champs obligatoires présents, chunk_id uniques (un doublon
    écraserait silencieusement un point Qdrant), ordre trié par chunk_id pour
    une exécution déterministe.

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
            f"Format inattendu dans {input_path.name} : clé 'chunks' (liste non vide) requise."
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


def resolve_device(device_arg: str) -> str:
    """Résout le device d'exécution ('auto' => CUDA si disponible, sinon CPU)."""
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """Charge le modèle d'embedding sur le device demandé."""
    return SentenceTransformer(model_name, device=device)


def embed_corpus(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> tuple[Any, float]:
    """Vectorise les textes (embeddings normalisés L2).

    Note : les documents BGE sont encodés SANS préfixe ; le préfixe de requête
    sera appliqué uniquement côté recherche.

    Returns:
        Tuple (matrice numpy [n_chunks, dim], durée en secondes).
    """
    started = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings, time.perf_counter() - started


def build_payload(chunk: dict[str, Any], minimal: bool) -> dict[str, Any]:
    """Construit le payload Qdrant d'un chunk.

    Args:
        chunk: chunk source.
        minimal: si vrai, seuls les cinq champs de traçabilité sont stockés ;
            sinon tokens_est et text sont ajoutés (utiles à la génération J7).
    """
    payload: dict[str, Any] = {
        "chunk_id": str(chunk["chunk_id"]),
        "document_id": str(chunk["document_id"]),
        "theme": str(chunk["theme"]),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
    }
    if not minimal:
        payload["tokens_est"] = int(chunk["tokens_est"])
        payload["text"] = str(chunk["text"])
    return payload


def build_points(
    chunks: list[dict[str, Any]],
    embeddings: Any,
    minimal_payload: bool,
) -> list[models.PointStruct]:
    """Associe chaque chunk à son PointStruct Qdrant (id = chunk_id uuid)."""
    points: list[models.PointStruct] = []
    for index, chunk in enumerate(chunks):
        points.append(models.PointStruct(
            id=str(chunk["chunk_id"]),
            vector=embeddings[index].tolist(),
            payload=build_payload(chunk, minimal_payload),
        ))
    return points


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def init_qdrant_client(
    qdrant_url: str | None, qdrant_path: Path
) -> tuple[QdrantClient, dict[str, Any]]:
    """Initialise le client Qdrant (serveur HTTP ou stockage local persistant).

    Args:
        qdrant_url: URL d'un serveur Qdrant ; si fournie, prime sur le mode local.
        qdrant_path: répertoire de persistance locale.
    """
    if qdrant_url:
        return QdrantClient(url=qdrant_url), {
            "mode": "server", "location": qdrant_url,
        }
    return QdrantClient(path=str(qdrant_path)), {
        "mode": "local", "location": str(qdrant_path),
    }


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    recreate: bool,
) -> tuple[str, Any]:
    """Crée ou recycle la collection dense et retourne son état.

    En mode réutilisation (`recreate=False`), la configuration existante est
    vérifiée : dimension et distance doivent correspondre exactement, sinon
    l'exécution est interrompue (mieux vaut échouer qu'indexer à côté).

    Returns:
        Tuple (action parmi created/recreated/reused, collection_info).
    """
    if client.collection_exists(collection_name):
        if not recreate:
            collection_info = client.get_collection(collection_name)
            vectors = collection_info.config.params.vectors
            if vectors.size != dimension or vectors.distance != models.Distance.COSINE:
                raise SystemExit(
                    f"Collection '{collection_name}' existante incompatible "
                    f"(dim={vectors.size}, distance={vectors.distance}) ; "
                    f"attendu (dim={dimension}, distance=COSINE). "
                    "Relancez sans --reuse-collection."
                )
            return "reused", collection_info
        client.delete_collection(collection_name)
        action = "recreated"
    else:
        action = "created"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT,
        ),
    )
    return action, client.get_collection(collection_name)


def upsert_points(
    events: EventLogger,
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    batch_size: int,
) -> tuple[int, float, list[str]]:
    """Insère les points par lots avec attente de confirmation (wait=True).

    Sémantique fail-stop : la première erreur d'upsert interrompt l'insertion
    (un index partiel doit être visible comme tel dans le rapport).

    Returns:
        Tuple (points insérés, durée en secondes, erreurs éventuelles).
    """
    started = time.perf_counter()
    uploaded = 0
    errors: list[str] = []
    for start in tqdm(
        range(0, len(points), batch_size), desc="upsert", unit="batch", ncols=80
    ):
        batch = points[start:start + batch_size]
        try:
            client.upsert(
                collection_name=collection_name, points=batch, wait=True,
            )
            uploaded += len(batch)
        except Exception as exc:  # noqa: BLE001 — erreur Qdrant => fail-stop tracé
            errors.append(
                f"batch [{start}:{start + len(batch)}] : "
                f"{type(exc).__name__}: {exc}"
            )
            events.error("upsert_failed", batch_start=start, error=str(exc))
            break
    return uploaded, time.perf_counter() - started, errors


def verify_index(
    client: QdrantClient,
    collection_name: str,
    expected_chunk_ids: list[str],
) -> dict[str, Any]:
    """Vérifie le nombre de points indexés et l'intégrité du payload échantillonné."""
    indexed_points = client.count(collection_name=collection_name, exact=True).count
    sample_ids = expected_chunk_ids[:EXAMPLE_SAMPLE_SIZE]
    retrieved = client.retrieve(
        collection_name=collection_name,
        ids=sample_ids,
        with_payload=True,
        with_vectors=False,
    )
    found_ids = sorted(str(point.id) for point in retrieved)
    payload_fields_ok = all(
        set(MINIMAL_PAYLOAD_FIELDS).issubset(point.payload) for point in retrieved
    ) if retrieved else False
    return {
        "expected_points": len(expected_chunk_ids),
        "indexed_points": indexed_points,
        "count_match": indexed_points == len(expected_chunk_ids),
        "sample_size": len(sample_ids),
        "sample_found": found_ids,
        "sample_all_found": len(found_ids) == len(sample_ids),
        "sample_payload_fields_ok": payload_fields_ok,
    }


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration d'exécution (tracée telle quelle dans le rapport)."""

    input_path: Path
    collection_name: str
    model_name: str
    device: str
    embed_batch_size: int
    upload_batch_size: int
    recreate_collection: bool
    minimal_payload: bool
    qdrant_target: dict[str, Any] = field(default_factory=dict)


def build_report(
    config: RunConfig,
    source_payload: dict[str, Any],
    chunk_count: int,
    dimension: int,
    max_seq_tokens: int,
    embed_seconds: float,
    upsert_seconds: float,
    points_uploaded: int,
    upload_errors: list[str],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Assemble le rapport d'indexation dense (sérialisable en JSON)."""
    verification_failed = bool(
        upload_errors
        or not verification["count_match"]
        or not verification["sample_all_found"]
        or not verification["sample_payload_fields_ok"]
    )
    return {
        "generator": "build_dense_index.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": Status.FAIL.value if verification_failed else Status.PASS.value,
        "input": {
            "file": config.input_path.name,
            "size_bytes": config.input_path.stat().st_size,
            "sha256": sha256_of(config.input_path),
            "chunks": chunk_count,
            "generator": source_payload.get("generator"),
            "generated_at": source_payload.get("generated_at"),
        },
        "model": {
            "name": config.model_name,
            "dimension": dimension,
            "max_seq_tokens": max_seq_tokens,
            "normalized_embeddings": True,
            "query_prefix_applied": False,
            "device": config.device,
            "sentence_transformers_version": _sentence_transformers_version(),
            "torch_version": torch.__version__,
        },
        "qdrant": {
            **config.qdrant_target,
            "collection": config.collection_name,
            "distance": models.Distance.COSINE.value,
            "hnsw": {"m": HNSW_M, "ef_construct": HNSW_EF_CONSTRUCT},
        },
        "indexing": {
            "embed_batch_size": config.embed_batch_size,
            "upload_batch_size": config.upload_batch_size,
            "recreate_collection": config.recreate_collection,
            "minimal_payload": config.minimal_payload,
            "payload_fields": list(MINIMAL_PAYLOAD_FIELDS)
            + ([] if config.minimal_payload else ["tokens_est", "text"]),
            "points_expected": chunk_count,
            "points_uploaded": points_uploaded,
            "upload_errors": upload_errors,
            "durations_seconds": {
                "embedding": round(embed_seconds, 2),
                "upsert": round(upsert_seconds, 2),
            },
            "throughput_chunks_per_second": (
                round(chunk_count / embed_seconds, 1) if embed_seconds > 0 else None
            ),
        },
        "verification": verification,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_report(corpus_dir: Path, report: dict[str, Any]) -> Path:
    """Écrit dense_index_report.json sous corpus/ et retourne son chemin."""
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
    """Parse les arguments CLI de l'indexeur dense."""
    parser = argparse.ArgumentParser(
        description=(
            "Indexe les chunks dans Qdrant avec BAAI/bge-small-en-v1.5 "
            "(embeddings normalisés + COSINE) et produit dense_index_report.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Codes de sortie : 0 = index vérifié · 1 = échec\n"
            "Exemples :\n"
            "  python build_dense_index.py\n"
            "  python build_dense_index.py --qdrant-url http://localhost:6333\n"
            "  python build_dense_index.py --reuse-collection --payload-minimal\n"
            "  python build_dense_index.py --limit 50   # smoke test (dev)"
        ),
    )
    parser.add_argument("--input", default=INPUT_FILENAME,
                        help=f"Fichier d'entrée sous corpus/ ({INPUT_FILENAME}).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"Modèle d'embedding ({DEFAULT_MODEL_NAME}).")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME,
                        help=f"Collection Qdrant cible ({DEFAULT_COLLECTION_NAME}).")
    parser.add_argument("--qdrant-url", default=None, metavar="URL",
                        help="URL d'un serveur Qdrant (sinon stockage local).")
    parser.add_argument("--qdrant-path", default=None, metavar="DIR",
                        help="Répertoire du Qdrant local "
                             "(défaut : <projet>/vector_store/qdrant).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device d'encodage (défaut : auto = CUDA si dispo).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
                        metavar="N",
                        help=(f"Taille des lots d'embedding "
                              f"({DEFAULT_EMBED_BATCH_SIZE})."))
    parser.add_argument("--upload-batch-size", type=int,
                        default=DEFAULT_UPLOAD_BATCH_SIZE, metavar="N",
                        help=(f"Taille des lots d'upsert Qdrant "
                              f"({DEFAULT_UPLOAD_BATCH_SIZE})."))
    parser.add_argument("--reuse-collection", action="store_true",
                        help="Ne recrée pas la collection : upsert idempotent sur "
                             "la collection existante (config vérifiée).")
    parser.add_argument("--payload-minimal", action="store_true",
                        help="Stocke uniquement chunk_id/document_id/theme/"
                             "page_start/page_end (sans text ni tokens_est).")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Indexer seulement les N premiers chunks (dev/test).")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Format des logs console (défaut : text).")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : embeddings BGE puis upsert Qdrant + rapport traçable."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.batch_size <= 0 or args.upload_batch_size <= 0:
        raise SystemExit("--batch-size et --upload-batch-size doivent être > 0.")

    input_path = locate_corpus(args.input)
    if input_path is None:
        raise SystemExit(
            f"{args.input} introuvable sous corpus/. Lancez chunk_documents.py."
        )
    chunks, source_payload = load_chunks(input_path)
    if args.limit > 0:
        chunks = chunks[:args.limit]
        events.warning("test_limit_mode", chunks=len(chunks))

    qdrant_path = (
        Path(args.qdrant_path) if args.qdrant_path
        else Path(__file__).resolve().parent.parent / VECTOR_STORE_DIRNAME / "qdrant"
    )
    client, qdrant_target = init_qdrant_client(args.qdrant_url, qdrant_path)

    config = RunConfig(
        input_path=input_path,
        collection_name=args.collection,
        model_name=args.model,
        device=resolve_device(args.device),
        embed_batch_size=args.batch_size,
        upload_batch_size=args.upload_batch_size,
        recreate_collection=not args.reuse_collection,
        minimal_payload=args.payload_minimal,
        qdrant_target=qdrant_target,
    )
    events.info(
        "indexation_started",
        chunks=len(chunks),
        model=config.model_name,
        device=config.device,
        collection=config.collection_name,
        qdrant_mode=qdrant_target["mode"],
    )

    try:
        model = load_embedding_model(config.model_name, config.device)
        dimension = int(model.get_sentence_embedding_dimension())
        max_seq_tokens = int(model.max_seq_length)
        events.info(
            "model_loaded",
            dimension=dimension,
            max_seq_tokens=max_seq_tokens,
            normalized_embeddings=True,
        )

        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings, embed_seconds = embed_corpus(
            model, texts, config.embed_batch_size,
        )

        action, _collection_info = ensure_collection(
            client, config.collection_name, dimension, config.recreate_collection,
        )
        events.info("collection_ready", action=action)

        points = build_points(chunks, embeddings, config.minimal_payload)
        points_uploaded, upsert_seconds, upload_errors = upsert_points(
            events, client, config.collection_name, points, config.upload_batch_size,
        )

        verification = verify_index(
            client,
            config.collection_name,
            [str(chunk["chunk_id"]) for chunk in chunks],
        )
    finally:
        client.close()

    report = build_report(
        config=config,
        source_payload=source_payload,
        chunk_count=len(chunks),
        dimension=dimension,
        max_seq_tokens=max_seq_tokens,
        embed_seconds=embed_seconds,
        upsert_seconds=upsert_seconds,
        points_uploaded=points_uploaded,
        upload_errors=upload_errors,
        verification=verification,
    )
    report_path = write_report(input_path.parent, report)

    status = Status(report["status"])
    events.info(
        "verification_completed",
        expected_points=verification["expected_points"],
        indexed_points=verification["indexed_points"],
        count_match=verification["count_match"],
        sample_all_found=verification["sample_all_found"],
    )
    events.info(
        "indexation_completed",
        status=status.value,
        points_uploaded=points_uploaded,
        collection=config.collection_name,
        report=str(report_path),
    )

    sys.exit(EXIT_OK if status is not Status.FAIL else EXIT_FAIL)


if __name__ == "__main__":
    main()