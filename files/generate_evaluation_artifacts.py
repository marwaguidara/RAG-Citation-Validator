"""
Génération des artefacts d'évaluation comparative (JOUR 8).

Ce script exécute le pipeline RAG complet dans 4 configurations pour un jeu
de 30 questions annotées, calcule les métriques d'évaluation, puis produit
trois artefacts consommés par le dashboard :

    corpus/evaluation_report.json   -> rapport détaillé (par requête x config)
    corpus/evaluation_results.csv   -> table plate (par requête x config)
    corpus/comparison_table.json    -> tableau agrégé + gains de chaque module

Configurations évaluées :
    1. Dense                          -- recherche dense (BGE/Qdrant) seulement
    2. Hybrid                         -- dense + BM25 (RRF)
    3. Hybrid + Rerank                -- hybride + reranker BGE cross-encoder
    4. Hybrid + Rerank + Verification -- pipeline complet + vérif NLI

Usage :
    python generate_evaluation_artifacts.py
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from citation_verifier import MNLICitationVerifier, verify_citations
from hybrid_search import EngineConfig, HybridSearchEngine
from rerank_results import CrossEncoderReranker, rerank_results
# Utilise la logique anti-hallucination de generate_answer.py (politique N°7)
# au lieu de la fonction locale generate_claim_answer, pour une cohérence totale.
from generate_answer import extract_grounded_claims, decide_response, REFUSAL_RESPONSE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CORPUS_DIR: Path = SCRIPT_DIR / "corpus"

CHUNKS_FILENAME = "chunks.json"
OUTPUT_REPORT = "evaluation_report.json"
OUTPUT_CSV = "evaluation_results.csv"
OUTPUT_COMPARISON = "comparison_table.json"

CONFIGS: tuple[str, ...] = (
    "Dense",
    "Hybrid",
    "Hybrid + Rerank",
    "Hybrid + Rerank + Verification",
)

METRIC_NAMES: tuple[str, ...] = (
    "Recall@3", "Recall@5", "Recall@10", "MRR",
    "Faithfulness", "Citation Accuracy",
    "Average Latency", "P95 Latency",
)

TOP_K_RETRIEVAL = 10
FETCH_K = 50

# ---------------------------------------------------------------------------
# Jeu de test annote (30 questions, 10 par theme)
# Ground truth document-level : un chunk est pertinent si son document_id
# figure dans `relevant_doc_ids`.
# ---------------------------------------------------------------------------

TEST_QUERIES: list[dict[str, Any]] = [
    # -- RAG (10 questions) -------------------------------------------------
    {"id": "q01", "query": "What is retrieval augmented generation?",
     "theme": "RAG", "difficulty": "simple",
     "relevant_doc_ids": ["2309.15217v2", "2506.06962v3"]},
    {"id": "q02", "query": "What are the key components of a RAG system?",
     "theme": "RAG", "difficulty": "simple",
     "relevant_doc_ids": ["2309.15217v2", "2110.01599v1"]},
    {"id": "q03", "query": "How does RAG combine retrieval and generation?",
     "theme": "RAG", "difficulty": "multi-chunk",
     "relevant_doc_ids": ["2309.15217v2", "2506.06962v3"]},
    {"id": "q04", "query": "What is the RAGAS framework for evaluating RAG?",
     "theme": "RAG", "difficulty": "simple",
     "relevant_doc_ids": ["2309.15217v2"]},
    {"id": "q05", "query": "How does AR-RAG differ from standard RAG for image generation?",
     "theme": "RAG", "difficulty": "simple",
     "relevant_doc_ids": ["2506.06962v3"]},
    {"id": "q06", "query": "What is the EVOR framework for code generation?",
     "theme": "RAG", "difficulty": "simple",
     "relevant_doc_ids": ["2402.12317v2"]},
    {"id": "q07", "query": "What improvements does RocketQA bring to passage retrieval?",
     "theme": "RAG", "difficulty": "medium",
     "relevant_doc_ids": ["2010.08191v2"]},
    {"id": "q08", "query": "What are dense passage retrieval models for RAG?",
     "theme": "RAG", "difficulty": "medium",
     "relevant_doc_ids": ["2110.01599v1"]},
    {"id": "q09", "query": "What is the difference between extractive and generative QA in RAG?",
     "theme": "RAG", "difficulty": "medium",
     "relevant_doc_ids": ["2204.07496v4"]},
    {"id": "q10", "query": "How does retrieval quality affect RAG generation quality?",
     "theme": "RAG", "difficulty": "multi-chunk",
     "relevant_doc_ids": ["2502.00306v2", "2309.15217v2"]},
    # -- Agents (10 questions) ----------------------------------------------
    {"id": "q11", "query": "How does ReAct combine reasoning and action in language models?",
     "theme": "Agents", "difficulty": "multi-chunk",
     "relevant_doc_ids": ["2601.12538v1"]},
    {"id": "q12", "query": "What is the difference between ReAct and chain-of-thought prompting?",
     "theme": "Agents", "difficulty": "simple",
     "relevant_doc_ids": ["2601.12538v1"]},
    {"id": "q13", "query": "How do LLM agents use external tools for information retrieval?",
     "theme": "Agents", "difficulty": "medium",
     "relevant_doc_ids": ["2409.11353v3", "2503.01763v2"]},
    {"id": "q14", "query": "What role does reasoning play in agentic systems?",
     "theme": "Agents", "difficulty": "simple",
     "relevant_doc_ids": ["2601.12538v1"]},
    {"id": "q15", "query": "What is hierarchical multi-agent collaboration?",
     "theme": "Agents", "difficulty": "medium",
     "relevant_doc_ids": ["2512.13930v1"]},
    {"id": "q16", "query": "How does adaptive reasoning suppression work in LLM agents?",
     "theme": "Agents", "difficulty": "simple",
     "relevant_doc_ids": ["2510.00071v2"]},
    {"id": "q17", "query": "What are the limitations of agentic reasoning in large language models?",
     "theme": "Agents", "difficulty": "medium",
     "relevant_doc_ids": ["2601.12538v1", "2508.04848v1"]},
    {"id": "q18", "query": "How do LLM agents retrieve code for programming tasks?",
     "theme": "Agents", "difficulty": "simple",
     "relevant_doc_ids": ["2604.14214v1"]},
    {"id": "q19", "query": "What is the importance of token efficiency in agent architectures?",
     "theme": "Agents", "difficulty": "medium",
     "relevant_doc_ids": ["2604.14214v1", "2504.16021v1"]},
    {"id": "q20", "query": "How does tool-use affect retrieval-augmented agents?",
     "theme": "Agents", "difficulty": "multi-chunk",
     "relevant_doc_ids": ["2409.11353v3"]},
    # -- Fine-tuning (10 questions) -----------------------------------------
    {"id": "q21", "query": "What is LoRA fine tuning and how does it work?",
     "theme": "Fine-tuning", "difficulty": "simple",
     "relevant_doc_ids": ["2607.11940v1", "2411.14961v3"]},
    {"id": "q22", "query": "How does Low-Rank Adaptation reduce model parameters?",
     "theme": "Fine-tuning", "difficulty": "simple",
     "relevant_doc_ids": ["2607.11940v1"]},
    {"id": "q23", "query": "What is the difference between LoRA and full fine-tuning?",
     "theme": "Fine-tuning", "difficulty": "medium",
     "relevant_doc_ids": ["2411.14961v3", "2402.12354v2"]},
    {"id": "q24", "query": "What are the memory benefits of LoRA fine tuning?",
     "theme": "Fine-tuning", "difficulty": "simple",
     "relevant_doc_ids": ["2607.11940v1"]},
    {"id": "q25", "query": "How does federated LoRA fine-tuning work?",
     "theme": "Fine-tuning", "difficulty": "simple",
     "relevant_doc_ids": ["2411.14961v3"]},
    {"id": "q26", "query": "What is parameter-efficient fine-tuning (PEFT)?",
     "theme": "Fine-tuning", "difficulty": "medium",
     "relevant_doc_ids": ["2411.14961v3", "2607.11940v1"]},
    {"id": "q27", "query": "What is the role of rank in LoRA adaptation?",
     "theme": "Fine-tuning", "difficulty": "medium",
     "relevant_doc_ids": ["2607.11940v1"]},
    {"id": "q28", "query": "How does LoRA reduce the memory footprint compared to full fine-tuning?",
     "theme": "Fine-tuning", "difficulty": "simple",
     "relevant_doc_ids": ["2607.11940v1"]},
    {"id": "q29", "query": "What is the difference between LoRA and prompt tuning?",
     "theme": "Fine-tuning", "difficulty": "medium",
     "relevant_doc_ids": ["2411.14961v3", "2210.12607v1"]},
    {"id": "q30", "query": "How does federated learning enhance LoRA fine-tuning?",
     "theme": "Fine-tuning", "difficulty": "simple",
     "relevant_doc_ids": ["2411.14961v3"]},
]



# ---------------------------------------------------------------------------
# Utilitaires metriques
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[str],
    doc_by_chunk: dict[str, str],
    relevant_docs: set[str],
    k: int,
) -> float:
    """Compute document-level Recall@k.

    A retrieved chunk counts as a hit when its parent document belongs to
    the ground-truth relevant documents; the score is the fraction of
    relevant documents surfaced within the top-k ranks.  Document level is
    preferred over chunk level here because a single relevant paper may
    span hundreds of chunks (chunk-level recall would saturate near zero
    regardless of retrieval quality).

    Args:
        retrieved_ids: ordered list of retrieved chunk_ids (rank 1 first).
        doc_by_chunk: mapping chunk_id -> parent document_id.
        relevant_docs: set of ground-truth relevant document_ids.
        k: cutoff rank.

    Returns:
        Fraction of relevant documents found in the top-k results.
    """
    if not relevant_docs:
        return 0.0
    found = {
        doc_by_chunk.get(chunk_id)
        for chunk_id in retrieved_ids[:k]
    } & relevant_docs
    return len(found) / len(relevant_docs)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    doc_by_chunk: dict[str, str],
    relevant_docs: set[str],
) -> float:
    """Compute document-level Mean Reciprocal Rank (MRR).

    Args:
        retrieved_ids: ordered list of retrieved chunk_ids.
        doc_by_chunk: mapping chunk_id -> parent document_id.
        relevant_docs: set of ground-truth relevant document_ids.

    Returns:
        Reciprocal rank of the first hit, or 0.0 when no hit occurs.
    """
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if doc_by_chunk.get(chunk_id) in relevant_docs:
            return 1.0 / rank
    return 0.0


def percentile_95(values: list[float]) -> float:
    """Compute the P95 value from a list of per-query latencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def generate_claim_answer(
    query: str,
    top_chunks: list[dict[str, Any]],
) -> str:
    """Génère une réponse ancrée dans les chunks (politique anti-hallucination).

    Délègue à ``generate_answer.extract_grounded_claims`` + ``decide_response``
    afin d'appliquer **exactement** la même logique anti-hallucination que le
    module ``generate_answer.py`` modifié : 3-5 affirmations ancrées, refus
    explicite si insuffisant, mapping claim → citations. Aucun fallback
    halluciné.

    Args:
        query: la question en langage naturel.
        top_chunks: liste ordonnée des chunks (top-1 en premier).

    Returns:
        Une réponse texte ancrée, ou le message de refus explicite.
    """
    claims = extract_grounded_claims(query, top_chunks)
    return decide_response(claims)


# ---------------------------------------------------------------------------
# Execution du pipeline par configuration
# ---------------------------------------------------------------------------

def run_dense_search(
    engine: HybridSearchEngine,
    query: str,
) -> tuple[list[str], float]:
    """Run dense-only retrieval (hybrid results re-sorted by dense score).

    The HybridSearchEngine returns RRF-fused results carrying both dense
    and BM25 scores.  For the Dense configuration we keep only results from
    the ``dense`` channel, re-sort by ``score_dense``, and return the top
    ``TOP_K_RETRIEVAL`` chunk ids plus latency.

    Args:
        engine: initialised HybridSearchEngine (Qdrant + BGE).
        query: natural-language question.

    Returns:
        Tuple of (chunk_ids, total_latency_ms).
    """
    response = engine.search(query, top_k=TOP_K_RETRIEVAL)
    dense_only = sorted(
        (r for r in response.results if "dense" in r.channels),
        key=lambda r: -(r.score_dense or 0.0),
    )
    chunk_ids = [str(r.chunk_id) for r in dense_only[:TOP_K_RETRIEVAL]]
    return chunk_ids, response.latencies_ms.total_ms


def run_hybrid_search(
    engine: HybridSearchEngine,
    query: str,
) -> tuple[list[str], float]:
    """Run hybrid RRF retrieval (dense + BM25).

    Args:
        engine: initialised HybridSearchEngine.
        query: natural-language question.

    Returns:
        Tuple of (chunk_ids, total_latency_ms).
    """
    response = engine.search(query, top_k=TOP_K_RETRIEVAL)
    chunk_ids = [str(r.chunk_id) for r in response.results[:TOP_K_RETRIEVAL]]
    return chunk_ids, response.latencies_ms.total_ms


def run_rerank(
    engine: HybridSearchEngine,
    reranker: CrossEncoderReranker,
    chunk_index: dict[str, dict[str, Any]],
    query: str,
) -> tuple[list[str], float]:
    """Run hybrid + BGE cross-encoder reranker pipeline.

    Args:
        engine: initialised HybridSearchEngine.
        reranker: initialised CrossEncoderReranker.
        chunk_index: mapping chunk_id -> chunk dict.
        query: natural-language question.

    Returns:
        Tuple of (chunk_ids, total_latency_ms).
    """
    hybrid_response = engine.search(query, top_k=20)
    reranked = rerank_results(
        query=query,
        hybrid_results=hybrid_response.results,
        chunk_index=chunk_index,
        reranker=reranker,
        top_k=TOP_K_RETRIEVAL,
        pool_size=20,
    )
    chunk_ids = [str(r.chunk_id) for r in reranked.results[:TOP_K_RETRIEVAL]]
    total_latency = hybrid_response.latencies_ms.total_ms + reranked.metrics.reranker_ms
    return chunk_ids, total_latency


def run_verification(
    engine: HybridSearchEngine,
    reranker: CrossEncoderReranker,
    verifier: MNLICitationVerifier,
    chunk_index: dict[str, dict[str, Any]],
    query: str,
) -> dict[str, Any]:
    """Run the full pipeline: hybrid -> rerank -> generate -> verify (NLI).

    Args:
        engine: initialised HybridSearchEngine.
        reranker: initialised CrossEncoderReranker.
        verifier: initialised MNLICitationVerifier.
        chunk_index: mapping chunk_id -> chunk dict.
        query: natural-language question.

    Returns:
        Dict with retrieved chunk ids, latency breakdown, answer text,
        faithfulness, citation accuracy and verdict counts.
    """
    t_start = time.perf_counter()

    # 1. Hybrid search (20 candidats pour le pool du reranker).
    hybrid_response = engine.search(query, top_k=20)
    search_latency = hybrid_response.latencies_ms.total_ms

    # 2. Reranking cross-encoder.
    reranked = rerank_results(
        query=query,
        hybrid_results=hybrid_response.results,
        chunk_index=chunk_index,
        reranker=reranker,
        top_k=TOP_K_RETRIEVAL,
        pool_size=20,
    )
    reranker_latency = reranked.metrics.reranker_ms

    # 3. Generation d'une reponse a affirmations reelles (pour le NLI).
    top_chunks = [
        chunk_index[str(r.chunk_id)]
        for r in reranked.results[:5]
        if str(r.chunk_id) in chunk_index
    ]
    answer = generate_claim_answer(query, top_chunks)

    # 4. Verification NLI des citations.
    sources = [
        {
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "page_start": int(r.page_start),
            "page_end": int(r.page_end),
        }
        for r in reranked.results[:5]
    ]
    verification_results, _segments = verify_citations(
        query=query,
        answer=answer,
        sources=sources,
        chunk_index=chunk_index,
        verifier=verifier,
        batch_size=8,
    )

    if verification_results:
        # RAGAS-style : une affirmation est supportee si AU MOINS UNE de ses
        # citations l'entaille => max du support par affirmation.
        by_claim: dict[str, list[float]] = {}
        for result in verification_results:
            by_claim.setdefault(result.claim_text, []).append(
                result.support_score
            )
        per_claim_max = [max(scores) for scores in by_claim.values()]
        faithfulness = statistics.mean(per_claim_max)
        accurate_claims = sum(1 for s in per_claim_max if s >= 0.40)
        citation_accuracy = accurate_claims / len(per_claim_max)
    else:
        faithfulness = 0.0
        citation_accuracy = 0.0

    total_latency = (time.perf_counter() - t_start) * 1000

    return {
        "chunk_ids": [str(r.chunk_id) for r in reranked.results[:TOP_K_RETRIEVAL]],
        "total_latency_ms": round(total_latency, 2),
        "search_latency_ms": round(search_latency, 2),
        "reranker_latency_ms": round(reranker_latency, 2),
        "answer": answer,
        "faithfulness": round(faithfulness, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "citations_verified": len(verification_results),
        "verdict_counts": {
            verdict: sum(1 for r in verification_results if r.verdict == verdict)
            for verdict in ("Supported", "Weak Support", "Unsupported")
        },
    }


# ---------------------------------------------------------------------------
# Boucle d'evaluation principale
# ---------------------------------------------------------------------------

def _build_config_metrics(
    retrieved_ids: list[str],
    doc_by_chunk: dict[str, str],
    relevant_docs: set[str],
    latency_ms: float,
) -> dict[str, Any]:
    """Build the per-config metric block for one query (document level)."""
    return {
        "retrieved_chunk_ids": retrieved_ids,
        "recall_at_3": round(recall_at_k(retrieved_ids, doc_by_chunk, relevant_docs, 3), 4),
        "recall_at_5": round(recall_at_k(retrieved_ids, doc_by_chunk, relevant_docs, 5), 4),
        "recall_at_10": round(recall_at_k(retrieved_ids, doc_by_chunk, relevant_docs, 10), 4),
        "mrr": round(mean_reciprocal_rank(retrieved_ids, doc_by_chunk, relevant_docs), 4),
        "faithfulness": None,
        "citation_accuracy": None,
        "avg_latency_ms": round(latency_ms, 2),
        "p95_latency_ms": None,
    }


def _aggregate_config(
    queries: list[dict[str, Any]],
    config_name: str,
    avg_latency: float,
    p95_latency: float,
) -> dict[str, Any]:
    """Aggregate per-query metrics for one configuration."""
    vals: dict[str, list[float]] = {
        "recall_at_3": [], "recall_at_5": [],
        "recall_at_10": [], "mrr": [],
    }
    faith_vals: list[float] = []
    cit_vals: list[float] = []
    for qr in queries:
        cfg = qr["configs"][config_name]
        for key in vals:
            vals[key].append(cfg[key])
        if cfg.get("faithfulness") is not None:
            faith_vals.append(cfg["faithfulness"])
        if cfg.get("citation_accuracy") is not None:
            cit_vals.append(cfg["citation_accuracy"])

    return {
        "recall_at_3": round(statistics.mean(vals["recall_at_3"]), 4),
        "recall_at_5": round(statistics.mean(vals["recall_at_5"]), 4),
        "recall_at_10": round(statistics.mean(vals["recall_at_10"]), 4),
        "mrr": round(statistics.mean(vals["mrr"]), 4),
        "faithfulness": (
            round(statistics.mean(faith_vals), 4) if faith_vals else None
        ),
        "citation_accuracy": (
            round(statistics.mean(cit_vals), 4) if cit_vals else None
        ),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "queries": len(queries),
    }


def run_evaluation() -> dict[str, Any]:
    """Run the full evaluation: 4 configurations x 30 annotated queries.

    Initialises the pipeline components (Qdrant engine, BGE reranker,
    MNLI verifier), executes each configuration for every query, computes
    per-query metrics, then aggregates per-config statistics.

    Returns:
        Comprehensive results dict ready for JSON serialisation.
    """
    chunks_path = CORPUS_DIR / CHUNKS_FILENAME
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = chunks_data["chunks"]
    chunk_index: dict[str, dict[str, Any]] = {
        str(c["chunk_id"]): c for c in chunks
    }

    # Ground truth document-level : docs pertinents + mapping chunk -> doc.
    doc_by_chunk: dict[str, str] = {
        str(c["chunk_id"]): c["document_id"] for c in chunks
    }
    query_ground_truth: list[set[str]] = [
        set(q["relevant_doc_ids"]) for q in TEST_QUERIES
    ]

    # Composants du pipeline.
    engine = HybridSearchEngine(EngineConfig(fetch_k=FETCH_K))
    reranker = CrossEncoderReranker()
    verifier = MNLICitationVerifier()

    results: dict[str, Any] = {
        "metadata": {
            "project": "RAG Citation Validator",
            "description": "Comparative evaluation of 4 RAG configurations",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_queries": len(TEST_QUERIES),
            "num_configs": len(CONFIGS),
            "configs": list(CONFIGS),
            "metrics": list(METRIC_NAMES),
            "corpus_info": {
                "chunks_file": CHUNKS_FILENAME,
                "total_chunks": len(chunks),
                "total_documents": len(set(doc_by_chunk.values())),
            },
        },
        "queries": [],
        "aggregate": {},
    }

    config_latencies: dict[str, list[float]] = {c: [] for c in CONFIGS}

    for qi, query_info in enumerate(TEST_QUERIES):
        query = query_info["query"]
        relevant_docs = query_ground_truth[qi]
        print(f"[{qi + 1:2d}/{len(TEST_QUERIES)}] {query[:58]}...", flush=True)

        query_result: dict[str, Any] = {
            "query_id": query_info["id"],
            "query": query,
            "theme": query_info["theme"],
            "difficulty": query_info["difficulty"],
            "relevant_doc_ids": query_info["relevant_doc_ids"],
            "configs": {},
        }

        # 1. Dense.
        dense_ids, dense_lat = run_dense_search(engine, query)
        config_latencies["Dense"].append(dense_lat)
        query_result["configs"]["Dense"] = _build_config_metrics(
            dense_ids, doc_by_chunk, relevant_docs, dense_lat
        )

        # 2. Hybrid (RRF).
        hybrid_ids, hybrid_lat = run_hybrid_search(engine, query)
        config_latencies["Hybrid"].append(hybrid_lat)
        query_result["configs"]["Hybrid"] = _build_config_metrics(
            hybrid_ids, doc_by_chunk, relevant_docs, hybrid_lat
        )

        # 3. Hybrid + Rerank.
        rerank_ids, rerank_lat = run_rerank(engine, reranker, chunk_index, query)
        config_latencies["Hybrid + Rerank"].append(rerank_lat)
        query_result["configs"]["Hybrid + Rerank"] = _build_config_metrics(
            rerank_ids, doc_by_chunk, relevant_docs, rerank_lat
        )

        # 4. Hybrid + Rerank + Verification (pipeline complet).
        verif = run_verification(engine, reranker, verifier, chunk_index, query)
        config_latencies["Hybrid + Rerank + Verification"].append(
            verif["total_latency_ms"]
        )
        cfg4 = _build_config_metrics(
            verif["chunk_ids"], doc_by_chunk, relevant_docs,
            verif["total_latency_ms"],
        )
        cfg4["faithfulness"] = verif["faithfulness"]
        cfg4["citation_accuracy"] = verif["citation_accuracy"]
        cfg4["answer"] = verif["answer"]
        cfg4["verification_details"] = {
            "citations_verified": verif["citations_verified"],
            "verdict_counts": verif["verdict_counts"],
        }
        query_result["configs"]["Hybrid + Rerank + Verification"] = cfg4

        results["queries"].append(query_result)

    engine.close()

    # Agregation par configuration (+ P95 au niveau config).
    for config_name in CONFIGS:
        lats = config_latencies[config_name]
        p95 = percentile_95(lats)
        avg_lat = statistics.mean(lats)
        for qr in results["queries"]:
            qr["configs"][config_name]["p95_latency_ms"] = round(p95, 2)
        results["aggregate"][config_name] = _aggregate_config(
            results["queries"], config_name, avg_lat, p95
        )

    return results


# ---------------------------------------------------------------------------
# Ecriture des artefacts
# ---------------------------------------------------------------------------

METRIC_KEY_MAP: dict[str, str] = {
    "recall_at_3": "Recall@3",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "mrr": "MRR",
    "faithfulness": "Faithfulness",
    "citation_accuracy": "Citation Accuracy",
    "avg_latency_ms": "Average Latency",
    "p95_latency_ms": "P95 Latency",
}

PCT_METRICS = {"Recall@3", "Recall@5", "Recall@10", "MRR",
               "Faithfulness", "Citation Accuracy"}


def _pct_gain(old: float | None, new: float | None) -> float | None:
    """Return the percentage gain from old to new (None when undefined)."""
    if old is None or new is None or old == 0:
        return None
    return round((new - old) / old * 100, 1)


def write_json_report(results: dict[str, Any], path: Path) -> None:
    """Write the full evaluation report as JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"=> {path}")


def write_csv_results(results: dict[str, Any], path: Path) -> None:
    """Write a flat CSV table (one row per query x config)."""
    rows: list[dict[str, Any]] = []
    for qr in results["queries"]:
        for config_name in CONFIGS:
            cfg = qr["configs"][config_name]
            rows.append({
                "query_id": qr["query_id"],
                "query": qr["query"],
                "theme": qr["theme"],
                "difficulty": qr["difficulty"],
                "config": config_name,
                "recall_at_3": cfg["recall_at_3"],
                "recall_at_5": cfg["recall_at_5"],
                "recall_at_10": cfg["recall_at_10"],
                "mrr": cfg["mrr"],
                "faithfulness": (
                    cfg["faithfulness"]
                    if cfg["faithfulness"] is not None else ""
                ),
                "citation_accuracy": (
                    cfg["citation_accuracy"]
                    if cfg["citation_accuracy"] is not None else ""
                ),
                "avg_latency_ms": cfg["avg_latency_ms"],
                "p95_latency_ms": cfg["p95_latency_ms"],
            })
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"=> {path}")


def write_comparison_table(results: dict[str, Any], path: Path) -> None:
    """Write the aggregated comparison table with per-config metrics and gains."""
    agg = results["aggregate"]

    # Table lisible : pourcentage pour les metriques de qualite, ms sinon.
    table: dict[str, dict[str, float | None]] = {}
    for config in CONFIGS:
        a = agg[config]
        row: dict[str, float | None] = {}
        for key, name in METRIC_KEY_MAP.items():
            val = a[key]
            if val is None:
                row[name] = None
            elif name in PCT_METRICS:
                row[name] = round(val * 100, 1)
            else:
                row[name] = round(val, 2)
        table[config] = row

    best_config: dict[str, str] = {}
    for metric_name in METRIC_NAMES:
        candidates = [
            (table[c][metric_name], c) for c in CONFIGS
            if table[c][metric_name] is not None
        ]
        if not candidates:
            best_config[metric_name] = ""
        elif metric_name in ("Average Latency", "P95 Latency"):
            best_config[metric_name] = min(candidates)[1]
        else:
            best_config[metric_name] = max(candidates)[1]

    gains: dict[str, Any] = {
        "bm25_gain": {
            "description": "Gain from adding BM25 (Dense -> Hybrid)",
            "source_config": "Dense",
            "target_config": "Hybrid",
            "metrics": {},
        },
        "reranker_gain": {
            "description": "Gain from adding BGE reranker (Hybrid -> Hybrid + Rerank)",
            "source_config": "Hybrid",
            "target_config": "Hybrid + Rerank",
            "metrics": {},
        },
        "nli_gain": {
            "description": (
                "Gain from NLI verification "
                "(Hybrid + Rerank -> Hybrid + Rerank + Verification)"
            ),
            "source_config": "Hybrid + Rerank",
            "target_config": "Hybrid + Rerank + Verification",
            "metrics": {},
        },
    }

    for info in gains.values():
        src = table[info["source_config"]]
        tgt = table[info["target_config"]]
        for metric_name in METRIC_NAMES:
            src_val = src.get(metric_name)
            tgt_val = tgt.get(metric_name)
            if src_val is None and tgt_val is None:
                continue
            if metric_name in ("Average Latency", "P95 Latency"):
                direction = "cost"
            elif src_val is None:
                direction = "new"
            else:
                direction = "improvement"
            info["metrics"][metric_name] = {
                "from": src_val,
                "to": tgt_val,
                "gain_pct": _pct_gain(src_val, tgt_val),
                "direction": direction,
            }

    comparison = {
        "configs": list(CONFIGS),
        "metrics": list(METRIC_NAMES),
        "best_config_per_metric": best_config,
        "table": table,
        "gains": gains,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"=> {path}")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all three evaluation artifacts."""
    print("=" * 70)
    print("Generation des artefacts d'evaluation comparative")
    print(f"  Queries : {len(TEST_QUERIES)}")
    print(f"  Configs : {', '.join(CONFIGS)}")
    print("=" * 70)

    results = run_evaluation()

    write_json_report(results, CORPUS_DIR / OUTPUT_REPORT)
    write_csv_results(results, CORPUS_DIR / OUTPUT_CSV)
    write_comparison_table(results, CORPUS_DIR / OUTPUT_COMPARISON)

    print("\n--- Resume ---")
    for config in CONFIGS:
        agg = results["aggregate"][config]
        print(f"\n{config}:")
        print(
            f"  Recall@3/5/10     : "
            f"{agg['recall_at_3'] * 100:.1f} / "
            f"{agg['recall_at_5'] * 100:.1f} / "
            f"{agg['recall_at_10'] * 100:.1f}"
        )
        print(f"  MRR               : {agg['mrr'] * 100:.1f}%")
        faith = agg.get("faithfulness")
        citacc = agg.get("citation_accuracy")
        if faith is not None:
            print(f"  Faithfulness      : {faith * 100:.1f}%")
        else:
            print("  Faithfulness      : N/A")
        if citacc is not None:
            print(f"  Citation Accuracy : {citacc * 100:.1f}%")
        else:
            print("  Citation Accuracy : N/A")
        print(
            f"  Avg / P95 latency : "
            f"{agg['avg_latency_ms']:.1f} / "
            f"{agg['p95_latency_ms']:.1f} ms"
        )

    print("\nOK - artefacts generes avec succes.")


if __name__ == "__main__":
    main()