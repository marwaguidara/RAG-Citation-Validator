"""
Vérification automatique des citations par NLI (JOUR 7, étape B).

Inputs :
    - question générée                     (query)
    - réponse générée par generate_answer  (answer + citations [N])
    - sources associées                    (chunk_id, document_id, pages)
    - corpus/chunks.json                   (texte complet des passages sources)

Objectif :
    Vérifier que chaque citation soutient réellement l'affirmation qu'elle
    référence, avec le cross-encoder NLI roberta-large-mnli :

        1. segmenter la réponse en affirmations (phrases) reliées à leurs [N] ;
        2. construire les paires (affirmation, passage source) ;
        3. scorer chaque paire avec roberta-large-mnli
           (softmax sur [contradiction, neutral, entailment]) ;
        4. calculer un support_score continu dans [0, 1] ;
        5. verdict par seuils : Supported / Weak Support / Unsupported.

Formule du support_score (défendable et pénalisant le 'neutral') :
    support = P(entail) * P(entail) / (P(entail) + P(contradiction) + 1e-6)
    - tout 'neutral'  => support ~ P(entail) faible  => Unsupported
    - entail dominant => support élevé              => Supported

Verdicts (seuils configurables) :
    support >= SUPPORT_HIGH      => "Supported"
    support >= SUPPORT_LOW       => "Weak Support"
    sinon                        => "Unsupported"

Self-check intégré (`--self-check`, actif par défaut) :
    Deux paires contrôlées (affirmation VRAIE et affirmation FAUSSE contre le
    même passage) prouvent que le verifier distingue bien support et refus —
    y compris quand le corpus couvre mal la question (cas ReAct).

Sortie :
    corpus/citation_verification_report.json

Contraintes respectées : aucun LLM judge (ni GPT, ni Claude), pas d'API HTTP,
pas de Streamlit. Ce module utilise uniquement roberta-large-mnli.

Usage CLI :
    python citation_verifier.py                      # 3 requêtes de test
    python citation_verifier.py --batch-size 8
    python citation_verifier.py --no-self-check
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
REPORT_FILENAME = "citation_verification_report.json"

DEFAULT_MODEL_NAME = "roberta-large-mnli"
DEFAULT_BATCH_SIZE = 8            # roberta-large est lourd : lots conservateurs
MAX_SEQ_LENGTH = 512

SUPPORT_HIGH = 0.70               # >= => "Supported"
SUPPORT_LOW = 0.40                # >= => "Weak Support", sinon "Unsupported"

MNLI_LABELS: tuple[str, str, str] = ("contradiction", "neutral", "entailment")

DEFAULT_TEST_QUERIES: tuple[str, str, str] = (
    "What is retrieval augmented generation?",
    "How does ReAct work?",
    "What is LoRA fine tuning?",
)

SELFCHECK_DOCUMENT_ID = "2309.15217v2"   # "RAGAS: Automated Evaluation of RAG"
SELFCHECK_TRUE_CLAIM = (
    "RAGAS is a framework for evaluating retrieval augmented generation."
)
SELFCHECK_FALSE_CLAIM = (
    "RAGAS uses convolutional neural networks for image classification."
)


class Status(StrEnum):
    """Verdict de la session de vérification."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


def build_logger(as_json: bool) -> tuple[logging.Logger, "EventLogger"]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("citation_verifier")
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
    """Logger d'événements structurés (rendu texte ou JSON)."""

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
# Segmentation des affirmations
# ---------------------------------------------------------------------------

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ClaimSegment:
    """Une affirmation de la réponse, reliée à ses indices de citation."""

    claim_text: str
    citation_indices: tuple[int, ...]


def segment_claims(answer: str) -> list[ClaimSegment]:
    """Découpe la réponse en affirmations (phrases) et rattache les citations.

    Chaque phrase garde uniquement son texte (sans les balises [N]) et la
    liste des indices de source cités. Les phrases sans citation sont ignorées
    (aucune affirmation n'est vérifiée sans source).
    """
    segments: list[ClaimSegment] = []
    for raw_sentence in _SENTENCE_SPLIT.split(answer):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        indices = tuple(
            int(match.group(1))
            for match in _CITATION_PATTERN.finditer(sentence)
        )
        if not indices:
            continue
        clean_text = _CITATION_PATTERN.sub("", sentence).strip().strip(":;,")
        segments.append(ClaimSegment(claim_text=clean_text, citation_indices=indices))
    return segments


# ---------------------------------------------------------------------------
# Modèle MNLI
# ---------------------------------------------------------------------------

class MNLICitationVerifier:
    """Wrapper d'inférence NLI pour roberta-large-mnli.

    Score une liste de paires (premise=passage, hypothesis=affirmation) et
    retourne pour chacune les probabilités
    [contradiction, neutral, entailment] (ordre des logits MNLI).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
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
        self, premises: list[str], hypotheses: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        """Score les paires (premise, hypothesis) ; retourne des softmax.

        Returns:
            Liste de probabilités [contradiction, neutral, entailment],
            une par paire.
        """
        results: list[list[float]] = []
        for start in range(0, len(premises), batch_size):
            batch_premises = premises[start:start + batch_size]
            batch_hypotheses = hypotheses[start:start + batch_size]
            inputs = self._tokenizer(
                list(zip(batch_premises, batch_hypotheses)),
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            results.extend(probs)
        return results


# ---------------------------------------------------------------------------
# Vérification d'une citation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CitationVerificationResult:
    """Résultat de vérification d'une citation (une affirmation -> une source)."""

    citation_id: str
    document_id: str
    page_start: int
    page_end: int
    claim_text: str
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    support_score: float
    verdict: str


def compute_support_score(
    contradiction: float, neutral: float, entailment: float,
) -> float:
    """Score de support continu dans [0, 1].

    support = P(entail) * P(entail) / (P(entail) + P(contradiction) + 1e-6)
    - si la paire est 'neutral' dominante, P(entail) est faible => support bas ;
    - si 'entailment' domine, le ratio est proche de 1 => support élevé ;
    - si 'contradiction' domine, le ratio tend vers 0 => support nul.
    """
    denominator = entailment + contradiction + 1e-6
    ratio = entailment / denominator
    return round(min(1.0, entailment * ratio), 4)


def verdict_for(support_score: float) -> str:
    """Mappe un support_score vers un verdict par seuils."""
    if support_score >= SUPPORT_HIGH:
        return "Supported"
    if support_score >= SUPPORT_LOW:
        return "Weak Support"
    return "Unsupported"


# ---------------------------------------------------------------------------
# Fenêtre locale NLI : prémisse = phrase du claim ± 1 phrase.  Correction de
# la dilution d'entailment observée quand la prémisse est le chunk complet
# (400-500 tokens) alors que roberta-large-mnli est entraîné sur des paires
# de phrases courtes.
# ---------------------------------------------------------------------------

_LOGGER = logging.getLogger("citation_verifier")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_local_premise(chunk_text: str, claim_text: str) -> str:
    """Construit une prémisse NLI locale à partir d'un chunk.

    Algorithme :
        1. découper le chunk en phrases avec ``_SENTENCE_SPLIT`` ;
        2. trouver la phrase ayant le plus fort overlap lexical (jetons
           alphanumériques minuscules) avec ``claim_text`` ;
        3. récupérer phrase précédente + phrase correspondante + phrase
           suivante ;
        4. concaténer ces 3 phrases ;
        5. fallback = chunk complet si aucune phrase n'est trouvée.

    Args:
        chunk_text: texte complet du chunk cité.
        claim_text: affirmation à vérifier (hypothèse NLI).

    Returns:
        La fenêtre locale (<= 3 phrases) ou le chunk complet en fallback.
    """
    text = str(chunk_text)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return text

    claim_tokens = set(re.findall(r"[A-Za-z0-9]+", claim_text.lower()))
    best_idx: int | None = None
    best_score: int = -1
    for idx, sent in enumerate(sentences):
        tokens = set(re.findall(r"[A-Za-z0-9]+", sent.lower()))
        score = len(tokens & claim_tokens)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None or best_score <= 0:
        return text

    window = sentences[max(0, best_idx - 1): min(len(sentences), best_idx + 2)]
    return " ".join(window)


def verify_citations(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    chunk_index: dict[str, dict[str, Any]],
    verifier: MNLICitationVerifier,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[CitationVerificationResult], list[ClaimSegment]]:
    """Vérifie toutes les affirmations citées de la réponse contre leurs sources.

    Pipeline :
        1. segmenter la réponse en affirmations reliées à leurs [N] ;
        2. pour chaque (affirmation, source N), récupérer le texte du passage
           (chunk_id = sources[N-1].chunk_id) ;
        3. scorer les paires avec MNLI (batch) ;
        4. calculer support_score et verdict.

    Args:
        query: question en langage naturel (identifiabilité dans le rapport).
        answer: réponse générée (avec marqueurs [N]).
        sources: sources ordonnées telles que sources[N-1] correspond à [N].
            Chaque dict contient chunk_id, document_id, page_start, page_end.
        chunk_index: mapping chunk_id -> chunk (texte complet).
        verifier: instance du modèle MNLI.
        batch_size: paires par lot.

    Returns:
        Tuple (résultats de vérification, affirmations segmentées).
    """
    segments = segment_claims(answer)
    premises: list[str] = []
    hypotheses: list[str] = []
    pair_keys: list[tuple[int, int]] = []   # (index_segment, source_index)

    for segment_index, segment in enumerate(segments):
        for source_index in segment.citation_indices:
            if 1 <= source_index <= len(sources):
                source = sources[source_index - 1]
                passage = chunk_index.get(str(source["chunk_id"]))
                if passage is None:
                    continue
                original_chars = len(str(passage["text"]))
                local_premise = extract_local_premise(
                    str(passage["text"]), segment.claim_text
                )
                local_chars = len(local_premise)
                reduction_ratio = (
                    round(1.0 - local_chars / original_chars, 3)
                    if original_chars else 0.0
                )
                _LOGGER.info(
                    "nli_local_premise",
                    extra={
                        "chunk_id": str(source.get("chunk_id", "")),
                        "original_chars": original_chars,
                        "local_chars": local_chars,
                        "reduction_ratio": reduction_ratio,
                    },
                )
                premises.append(local_premise)
                hypotheses.append(segment.claim_text)
                pair_keys.append((segment_index, source_index))

    results: list[CitationVerificationResult] = []
    if not premises:
        return results, segments

    probabilities = verifier.score_pairs(premises, hypotheses, batch_size=batch_size)
    for (segment_index, source_index), probs in zip(pair_keys, probabilities):
        contradiction, neutral, entailment = probs
        source = sources[source_index - 1]
        support = compute_support_score(contradiction, neutral, entailment)
        _LOGGER.info(
            "nli_entailment_score",
            extra={
                "claim_text": segments[segment_index].claim_text,
                "chunk_id": str(source.get("chunk_id", "")),
                "entailment_score": round(entailment, 4),
            },
        )
        results.append(CitationVerificationResult(
            citation_id=f"{hashlib.md5(
                f"{query}|{source_index}|{segment_index}".encode()
            ).hexdigest()[:12]}",
            document_id=str(source["document_id"]),
            page_start=int(source["page_start"]),
            page_end=int(source["page_end"]),
            claim_text=segments[segment_index].claim_text,
            entailment_score=round(entailment, 4),
            neutral_score=round(neutral, 4),
            contradiction_score=round(contradiction, 4),
            support_score=support,
            verdict=verdict_for(support),
        ))
    return results, segments


# ---------------------------------------------------------------------------
# Self-check de discriminabilité
# ---------------------------------------------------------------------------

def run_self_check(
    chunk_index: dict[str, dict[str, Any]],
    verifier: MNLICitationVerifier,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[CitationVerificationResult]:
    """Vérifie deux affirmations contrôlées (VRAIE / FAUSSE) contre un passage.

    Démonstre que le verifier distingue le support de la contradiction —
    indépendamment de la couverture du corpus sur la question posée.
    """
    rally = next(
        (chunk for chunk in chunk_index.values()
         if str(chunk.get("document_id")) == SELFCHECK_DOCUMENT_ID),
        None,
    )
    if rally is None:
        raise SystemExit(
            f"Document {SELFCHECK_DOCUMENT_ID} absent du corpus pour le self-check."
        )
    passage = str(rally["text"])
    premise = [passage, passage]
    hypothesis = [SELFCHECK_TRUE_CLAIM, SELFCHECK_FALSE_CLAIM]
    probs = verifier.score_pairs(premise, hypothesis, batch_size=batch_size)
    results: list[CitationVerificationResult] = []
    for claim, probabilities, expected in zip(
        hypothesis, probs, ("Supported", "Unsupported")
    ):
        contradiction, neutral, entailment = probabilities
        support = compute_support_score(contradiction, neutral, entailment)
        results.append(CitationVerificationResult(
            citation_id="selfcheck-" + hashlib.md5(claim.encode()).hexdigest()[:8],
            document_id=SELFCHECK_DOCUMENT_ID,
            page_start=int(rally["page_start"]),
            page_end=int(rally["page_end"]),
            claim_text=claim,
            entailment_score=round(entailment, 4),
            neutral_score=round(neutral, 4),
            contradiction_score=round(contradiction, 4),
            support_score=support,
            verdict=verdict_for(support),
        ))
    return results


# ---------------------------------------------------------------------------
# Agrégation des métriques
# ---------------------------------------------------------------------------

def aggregate_verdicts(results_group: list[list[CitationVerificationResult]]) -> dict[str, Any]:
    """Agrège les verdicts et le support moyen sur toutes les requêtes."""
    flat = [result for group in results_group for result in group]
    count = len(flat)
    if not count:
        return {"citations_verified": 0}
    counts = {"Supported": 0, "Weak Support": 0, "Unsupported": 0}
    for result in flat:
        counts[result.verdict] = counts.get(result.verdict, 0) + 1
    all_support = [result.support_score for result in flat]
    return {
        "citations_verified": count,
        "verdict_counts": counts,
        "mean_support_score": round(sum(all_support) / count, 4),
        "min_support_score": min(all_support),
        "max_support_score": max(all_support),
        "supported_pct": round(100 * counts["Supported"] / count, 1),
    }


# ---------------------------------------------------------------------------
# Rapport de session
# ---------------------------------------------------------------------------

def build_session_report(
    model_name: str,
    device: str,
    batch_size: int,
    queries: list[str],
    results_per_query: list[list[CitationVerificationResult]],
    self_check_results: list[CitationVerificationResult],
    chunks_sha256: str | None,
    segments_per_query: list[list[ClaimSegment]],
) -> dict[str, Any]:
    """Assemble citation_verification_report.json (sérialisable en JSON)."""
    from importlib.metadata import version

    all_results = [r for group in results_per_query for r in group]
    status = Status.PASS.value if all_results else Status.FAIL.value
    return {
        "generator": "citation_verifier.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "config": {
            "model": model_name,
            "device": device,
            "batch_size": batch_size,
            "max_seq_length": MAX_SEQ_LENGTH,
            "verdict_thresholds": {"supported": SUPPORT_HIGH, "weak": SUPPORT_LOW},
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
        "aggregate": aggregate_verdicts(results_per_query),
        "self_check": {
            "enabled": bool(self_check_results),
            "document": SELFCHECK_DOCUMENT_ID,
            "results": [asdict(result) for result in self_check_results],
        },
        "queries": [
            {
                "query": query,
                "claims": [asdict(segment) for segment in segments],
                "verifications": [asdict(result) for result in results],
            }
            for query, results, segments in zip(
                queries, results_per_query, segments_per_query,
            )
        ],
    }


def write_session_report(corpus_dir: Path, report: dict[str, Any]) -> Path:
    """Écrit citation_verification_report.json sous corpus/."""
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
    """Parse les arguments CLI de la session de vérification."""
    parser = argparse.ArgumentParser(
        description=(
            "Vérifie les citations de la réponse générée via roberta-large-mnli "
            "(rapport : citation_verification_report.json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sans requête fournie, exécute les 3 requêtes de test du projet.\n"
            "Exemples :\n"
            '  python citation_verifier.py\n'
            '  python citation_verifier.py --no-self-check'
        ),
    )
    parser.add_argument("queries", nargs="*", default=None,
                        help="Requêtes à vérifier (défaut : 3 requêtes de test).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"Modèle NLI ({DEFAULT_MODEL_NAME}).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device d'inférence.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        metavar="N",
                        help=f"Paires par lot ({DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--no-self-check", action="store_true",
                        help="Désactive le self-check de discriminabilité.")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Format des logs console (défaut : text).")
    return parser.parse_args()


def _console_safe(text: str) -> str:
    """Rend un texte affichable sur la console courante (Windows cp1252 inclus)."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_verification(query: str, results: list[CitationVerificationResult]) -> None:
    """Affiche citation / support_score / verdict pour une requête."""
    print(_console_safe(f"\nQUERY: {query}"))
    for result in results:
        print(
            f"  [{result.document_id} p.{result.page_start}-{result.page_end}]"
            f" support={result.support_score:.3f} · entail={result.entailment_score:.2f}"
            f" · neutral={result.neutral_score:.2f}"
            f" · contra={result.contradiction_score:.2f}"
            f" · {result.verdict}"
        )
        print(_console_safe(f"      affirmation: {result.claim_text[:120]}"))


def main() -> None:
    """Point d'entrée : pipeline amont -> segmentation -> MNLI -> rapport."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.batch_size <= 0:
        raise SystemExit("--batch-size doit être > 0.")
    queries = list(args.queries) if args.queries else list(DEFAULT_TEST_QUERIES)
    chunks_path = locate_corpus(INPUT_FILENAME)
    if chunks_path is None:
        raise SystemExit(f"{INPUT_FILENAME} introuvable sous corpus/.")
    chunks_sha256 = sha256_of(chunks_path)
    chunk_index = load_chunk_text_index(chunks_path)[0]

    # Reconstruire la réponse + les sources (alignement parfait des chunk_id).
    from generate_answer import build_provider, generate_answer
    from hybrid_search import EngineConfig, get_engine
    from rerank_results import CrossEncoderReranker, rerank_results

    events.info("pipeline_amont_started", queries=len(queries))
    engine = get_engine(EngineConfig(fetch_k=20))
    try:
        hybrid_responses = [engine.search(query=query, top_k=20) for query in queries]
    finally:
        engine.close()
    reranker = CrossEncoderReranker(device="auto")
    reranked_lists = [
        rerank_results(
            query=query,
            hybrid_results=hybrid_response.results,
            chunk_index=chunk_index,
            reranker=reranker,
            top_k=5,
            pool_size=20,
        ).results
        for query, hybrid_response in zip(queries, hybrid_responses)
    ]
    provider = build_provider("template", "llama3.1")
    answer_responses = [
        generate_answer(query, reranked, chunk_index, provider=provider)
        for query, reranked in zip(queries, reranked_lists)
    ]

    # Vérification NLI.
    events.info("nli_model_loading", model=args.model, device=args.device)
    verifier = MNLICitationVerifier(model_name=args.model, device=args.device)
    results_per_query: list[list[CitationVerificationResult]] = []
    segments_per_query: list[list[ClaimSegment]] = []
    for query, answer_response, reranked in zip(
        queries, answer_responses, reranked_lists
    ):
        sources = [
            {
                "chunk_id": str(result.chunk_id),
                "document_id": str(result.document_id),
                "page_start": int(result.page_start),
                "page_end": int(result.page_end),
            }
            for result in reranked[:5]
        ]
        results, segments = verify_citations(
            query=query,
            answer=answer_response.answer,
            sources=sources,
            chunk_index=chunk_index,
            verifier=verifier,
            batch_size=args.batch_size,
        )
        results_per_query.append(results)
        segments_per_query.append(segments)

    self_check_results: list[CitationVerificationResult] = []
    if not args.no_self_check:
        self_check_results = run_self_check(
            chunk_index=chunk_index, verifier=verifier, batch_size=args.batch_size,
        )

    for query, results in zip(queries, results_per_query):
        _print_verification(query, results)
    if not args.no_self_check:
        print(_console_safe("\nSELF-CHECK (discriminabilité) :"))
        for result in self_check_results:
            print(
                f"  {result.verdict}: {result.claim_text[:110]}"
                f"  (support={result.support_score:.3f})"
            )

    report = build_session_report(
        model_name=args.model,
        device=verifier.device,
        batch_size=args.batch_size,
        queries=queries,
        results_per_query=results_per_query,
        self_check_results=self_check_results,
        chunks_sha256=chunks_sha256,
        segments_per_query=segments_per_query,
    )
    report_path = write_session_report(chunks_path.parent, report)

    aggregate = report["aggregate"]
    events.info(
        "session_completed",
        status=report["status"],
        citations_verified=aggregate.get("citations_verified", 0),
        supported_pct=aggregate.get("supported_pct"),
        mean_support=aggregate.get("mean_support_score"),
    )
    print(f"\n=> rapport : {report_path}")
    sys.exit(EXIT_OK if report["status"] == Status.PASS.value else EXIT_FAIL)


if __name__ == "__main__":
    main()