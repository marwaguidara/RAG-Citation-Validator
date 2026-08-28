"""
Génération d'une réponse RAG à partir des chunks rerankés (JOUR 7, étape A).

Inputs :
    - query                            (question en langage naturel)
    - top-5 rerankés                   (RerankedResult issus de rerank_results)
    - corpus/chunks.json               (texte complet des chunks)

Objectif :
    Construire un contexte, appeler un LLM, générer une réponse concise citée,
    puis produire des citations explicites reliant des phrases à des sources.

Format de sortie (contrat) :
    - answer      : réponse RAG synthétique et ancrée ; chaque phrase factuelle
                    se termine par un point puis ses citations [N] ;
    - claims      : affirmations vérifiables (text -> citations) ;
    - sources     : sources ordonnées, avec drapeau `extracted` si citées.

Choix techniques :
    - Fournisseur LLM AGNOSTIQUE (objet `LLMProvider` exposant generate(...)) :
        * OllamaProvider   : LLM local via HTTP (zéro clé API).
        * TemplateProvider : déterministe hors ligne, valide le pipeline sans LLM.
    - Prompt structuré en JSON blob : le LLM renvoie {"claims":[...]}, un parser
      tolérant extrait affirmations et citations [num_source].
    - Politique anti-hallucination : chaque affirmation est ancrée au contexte ;
      le contrôle strict final est délégué à la Citation Verification NLI (aval).

Output : generation_report.json (rapport traçable par requête).

Usage programme :
    from generate_answer import generate_answer
    response = generate_answer(query, top5, chunk_index)

Usage CLI :
    python generate_answer.py                       # 3 requêtes de test
    python generate_answer.py --provider template   # hors ligne (validation)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import requests  # pour le fournisseur Ollama (uniquement si utilisé)

# Contrôle final du support NLI : la politique « tous les scores < SUPPORT_LOW
# => refus explicite » est appliquée ICI (generate_answer), en réutilisant
# citation_verifier SANS le modifier. Si torch/transformers sont absents,
# la gate est neutralisée et le comportement antérieur est conservé.
try:
    from citation_verifier import (  # noqa: E402
        MNLICitationVerifier,
        SUPPORT_HIGH,
        SUPPORT_LOW,
        compute_support_score,
        extract_local_premise,
        segment_claims,
    )
    _NLI_AVAILABLE = True
except ImportError:  # pragma: no cover - environnement sans torch
    _NLI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "[%(asctime)s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

INPUT_FILENAME = "chunks.json"
REPORT_FILENAME = "generation_report.json"

LOCAL_CORPUS_MAX_CHARS = 1_200     # longueur max d'un chunk injecté dans le prompt
MAX_SOURCES = 5                    # nb max de sources citables (top-5 reranké)
TEMPLATE_MODEL_NAME = "template"   # identifiant du fournisseur hors ligne
EXAMPLE_LIMIT = 10                 # nb max de problèmes conservés au rapport

# Politique anti-hallucination : réponse fondée sur les chunks uniquement.
MAX_CLAIMS = 5                     # nb max d'affirmations (borné haut)
# Plancher de claims ancrés pour accepter la réponse. Abaissé de 3 à 1 : un LLM
# instruction-tuné (ex. qwen2.5:3b) produit souvent UNE phrase complète qui
# répond directement à la question ; l'ancien plancher 3 rejetait cette sortie
# parfaitement ancrée et forçait le repli Template => sortie identique partout.
# La qualité reste garantie par le grounding lexical (GROUNDING_COVERAGE) + la
# vérification NLI (roberta-large-mnli) en aval : 0 claim ancré => refus.
MIN_CLAIMS_FOR_ANSWER = 1
CLAIM_MIN_CHARS = 40               # affirmation trop courte => bruit, ignorée
CLAIM_MAX_CHARS = 160              # affirmation longue => risque de neutral NLI
# Couverture lexicale minimale d'une phrase LLM par sa source citée. Tolérant
# (0.45) pour autoriser la SYNTHÈSE / la paraphrase : un seuil trop élevé
# (0.75 historique) rejetait toute phrase reformulée et forçait l'extraction
# verbatim (sortie "copier-coller"). Contrôle strict final = NLI (aval).
GROUNDING_COVERAGE = 0.45
# La reponse directe (claims[0]) est une SYNTHESE qui repond a la question :
# seuil lexical relache (elle reformule, donc couverture token plus faible) ;
# le controle semantique reste assure par la gate NLI en aval.
ANSWER_GROUNDING_COVERAGE = 0.15
# Tolérance paraphrase de la gate NLI (apply_nli_gate). Cause racine mesurée
# (audit_qfocus + replay) : roberta-large-mnli évalue la SYNTHESE/reformulation
# comme "neutral" -> support <= 0.24, très sous SUPPORT_LOW (0.40), même pour
# des claims entièrement ancrés et corrects (Q2/Q3). Avant le repli extractif,
# la gate CONSERVE la réponse LLM si chaque claim a une couverture >= ce seuil
# de ses jetons contre AU MOINS une source du contexte (toutes sources
# confondues), ET n'est contredit par aucune source (garde contradiction).
PARAPHRASE_COVERAGE_RESCUE = 0.40
REFUSAL_RESPONSE = (
    "The provided sources do not contain sufficient information."
)
# Message utilisateur clair lors du basculement Ollama -> Template.
OLLAMA_FALLBACK_MESSAGE = (
    "Ollama non disponible (serveur injoignable ou modèle non installé). "
    "Utilisation du provider Template."
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# === TRACE TEMPORAIRE (audit flux Ollama) : RAG_TRACE=1 pour activer ===
_TRACE = os.environ.get("RAG_TRACE", "") == "1"


def _trace(*parts: Any) -> None:
    if _TRACE:
        print("[TRACE]", *parts, file=sys.stderr)

DEFAULT_TEST_QUERIES: tuple[str, str, str] = (
    "What is retrieval augmented generation?",
    "How does ReAct work?",
    "What is LoRA fine tuning?",
)

SYSTEM_PROMPT = (
    "You are a rigorous AI assistant that answers the user's question using "
    "ONLY the numbered sources given in the context. STRICT RULES:\n"
    "1. ANSWER THE QUESTION FIRST: claims[0] MUST be ONE complete sentence "
    "that DIRECTLY answers the question (a definition for 'what is', the key "
    "differences for 'difference between', the reason for 'why'). Synthesize "
    "it from the sources; do NOT quote a random fact from a source.\n"
    "2. THEN EVIDENCE: claims[1] and claims[2] add 1-2 supporting facts or "
    "benefits taken from the sources, one fact per claim, each relevant to "
    "the question.\n"
    "3. GROUNDED ONLY: use ONLY facts present in the context. Never invent "
    "numbers, figures, names, or conclusions that are not stated in a source.\n"
    "4. CITATION RULES: every claim cites the 1-2 sources that most DIRECTLY "
    "support it (the most specific and relevant ones, never all sources). "
    "Put the citation numbers ONLY in the claim's \"citations\" array.\n"
    "5. CITATION-CONTENT CONSISTENCY: every fact in a claim must come from "
    "the source(s) listed in ITS OWN citations. Sources may describe "
    "different tasks or modalities (e.g., code generation vs. image "
    "generation): do not mix them, and only use content relevant to the "
    "question asked.\n"
    "6. If the context does not contain enough information to answer, reply "
    "exactly: \"The provided sources do not contain sufficient information.\" "
    "Do not answer from your own knowledge.\n"
    "7. OUTPUT FORMAT: reply with a single JSON object (no markdown) of this "
    "exact schema:\n"
    "{\"claims\":[{\"text\":\"...\",\"citations\":[1]}]}\n"
    "where claims[0] is your direct answer to the question and each 'text' "
    "is a complete sentence of at most 160 characters."
)
class Status(StrEnum):
    """Verdict de la session de génération."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


def build_logger(as_json: bool) -> tuple[logging.Logger, "EventLogger"]:
    """Configure et retourne (logger stdlib, logger d'événements structurés)."""
    logger = logging.getLogger("generate_answer")
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
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_chunk_text_index(
    input_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Charge chunks.json en un index {chunk_id -> chunk} et des métadonnées."""
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    chunks = payload.get("chunks", [])
    index: dict[str, dict[str, Any]] = {}
    themes: dict[str, int] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id:
            continue
        theme = str(chunk.get("theme", "") or "")
        themes[theme] = themes.get(theme, 0) + 1
        index[chunk_id] = {
            "text": str(chunk.get("text", "")),
            "document_id": str(chunk.get("document_id", "")),
            "page_start": int(chunk.get("page_start", 1) or 1),
            "page_end": int(chunk.get("page_end", 1) or 1),
            "tokens_est": int(chunk.get("tokens_est", 0) or 0),
            "theme": theme,
        }
    metadata = {
        "chunks_file": str(input_path),
        "chunks_total": len(index),
        "chunks_sha256": sha256_of(input_path),
        "corpus_name": str(payload.get("corpus_name", Path(input_path).stem)),
        "themes": themes,
    }
    return index, metadata
@runtime_checkable
class LLMProvider(Protocol):
    """Contrat d'un fournisseur de génération de texte."""

    name: str

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Génère une réponse texte pour `prompt`."""

    def generate_claims(
        self,
        query: str,
        sources: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> list["Claim"]:
        """Génère des affirmations vérifiables, chacune citant ses sources."""

    def config_dict(self) -> dict[str, Any]:
        """Retourne les paramètres d'exécution (traçabilité du rapport)."""


class OllamaProvider:
    """Fournisseur LLM local via l'API HTTP d'Ollama (/api/generate).

    Zéro configuration externe si Ollama est lancé localement (port 11434).
    La génération est requise en mode NON-streaming pour rester mesurable.

    Au démarrage, un self-test (GET /api/tags) vérifie que le serveur répond ET
    que le modèle demandé est installé ; sinon basculement automatique vers
    ``TemplateProvider`` (hors ligne) avec un message utilisateur explicite.
    """

    _CHECK_TIMEOUT = 5.0

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.name = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Test de connexion au démarrage : serveur injoignable ou modèle absent
        # => bascule vers le provider hors ligne, sans interrompre le pipeline.
        self._template: TemplateProvider | None = None
        if not self._check_available():
            logging.getLogger("generate_answer").warning(OLLAMA_FALLBACK_MESSAGE)
            self._emit_fallback_message()
            self._template = TemplateProvider()
            self.name = self._template.name

    @staticmethod
    def _emit_fallback_message() -> None:
        """Affiche un message utilisateur clair (stderr) lors du repli."""
        print(OLLAMA_FALLBACK_MESSAGE, file=sys.stderr)

    def _check_available(self) -> bool:
        """Serveur joignable ET le modèle demandé est réellement installé."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=self._CHECK_TIMEOUT
            )
            response.raise_for_status()
            models = (response.json() or {}).get("models") or []
        except (requests.RequestException, ValueError):
            return False
        installed = {m.get("name") for m in models}
        return any(
            name == self.model or name.startswith(self.model + ":")
            for name in installed
        )

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        if self._template is not None:
            return self._template.generate(prompt, temperature=temperature)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=600,
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        except (requests.RequestException, ValueError):
            # Filet de sécurité au runtime (ex. modèle supprimé, 400 HTTP,
            # timeout réseau) : repli hors ligne sans interrompre.
            logging.getLogger("generate_answer").warning(OLLAMA_FALLBACK_MESSAGE)
            self._emit_fallback_message()
            self._template = TemplateProvider()
            self.name = self._template.name
            return self._template.generate(prompt, temperature=temperature)

    @staticmethod
    def _sanitize_claim_citations(
        candidates: list[Claim], n_sources: int, max_per_claim: int = 2,
    ) -> list[Claim]:
        """Sanitation déterministe des citations LLM (anti sur-citation).

        1. supprime les indices hors pool (hallucinations type [8]) ;
        2. plafonne chaque claim à ``max_per_claim`` citations ;
        3. retire les marqueurs [N] du TEXTE du claim (le rendu final ajoute
           proprement les marqueurs, sans jamais dupliquer).
        """
        sanitized: list[Claim] = []
        for claim in candidates:
            inline = [
                int(m.group(1)) for m in _CITATION_PATTERN.finditer(claim.text)
            ]
            valid = sorted({
                i for i in (*claim.citations, *inline) if 1 <= i <= n_sources
            })
            text = _CITATION_PATTERN.sub("", claim.text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            # Connecteurs orphelins laissés par le retrait des [N] inline
            # (ex. "as described in and ." -> "as described in").
            text = re.sub(r"[\s.,;:]+$", "", text)
            text = re.sub(r"\s*(?:,|\band\b|\bor\b)$", "", text,
                          flags=re.IGNORECASE).strip()
            text = re.sub(r"[\s.,;:]+$", "", text)
            if not valid or not text:
                continue
            sanitized.append(Claim(
                text=text, citations=valid[:max_per_claim],
            ))
        return sanitized

    def generate_claims(
        self,
        query: str,
        sources: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> list[Claim]:
        """Claims du LLM, ancrés puis exploités ; sinon extraction déterministe."""
        if self._template is not None:
            return self._template.generate_claims(
                query, sources, temperature=temperature
            )
        prompt = build_prompt(query, sources)
        _trace("STEP1 PROMPT>>>\n", prompt, "\n<<<")
        raw = self.generate(prompt, temperature=temperature)
        _trace("STEP2 RAW_LLM>>>\n", raw, "\n<<<")
        candidates = parse_claim_json(raw)
        candidates = self._sanitize_claim_citations(candidates, len(sources))
        _trace("STEP3 AFTER_PARSE+SANITIZE candidates=", [
            (c.text[:70], c.citations) for c in candidates
        ])
        grounded: list[Claim] = []
        for position, candidate in enumerate(candidates):
            # claims[0] = réponse directe à la question (seuil relâché) ;
            # claims suivants = preuves (seuil standard).
            is_answer = position == 0
            threshold = ANSWER_GROUNDING_COVERAGE if is_answer else GROUNDING_COVERAGE
            ok = _claim_grounded(candidate, sources, is_answer=is_answer)
            # Réparation sémantique des citations (NLI, pas lexicale) :
            # TOUS les claims sont remappés vers les sources qui soutiennent
            # réellement le claim selon la NLI (2 max). Si aucune source ne
            # passe SUPPORT_LOW, on conserve les citations du LLM et la gate
            # NLI décidera en aval.
            # Correctif audit_nli_chain : le LLM citait [3,4] alors que le
            # fait ("two to four times of execution accuracy") n'est présent
            # que dans la source [3] ; la paire (claim, source[4]) retombait
            # à support=0.013 alors que (claim, source[3]) vaut 0.799.
            best = _best_supported_citations(candidate.text, sources) if sources else []
            if best and sorted(best) != sorted(candidate.citations):
                _trace(
                    "STEP4 REPAIR role=", "ANSWER" if is_answer else "evidence",
                    "cits", candidate.citations, "->", best,
                )
                candidate = Claim(text=candidate.text, citations=best)
                ok = True
            covs = [
                round(
                    len(_tokenize(candidate.text) & _tokenize(
                        str(sources[i - 1].get("text", ""))
                    )) / max(len(_tokenize(candidate.text)), 1),
                    2,
                )
                for i in candidate.citations if 1 <= i <= len(sources)
            ]
            _trace(
                "STEP4 GROUND text=", candidate.text[:60], "cits=", candidate.citations,
                "covs=", covs, "score>=", threshold,
                "role=", "ANSWER" if is_answer else "evidence",
                "->", "OK" if ok else "REJECT",
            )
            if ok:
                grounded.append(candidate)
        _trace("STEP4 AFTER_GROUND grounded=", len(grounded), "of", len(candidates))
        if MIN_CLAIMS_FOR_ANSWER <= len(grounded) <= MAX_CLAIMS:
            _trace("STEP4 -> UTILISE LES CLAIMS LLM (pas de fallback)")
            return grounded[:MAX_CLAIMS]
        _trace("STEP4 -> FALLBACK TEMPLATE (claims rejetés) => sortie extractive")
        return extract_grounded_claims(query, sources)

    def config_dict(self) -> dict[str, Any]:
        if self._template is not None:
            return self._template.config_dict()
        return {"provider": self.name, "model": self.model, "base_url": self.base_url}
class TemplateProvider:
    """Fournisseur hors ligne et déterministe : affirmations ancrées au contexte.

    Chaque affirmation est une phrase extraite mot pour mot d'un chunk et n'est
    citée que vers les sources qui la contiennent littéralement. Ce provider est
    un RECOURS extractif : pour une vraie synthèse, utiliser OllamaProvider avec
    un modèle installé.
    """

    def __init__(self) -> None:
        self.name = TEMPLATE_MODEL_NAME

    def generate_claims(
        self,
        query: str,
        sources: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> list[Claim]:
        """Retourne les affirmations ancrées extraites du contexte."""
        return extract_grounded_claims(query, sources)

    @staticmethod
    def _context_sources(prompt: str) -> list[dict[str, Any]]:
        """Parse le bloc 'Context: [N] doc p.x-y:\\ntext' du prompt en sources."""
        sources: list[dict[str, Any]] = []
        in_context = False
        current_text: list[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Context:"):
                in_context = True
                continue
            if stripped.startswith("Question:"):
                break
            if not in_context:
                continue
            if re.match(r"^\[\d+\]\s+", stripped):
                if current_text:
                    sources.append({"text": "\n".join(current_text)})
                    current_text = []
            else:
                current_text.append(stripped)
        if current_text:
            sources.append({"text": "\n".join(current_text)})
        return sources

    @staticmethod
    def _question(prompt: str) -> str:
        """Extrait la question à partir de la ligne 'Question:' du prompt."""
        match = re.search(r"^Question:\s*(.+)$", prompt, re.MULTILINE)
        return match.group(1).strip() if match else "this question"

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Compatibilité : rend une réponse ancrée depuis le prompt seul."""
        question = self._question(prompt)
        sources = self._context_sources(prompt)
        claims = extract_grounded_claims(question, sources)
        return decide_response(claims)

    def config_dict(self) -> dict[str, Any]:
        return {"provider": self.name, "model": "offline-grounded-claims"}


# ---------------------------------------------------------------------------
# Ancrage des affirmations (claims) : politique anti-hallucination
# ---------------------------------------------------------------------------

_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases (convention identique à citation_verifier)."""
    return [s.strip() for s in _SENTENCE_SPLITTER.split(text) if s.strip()]


def _tokenize(text: str) -> set[str]:
    """Jetons alphanumériques minuscules d'un texte (recouvrement lexical)."""
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _is_noisy_sentence(sent: str) -> bool:
    """Filtre les phrases non exploitables : références, captions, trop courtes."""
    if "arXiv:" in sent or "http" in sent:
        return True
    if len(sent) < CLAIM_MIN_CHARS or len(sent) > CLAIM_MAX_CHARS:
        return True
    if sum(ch.isalpha() for ch in sent) / max(len(sent), 1) < 0.55:
        return True
    lowered = sent.lstrip().lower()
    if lowered.startswith(
        (
            "figure", "table", "abstract", "fig. ", "tab. ", "algorithm ",
            "acknowledg", "references",
        )
    ):
        return True
    return False


def _best_claim_sentence(
    chunk_text: str, query_tokens: set[str],
) -> str | None:
    """Phrase du chunk la plus ancrée : recouvrement lexical max, puis la + courte."""
    best: str | None = None
    best_score = -1
    for sent in split_sentences(chunk_text):
        if _is_noisy_sentence(sent):
            continue
        score = len(_tokenize(sent) & query_tokens)
        if score > best_score or (
            score == best_score and best is not None and len(sent) < len(best)
        ):
            best = sent
            best_score = score
    return best


def _normalize_text(text: str) -> str:
    """Normalise les espaces pour les comparaisons de présence littérale."""
    return " ".join(text.split())


def extract_grounded_claims(
    query: str,
    sources: list[dict[str, Any]],
) -> list[Claim]:
    """Extrait au plus ``MAX_CLAIMS`` affirmations ancrées, une par source.

    Une affirmation n'est retenue que si elle est une phrase du texte d'une
    source (grounding littéral : chaque citation pointe vers une source qui
    contient mot pour mot l'affirmation).

    Args:
        query: question (son lexique sert de priorité de sélection).
        sources: sources ordonnées (top-1 en premier) avec clé ``text``.

    Returns:
        Liste ordonnée de ``Claim``, citations dédupliquées et triées.
    """
    if not sources:
        return []
    query_tokens = _tokenize(query)
    claims: list[Claim] = []
    for index, source in enumerate(sources, start=1):
        if len(claims) >= MAX_CLAIMS:
            break
        sent = _best_claim_sentence(str(source.get("text", "")), query_tokens)
        if sent is None:
            continue
        citations = [index]
        norm_sent = _normalize_text(sent)
        for other_index, other in enumerate(sources, start=1):
            if other_index != index and norm_sent in _normalize_text(
                str(other.get("text", ""))
            ):
                citations.append(other_index)
        claims.append(Claim(text=sent, citations=sorted(set(citations))))
    return claims[:MAX_CLAIMS]
def render_claims_answer(claims: list[Claim]) -> str:
    """Assemble les affirmations en une réponse 'grounded synthesis' (prose).

    Chaque fait ancré devient UNE phrase terminant par un point. Si le LLM a
    DÉJÀ cité en ligne (ex. 'as described in [2] and [3]'), aucun marqueur
    supplémentaire n'est ajouté : sinon ``segment_claims`` compterait les
    indices (2, 3, 2, 3) et produirait des lignes MNLI dupliquées. Les
    marqueurs [N] ne sont ajoutés que pour une phrase sans citation inline.

    Compatibilité Citation Verification : ``segment_claims`` découpe sur
    ``[.!?] + espace`` et ne garde que les phrases portant au moins une
    citation [N] ; chaque phrase produite respecte ces deux conditions.
    """
    sentences: list[str] = []
    for claim in claims:
        text = claim.text.rstrip()
        inline = {int(m.group(1)) for m in _CITATION_PATTERN.finditer(text)}
        if inline:
            # Citations déjà présentes dans la phrase : ne pas dupliquer.
            if not text.endswith((".", "!", "?")):
                text += "."
            sentences.append(text)
        else:
            text = text.rstrip(".").rstrip()
            markers = "".join(f"[{i}]" for i in sorted(claim.citations))
            sentences.append(f"{text} {markers}.")
    return " ".join(sentences)


def decide_response(claims: list[Claim]) -> str:
    """Politique de réponse : 3-5 affirmations ancrées, sinon refus explicite."""
    if MIN_CLAIMS_FOR_ANSWER <= len(claims) <= MAX_CLAIMS:
        return render_claims_answer(claims)
    return REFUSAL_RESPONSE


_NLI_VERIFIER: "MNLICitationVerifier | None" = None


def _get_nli_verifier() -> "MNLICitationVerifier":
    """Vérifieur MNLI partagé, chargé une seule fois (paresseux)."""
    global _NLI_VERIFIER
    if _NLI_VERIFIER is None:
        _NLI_VERIFIER = MNLICitationVerifier(device="auto")
    return _NLI_VERIFIER


def _gate_premise(chunk_text: str, claim_text: str) -> str:
    """Prémisse de gate : phrase du chunk la plus alignée avec le claim.

    Mesure (audit_union_gate.py) : roberta-large-mnli est entraîné sur des
    paires de PHRASES courtes. La fenêtre ±1 phrases d'``extract_local_premise``
    dilue l'entailment (claim pur : 0.3409 avec fenêtre vs 0.4817 avec la
    phrase de définition seule sur 2402.12317v2). La gate utilise donc la
    phrase à recouvrement lexical maximal ; fallback = fenêtre locale, puis
    chunk complet (comportement ``extract_local_premise``).
    """
    sentences = [s for s in split_sentences(chunk_text) if s.strip()]
    if not sentences:
        return extract_local_premise(chunk_text, claim_text)
    claim_tokens = _tokenize(claim_text)
    best_sentence, best_overlap = sentences[0], -1
    for sentence in sentences:
        overlap = len(_tokenize(sentence) & claim_tokens)
        if overlap > best_overlap:
            best_sentence, best_overlap = sentence, overlap
    if best_overlap <= 0:
        return extract_local_premise(chunk_text, claim_text)
    return best_sentence


def _best_supported_citations(
    claim_text: str,
    sources: list[dict[str, Any]],
    max_citations: int = 2,
    batch_size: int = 8,
) -> list[int]:
    """Sources qui soutiennent RÉELLEMENT le claim selon la NLI (gate).

    Score chaque paire (claim, source) avec la même prémisse que la gate
    (``_gate_premise``), la même formule (``compute_support_score``) et le
    même seuil (SUPPORT_LOW), puis retourne au plus ``max_citations`` indices
    (1-based) triés par support décroissant.

    Mesure clé (audit_nli_per_source.py) : le claim-réponse « RAG is a
    technique that allows LLMs to utilize external knowledge » atteint
    support=0.7091 sur la source [2] (Ragas) mais 0.0424 sur la source [1] à
    meilleur overlap lexical : la réparation des citations doit donc être
    SÉMANTIQUE (NLI), pas lexicale. Les scores sont enregistrés dans
    ``_LAST_NLI_CACHE`` (mêmes clés md5(premise||hypothesis) que la gate).

    Returns:
        Indices de sources (1-based) au meilleur support, <= max_citations ;
        liste vide si aucune source n'atteint SUPPORT_LOW.
    """
    if not _NLI_AVAILABLE or not sources:
        return []
    premises: list[str] = []
    indexes: list[int] = []
    for index, source in enumerate(sources, start=1):
        chunk_text = str(source.get("text", ""))
        if not chunk_text:
            continue
        premises.append(_gate_premise(chunk_text, claim_text))
        indexes.append(index)
    if not premises:
        return []
    probabilities = _get_nli_verifier().score_pairs(
        premises, [claim_text] * len(premises), batch_size=batch_size
    )
    scored: list[tuple[float, int]] = []
    for index, premise, probs in zip(indexes, premises, probabilities):
        support = compute_support_score(*probs)
        _LAST_NLI_CACHE[_pair_key(premise, claim_text)] = {
            "claim_text": claim_text,
            "source_index": index,
            "premise": premise,
            "hypothesis": claim_text,
            "support_score": support,
            "verdict": verdict_label(support),
        }
        if support >= SUPPORT_LOW:
            scored.append((support, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [index for _support, index in scored[:max_citations]]


def _all_citations_unsupported(
    answer: str,
    sources_map: list[dict[str, Any]],
    batch_size: int = 8,
) -> bool:
    """True ssi TOUTES les paires (segment, source citée) sont sous SUPPORT_LOW.

    Réutilise la segmentation et le scoring NLI de citation_verifier.py (sans
    le modifier) : même fenêtre locale (extract_local_premise), même formule
    de support (compute_support_score), même seuil (SUPPORT_LOW). Une seule
    paire suffisamment soutenue suffit à valider la réponse.

    Chaque score calculé est enregistré dans ``_LAST_NLI_CACHE`` (clé =
    md5(premise||hypothesis)) et consultable via ``get_last_nli_scores()``
    afin d'éviter toute nouvelle inférence NLI redondante en aval (voir la
    docstring de ``get_last_nli_scores`` pour le branchement futur).
    """
    if not _NLI_AVAILABLE:
        return False
    segments = segment_claims(answer)
    if not segments:
        return False
    premises: list[str] = []
    hypotheses: list[str] = []
    pair_meta: list[tuple[int, int]] = []  # (segment_index, source_index 1-based)
    for seg_idx, segment in enumerate(segments):
        for index in segment.citation_indices:
            if 1 <= index <= len(sources_map):
                chunk_text = str(sources_map[index - 1].get("text", ""))
                if not chunk_text:
                    continue
                premises.append(
                    _gate_premise(chunk_text, segment.claim_text)
                )
                hypotheses.append(segment.claim_text)
                pair_meta.append((seg_idx, index))
    if not premises:
        return False
    probabilities = _get_nli_verifier().score_pairs(
        premises, hypotheses, batch_size=batch_size
    )
    all_below = True
    for (seg_idx, src_idx), premise, hypothesis, (
        contradiction, neutral, entailment,
    ) in zip(pair_meta, premises, hypotheses, probabilities):
        support = compute_support_score(contradiction, neutral, entailment)
        _LAST_NLI_CACHE[_pair_key(premise, hypothesis)] = {
            "claim_text": segments[seg_idx].claim_text,
            "source_index": src_idx,
            "premise": premise,
            "hypothesis": hypothesis,
            "support_score": support,
            "verdict": verdict_label(support),
        }
        if support >= SUPPORT_LOW:
            all_below = False
    return all_below


def _claims_grounded_any_source(
    claims: list[Claim],
    sources_map: list[dict[str, Any]],
    threshold: float,
    batch_size: int = 8,
) -> bool:
    """True ssi CHAQUE claim est ancré dans le contexte ET non contredit.

    Deux gardes combinées (tolérance paraphrase) :
      1. COUVERTURE : chaque claim partage >= ``threshold`` de ses jetons avec
         AU MOINS une source (couverture max sur toutes les sources). C'est ce
         qui distingue une synthèse ancrée (vocabulaire présent dans le
         contexte) d'un écart de contexte (aucune source ne couvre le claim).
      2. CONTRADICTION : pour la meilleure source de couverture, la paire NLI
         (prémisse = phrase la plus alignée, ``_gate_premise``) ne doit pas
         être classée "contradiction" (P(contradiction) < 0.5) : un claim
         contredit par le contexte est un échec de grounding même s'il partage
         du vocabulaire.

    Cette tolérance contourne le biais "neutral" de roberta-large-mnli pour la
    paraphrase (support <= 0.24 < SUPPORT_LOW mesuré sur Q2/Q3) SANS relâcher
    la détection d'hallucination pure (claim hors-contexte => refus conservé).
    """
    if not claims:
        return False
    verifier = _get_nli_verifier()
    for claim in claims:
        tokens = _tokenize(claim.text)
        if not tokens:
            return False
        best_source_idx, best_coverage = -1, 0.0
        for index, source in enumerate(sources_map, start=1):
            source_tokens = _tokenize(str(source.get("text", "")))
            if not source_tokens:
                continue
            coverage = len(tokens & source_tokens) / len(tokens)
            if coverage > best_coverage:
                best_coverage, best_source_idx = coverage, index
        if best_source_idx < 0 or best_coverage < threshold:
            return False
        # Garde contradiction sur la meilleure source de couverture.
        premise = _gate_premise(str(sources_map[best_source_idx - 1].get("text", "")), claim.text)
        probs = verifier.score_pairs([premise], [claim.text], batch_size=1)[0]
        if probs[0] >= 0.5:  # P(contradiction) dominante => claim contredit
            _trace(
                "PARAPHRASE GATE: claim contredit par source[",
                best_source_idx, "] cov=", round(best_coverage, 2),
                "contradiction=", round(probs[0], 2),
            )
            return False
        _trace(
            "PARAPHRASE GATE: claim ancré source[", best_source_idx,
            "] cov=", round(best_coverage, 2),
            "contradiction=", round(probs[0], 2), "OK",
        )
    return True


def apply_nli_gate(
    query: str,
    claims: list[Claim],
    answer_text: str,
    sources_map: list[dict[str, Any]],
) -> tuple[list[Claim], str]:
    """Étape 3bis : gate NLI + tolérance paraphrase + repli extractif pertinent.

    Si TOUTES les citations de la réponse ont un support < SUPPORT_LOW :
      0. TOLÉRANCE PARAPHRASE (cause racine audit_qfocus, mesurée) :
         roberta-large-mnli juge la SYNTHESE comme "neutral" (support <= 0.24
         < SUPPORT_LOW, même contre TOUTES les sources -- replay_gate_best).
         Si chaque claim est ancré dans le contexte (couverture >=
         PARAPHRASE_COVERAGE_RESCUE, toute source confondue) et non contredit
         par la meilleure source (garde contradiction), la réponse LLM
         centre-question est CONSERVÉE au lieu du repli extractif (qui, lui,
         produit un assemblage FAiD/EVOR/RACG supporté mais hors question).
      1. sinon, tentative de repli extractif (claims verbatim, support NLI
         mesuré 0.78-0.89), filtrés par pertinence lexicale à la question ;
      2. refus explicite si MÊME l'extractif est non supporté (pool réellement
         sans contenu pertinent pour la question).

    Résidu documenté (mesuré replay_rescue) : un claim long construit à plus
    de 50 % de vocabulaire présent dans sa source citée peut franchir la
    tolérance même s'il contient un fragment contredit par la question (ex.
    "real-world images" pour une question RAG texte). La détection de ce type
    de contamination nécessiterait un filtre sémantique hors périmètre
    (retrieval/reranking, intouchables).

    Returns:
        (claims finaux, answer_text final) ; claims=[] si refus.
    """
    if answer_text == REFUSAL_RESPONSE:
        return [], answer_text
    if not _all_citations_unsupported(answer_text, sources_map):
        return claims, answer_text
    _trace("STEP5 NLI GATE: tous les supports LLM < SUPPORT_LOW")
    # --- Tolérance paraphrase (cause racine audit_qfocus) ---
    # Les claims répondent à la question mais MNLI évalue la reformulation
    # comme "neutral" (support <= 0.24). Si CHAQUE claim est ancré dans le
    # contexte (couverture >= PARAPHRASE_COVERAGE_RESCUE, toute source
    # confondue) et non contredit, on CONSERVE la réponse LLM : le repli
    # extractif ne produit, lui, qu'un assemblage supporté hors question
    # (FAiD / EVOR / RACG observés sur les 3 questions).
    if _claims_grounded_any_source(claims, sources_map, PARAPHRASE_COVERAGE_RESCUE):
        _trace("STEP5 NLI GATE: tolérance paraphrase => réponse LLM CONSERVÉE")
        return claims, answer_text
    fallback_claims = extract_grounded_claims(query, sources_map)
    query_tokens = _tokenize(query)
    fallback_claims = [
        c for c in fallback_claims
        if _tokenize(c.text) & query_tokens
    ]
    fallback_answer = decide_response(fallback_claims)
    if (
        fallback_answer != REFUSAL_RESPONSE
        and not _all_citations_unsupported(fallback_answer, sources_map)
    ):
        _trace("STEP5 NLI GATE: repli extractif SUPPORTED => utilisé")
        return fallback_claims, fallback_answer
    _trace("STEP5 NLI GATE: repli extractif non supporté => REFUS")
    return [], REFUSAL_RESPONSE


def _pair_key(premise: str, hypothesis: str) -> str:
    """Clé de cache stable pour une paire NLI (prémisse, hypothèse)."""
    return hashlib.md5(f"{premise}||{hypothesis}".encode()).hexdigest()


_LAST_NLI_CACHE: dict[str, dict[str, Any]] = {}


def verdict_label(support_score: float) -> str:
    """Verdict par seuils (identique à citation_verifier.verdict_for)."""
    if support_score >= SUPPORT_HIGH:
        return "Supported"
    if support_score >= SUPPORT_LOW:
        return "Weak Support"
    return "Unsupported"


def get_last_nli_scores() -> list[dict[str, Any]]:
    """Expose les scores NLI calculés par la dernière exécution du gate.

    BRANCHEMENT FUTUR (app.py NON modifié ici) : pour supprimer les ~5 s
    d'inférences NLI redondantes par requête, ``app.run_pipeline`` pourrait,
    avant d'appeler ``verify_citations()``, consulter ``get_last_nli_scores()``
    et réutiliser les enregistrements {claim_text, source_index,
    support_score, verdict} déjà produits par le gate de ``generate_answer``
    (mêmes clés md5(premise||hypothesis) que les paires de
    ``verify_citations``). Tant que cet appel n'est pas branché,
    ``verify_citations`` recalcule ses propres scores, sans risque de
    divergence : les deux chemins partagent fenêtre locale, formule et seuil.
    """
    return list(_LAST_NLI_CACHE.values())



def _parse_claims_regex(json_str: str) -> dict[str, Any]:
    """Fallback regex parser: extracts {"text":"...","citations":[N,...]} pairs.

    Used when ``json.loads`` fails entirely (e.g. LLM emits malformed JSON
    with unescaped control characters or interleaved text). Returns a dict
    with key ``"claims"`` compatible with the JSON parser output.
    """
    claims: list[dict[str, Any]] = []
    # Match each {"text":"...","citations":[...]} object.
    # The text is captured non-greedily up to the next "citations" key.
    pattern = re.compile(
        r'"text"\s*:\s*"(.*?)"\s*,\s*"citations"\s*:\s*\[([^\]]*)\]',
        re.DOTALL,
    )
    for match in pattern.finditer(json_str):
        text = match.group(1).strip()
        # Unescape basic JSON string sequences
        text = text.replace('\\"', '"').replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        text = text.strip()
        if not text:
            continue
        cite_strs = match.group(2).strip().split(",")
        citations: list[int] = []
        for s in cite_strs:
            s = s.strip().strip('"')
            try:
                citations.append(int(s))
            except (TypeError, ValueError):
                continue
        if citations:
            claims.append({"text": text, "citations": sorted(set(citations))})
    return {"claims": claims}


def parse_claim_json(raw: str) -> list[Claim]:
    """Parse une sortie LLM au format ``{"claims":[...]}`` (tolérant).

    Handles LLM outputs that contain literal control characters (\\r, \\n)
    inside JSON strings — a common output of qwen2.5 — by trying
    ``strict=False`` first, then falling back to a regex-based parser.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        _trace("PARSE_FAIL: no JSON object found in raw output")
        return []
    json_str = raw[start:end + 1]
    try:
        payload = json.loads(json_str, strict=False)
    except json.JSONDecodeError as exc:
        _trace("PARSE_FAIL: json.loads(strict=False) failed:", exc)
        _trace("PARSE_FAIL: attempting regex fallback on:", json_str[:200])
        payload = _parse_claims_regex(json_str)
    claims: list[Claim] = []
    for item in payload.get("claims") or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        citations: list[int] = []
        for index in item.get("citations") or []:
            try:
                citations.append(int(index))
            except (TypeError, ValueError):
                continue
        if citations:
            claims.append(Claim(text=text, citations=sorted(set(citations))))
    _trace("PARSE result: claims found =", len(claims))
    return claims


def _claim_grounded(
    claim: Claim, sources: list[dict[str, Any]], is_answer: bool = False,
) -> bool:
    """Vérifie qu'au moins une citation couvre >= seuil lexical des tokens.

    Seuil de GROUNDING_COVERAGE (0.45) pour les claims de preuve. La réponse
    DIRECTE (claims[0], is_answer=True) est une SYNTHÈSE qui reformule la
    définition : par nature sa couverture token est plus faible, elle est donc
    évaluée au seuil relâché ANSWER_GROUNDING_COVERAGE (0.15). Le contrôle
    sémantique reste assuré par la gate NLI en aval (apply_nli_gate).
    """
    threshold = ANSWER_GROUNDING_COVERAGE if is_answer else GROUNDING_COVERAGE
    tokens = _tokenize(claim.text)
    if not tokens:
        return False
    for index in claim.citations:
        if not 1 <= index <= len(sources):
            continue
        source_tokens = _tokenize(str(sources[index - 1].get("text", "")))
        if not source_tokens:
            continue
        coverage = len(tokens & source_tokens) / len(tokens)
        if coverage >= threshold:
            return True
    return False


def build_provider(provider: str, model: str) -> LLMProvider:
    """Instancie le fournisseur demandé ('template' ou 'ollama')."""
    if provider == "template":
        return TemplateProvider()
    if provider == "ollama":
        return OllamaProvider(model=model)
    raise SystemExit(
        f"Fournisseur '{provider}' inconnu. Choisir : template, ollama."
    )


# ---------------------------------------------------------------------------
# Construction du contexte / prompt
# ---------------------------------------------------------------------------

def build_prompt(query: str, sources: list[dict[str, Any]]) -> str:
    """Assemble le prompt final : contexte + question + consignes.

    Args:
        query: question en langage naturel.
        sources: liste ordonnée de dicts {document_id, page_start, page_end,
            text} (au plus MAX_SOURCES).
    """
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        text = str(source["text"])
        if len(text) > LOCAL_CORPUS_MAX_CHARS:
            text = text[:LOCAL_CORPUS_MAX_CHARS] + " [...]"
        blocks.append(
            f"[{index}] {source['document_id']} p.{source['page_start']}-"
            f"{source['page_end']}:\n{text}"
        )
    context = "\n\n".join(blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Instructions:\n{SYSTEM_PROMPT}"
    )


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(text: str, max_index: int) -> list[dict[str, Any]]:
    """Extrait les citations `[N]` de la réponse (borné à [1..max_index]).

    Retourne une liste d'objets {'reference': N} triée, sans doublon ; tout
    numéro hors [1..max_index] est ignoré (aucune citation n'est inventée).
    """
    seen: set[int] = set()
    for match in _CITATION_PATTERN.finditer(text):
        index = int(match.group(1))
        if 1 <= index <= max_index:
            seen.add(index)
    return [{"reference": index} for index in sorted(seen)]


def extract_sources_lines(text: str) -> dict[int, str]:
    """Parse les lignes 'Sources :' / '[N] ...' émises par le LLM.

    Returns:
        Mapping numéro -> ligne source brute (non vérifiée). Vide si aucune.
    """
    sources: dict[int, str] = {}
    in_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("sources"):
            in_block = True
            continue
        if in_block and line:
            match = re.match(r"^\[(\d+)\]\s+(.+)$", line)
            if match:
                sources[int(match.group(1))] = match.group(2)
    return sources


def sanitize_answer(raw: str) -> str:
    """Nettoie la sortie brute du LLM (strip lignes de commande/fin)."""
    lines = [
        line for line in raw.splitlines()
        if not line.lower().strip().startswith(
            ("sources", "[/", "[/INST", "assistant")
        )
    ]
    return "\n".join(lines).strip()
# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Citation:
    """Une citation extraite de la réponse (numéro de source)."""

    source_index: int


@dataclass(slots=True)
class Claim:
    """Une affirmation vérifiable, ancrée dans le contexte, avec ses sources."""

    text: str
    citations: list[int]


@dataclass(slots=True)
class SourceRef:
    """Une source citée : document + pages + drapeau d'extraction."""

    source_index: int
    document_id: str
    page_start: int
    page_end: int
    extracted: bool = False   # si la source apparaît effectivement citée


@dataclass(slots=True)
class AnswerResponse:
    """Réponse RAG complète : texte, affirmations, citations et latence."""

    query: str
    answer: str
    citations: list[Citation]
    sources: list[SourceRef]
    generation_latency: float
    claims: list[Claim]


def generate_answer(
    query: str,
    reranked_results: list[Any],
    chunk_index: dict[str, dict[str, Any]],
    provider: LLMProvider | None = None,
    temperature: float = 0.2,
) -> AnswerResponse:
    """Génère une réponse RAG citée à partir des chunks rerankés.

    Pipeline :
        1. construire la liste ordonnée des sources (document_id + pages) ;
        2. appeler le fournisseur LLM (claims ancrés) ;
        3. appliquer la politique 3-5 affirmations, sinon refus explicite ;
        4. construire AnswerResponse (answer, citations, sources, latence).

    Args:
        query: question en langage naturel.
        reranked_results: top-K final du reranker (RerankedResult).
        chunk_index: mapping chunk_id -> chunk (texte complet).
        provider: fournisseur LLM (défaut : TemplateProvider hors ligne).
        temperature: température de génération (Ollama).

    Returns:
        AnswerResponse avec réponse, citations ordonnées, sources et latence.
    """
    started_total = time.perf_counter()
    if provider is None:
        provider = TemplateProvider()

    # 1. Sources & contexte (au plus MAX_SOURCES, texte complet pour ancrage).
    sources_map: list[dict[str, Any]] = []
    for result in reranked_results[:MAX_SOURCES]:
        chunk = chunk_index.get(str(result.chunk_id))
        if chunk is None:
            continue
        sources_map.append({
            "document_id": str(result.document_id),
            "page_start": int(result.page_start),
            "page_end": int(result.page_end),
            "text": str(chunk.get("text", "")),
        })
    if not sources_map:
        raise SystemExit("Aucune source construite pour la génération.")

    # 2. Affirmations ancrées (politique anti-hallucination) — mesuré.
    started = time.perf_counter()
    claims = provider.generate_claims(query, sources_map, temperature=temperature)
    generation_ms = (time.perf_counter() - started) * 1000
    _trace("STEP5 AFTER_generate_claims claims=", [
        (c.text[:60], c.citations) for c in claims
    ])

    # 3. Politique : 3-5 affirmations vérifiables, sinon refus explicite.
    answer_text = decide_response(claims)
    _trace("STEP5 AFTER_decide_response answer>>>\n", answer_text, "\n<<<")
    # 3bis. Gate NLI + repli extractif pertinent (fonction réelle partagée
    # avec les audits : apply_nli_gate).
    claims, answer_text = apply_nli_gate(query, claims, answer_text, sources_map)
    if answer_text == REFUSAL_RESPONSE:
        claims = []

    # 4. Mapping explicite claim -> citations + sources citées.
    citations = [
        Citation(source_index=index)
        for claim in claims
        for index in claim.citations
    ]
    cited_indices = {citation.source_index for citation in citations}

    sources: list[SourceRef] = []
    for index, source in enumerate(sources_map, start=1):
        sources.append(SourceRef(
            source_index=index,
            document_id=source["document_id"],
            page_start=source["page_start"],
            page_end=source["page_end"],
            extracted=index in cited_indices,
        ))

    total_ms = (time.perf_counter() - started_total) * 1000
    result = AnswerResponse(
        query=query,
        answer=answer_text,
        citations=citations,
        sources=sources,
        generation_latency=round(total_ms, 2),
        claims=claims,
    )
    _trace("STEP6 FINAL answer renvoyé à l'app>>>\n", result.answer, "\n<<<")
    return result
def aggregate_generation(responses: list[AnswerResponse]) -> dict[str, Any]:
    """Agrège les métriques de génération de toutes les requêtes."""
    count = len(responses)
    if not count:
        return {"queries": 0}
    return {
        "queries": count,
        "mean_generation_ms": round(
            sum(r.generation_latency for r in responses) / count, 2
        ),
        "max_generation_ms": max(r.generation_latency for r in responses),
        "total_citations": sum(len(r.citations) for r in responses),
        "mean_citations_per_answer": round(
            sum(len(r.citations) for r in responses) / count, 2
        ),
    }


def assess_citation_verification_readiness(
    responses: list[AnswerResponse],
    reranked_results_lists: list[list[Any]],
) -> dict[str, Any]:
    """Vérifie que la sortie est prête pour la vérification NLI des citations.

    Contrôles : réponse non vide, citations présentes, indices dans le pool.
    Ce n'est PAS la vérification elle-même.
    """
    problems: list[str] = []
    answers_with_citations = 0
    for index, (response, reranked) in enumerate(
        zip(responses, reranked_results_lists)
    ):
        if not response.answer.strip():
            problems.append(f"requête {index} : réponse vide")
            continue
        if response.citations:
            answers_with_citations += 1
            for citation in response.citations:
                if citation.source_index > len(reranked):
                    problems.append(
                        f"requête {index} : citation "
                        f"{citation.source_index} hors pool"
                    )
        else:
            problems.append(f"requête {index} : aucune citation [N] détectée")
    ready = not problems and len(responses) > 0
    return {
        "ready_for_citation_verification": ready,
        "answers_with_citations": answers_with_citations,
        "answers_total": len(responses),
        "problems": problems[:EXAMPLE_LIMIT],
    }


def build_session_report(
    provider: LLMProvider,
    queries: list[str],
    responses: list[AnswerResponse],
    reranked_results_lists: list[list[Any]],
    chunks_sha256: str,
) -> dict[str, Any]:
    """Construit le rapport structuré de la session de génération."""
    readiness = assess_citation_verification_readiness(
        responses, reranked_results_lists
    )
    problems = readiness["problems"]
    if not queries:
        status = Status.FAIL.value
    elif readiness["ready_for_citation_verification"]:
        status = Status.PASS.value
    elif problems:
        status = Status.WARNING.value
    else:
        status = Status.FAIL.value

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": status,
        "provider": provider.config_dict(),
        "chunks_sha256": chunks_sha256,
        "generation": {
            "queries": len(queries),
            "answers_with_citations": readiness["answers_with_citations"],
            "problems": problems,
            "aggregate_generation": aggregate_generation(responses),
        },
        "citation_verification_readiness": readiness,
        "versions": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "requests_version": requests.__version__,
        },
        "queries": [
            {
                "query": response.query,
                "answer": response.answer,
                "claims": [
                    {"text": claim.text, "citations": list(claim.citations)}
                    for claim in response.claims
                ],
                "citations": [asdict(citation) for citation in response.citations],
                "sources": [
                    {
                        "document_id": source.document_id,
                        "page_start": source.page_start,
                        "page_end": source.page_end,
                        "extracted": source.extracted,
                    }
                    for source in response.sources
                ],
                "generation_latency_ms": response.generation_latency,
            }
            for response in responses
        ],
    }


def write_session_report(corpus_dir: Path, report: dict[str, Any]) -> Path:
    """Écrit generation_report.json sous corpus/ et retourne son chemin."""
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
    """Parse les arguments CLI de la session de génération."""
    parser = argparse.ArgumentParser(
        description=(
            "Génère une réponse RAG citée à partir du top-5 reranké "
            "(rapport : generation_report.json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sans requête fournie, exécute les 3 requêtes de test du projet.\n"
            "Exemples :\n"
            '  python generate_answer.py --provider template\n'
            '  python generate_answer.py --provider ollama --model llama3.1 '
            '"How does ReAct work?"'
        ),
    )
    parser.add_argument(
        "queries", nargs="*", default=None,
        help="Requêtes à exécuter (défaut : 3 requêtes de test).",
    )
    parser.add_argument(
        "--provider", choices=("template", "ollama"), default="template",
        help="Fournisseur LLM (défaut : template hors-ligne).",
    )
    parser.add_argument(
        "--model", default="llama3.1",
        help="Modèle du fournisseur ollama (défaut : llama3.1).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Température de génération (défaut : 0.2).",
    )
    parser.add_argument(
        "--pool-size", type=int, default=20,
        help="Candidats hybrides en amont (20).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top final après reranking (5).",
    )
    parser.add_argument(
        "--log-format", choices=("text", "json"), default="text",
        help="Format des logs console (défaut : text).",
    )
    return parser.parse_args()


def _console_safe(text: str) -> str:
    """Rend un texte affichable sur la console courante (Windows cp1252 inclus)."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_answer(response: AnswerResponse) -> None:
    """Affiche réponse + mapping claims + bloc Sources + latence dans la console."""
    print(_console_safe(f"\nQUERY: {response.query}"))
    print(_console_safe(f"ANSWER: {response.answer}"))
    if response.claims:
        print("CLAIMS (claim -> citations):")
        for claim in response.claims:
            print(_console_safe(f"  - {claim.text} -> {claim.citations}"))
    else:
        print("CLAIMS: aucune (réponse de refus ou contexte insuffisant)")
    cited = [source for source in response.sources if source.extracted]
    if cited:
        print("\nSources :")
        for source in cited:
            print(
                _console_safe(
                    f"[{source.source_index}] {source.document_id} "
                    f"p.{source.page_start}-{source.page_end}"
                )
            )
    else:
        print("Sources : aucune source citée.")
    print(f"LATENCE génération : {response.generation_latency} ms")
def main() -> None:
    """Point d'entrée : hybrid search -> reranking -> génération + rapport."""
    args = parse_args()
    _, events = build_logger(as_json=args.log_format == "json")

    if args.pool_size < args.top_k:
        raise SystemExit("--pool-size doit être >= --top-k.")
    if not 0.0 <= args.temperature <= 1.0:
        raise SystemExit("--temperature doit être dans [0 ; 1].")

    queries = list(args.queries) if args.queries else list(DEFAULT_TEST_QUERIES)
    chunks_path = locate_corpus(INPUT_FILENAME)
    if chunks_path is None:
        raise SystemExit(f"{INPUT_FILENAME} introuvable sous corpus/.")
    chunks_sha256 = sha256_of(chunks_path)
    chunk_index = load_chunk_text_index(chunks_path)[0]

    # Étape amont : hybrid search -> reranking (réutilise les scripts existants).
    from hybrid_search import EngineConfig, get_engine
    from rerank_results import CrossEncoderReranker, rerank_results

    engine = get_engine(EngineConfig(fetch_k=max(args.pool_size, args.top_k)))
    try:
        hybrid_responses = [
            engine.search(query=query, top_k=args.pool_size) for query in queries
        ]
    finally:
        engine.close()
    reranker = CrossEncoderReranker(device="auto")
    reranked_lists = [
        rerank_results(
            query=query,
            hybrid_results=hybrid_response.results,
            chunk_index=chunk_index,
            reranker=reranker,
            top_k=args.top_k,
            pool_size=args.pool_size,
        ).results
        for query, hybrid_response in zip(queries, hybrid_responses)
    ]

    # Génération.
    provider = build_provider(args.provider, args.model)
    events.info(
        "generation_started",
        provider=provider.config_dict(),
        queries=len(queries),
        temperature=args.temperature,
    )
    responses = [
        generate_answer(
            query, reranked, chunk_index,
            provider=provider, temperature=args.temperature,
        )
        for query, reranked in zip(queries, reranked_lists)
    ]
    for response in responses:
        _print_answer(response)

    report = build_session_report(
        provider=provider,
        queries=queries,
        responses=responses,
        reranked_results_lists=reranked_lists,
        chunks_sha256=chunks_sha256,
    )
    report_path = write_session_report(chunks_path.parent, report)

    aggregate = report["generation"]["aggregate_generation"]
    readiness = report["citation_verification_readiness"]
    events.info(
        "session_completed",
        status=report["status"],
        mean_generation_ms=aggregate.get("mean_generation_ms"),
        total_citations=aggregate.get("total_citations"),
    )
    events.info(
        "citation_verification_readiness",
        ready=readiness["ready_for_citation_verification"],
        answers_with_citations=readiness["answers_with_citations"],
        problems=len(readiness["problems"]),
    )
    print(f"\n=> rapport : {report_path}")
    sys.exit(EXIT_OK if report["status"] == Status.PASS.value else EXIT_FAIL)


if __name__ == "__main__":
    main()