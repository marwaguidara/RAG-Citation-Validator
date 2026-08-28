"""
RAG Citation Validator — Interface finale (Streamlit).

Workflow exposé (repose 100 % sur les modules backend existants, aucune
logique métier n'est dupliquée ici) :

    Question
        -> Hybrid Search            (hybrid_search.HybridSearchEngine)
        -> Reranker                 (rerank_results.rerank_results)
        -> Generation               (generate_answer.generate_answer)
        -> Citation Verification    (citation_verifier.verify_citations)
        -> Réponse finale citée

Pour chaque citation de la réponse, l'interface affiche :
    - le support score
    - le verdict (Supported / Weak Support / Unsupported)
    - le document source, les pages
    - le thème
    - le temps d'exécution total

Ce module ne fait que : charger les ressources (cache), appeler le pipeline,
et rendre les résultats. Il ne contient aucun calcul de score, aucune règle
métier — tout est délégué aux modules `files/`.

Lancement :
    streamlit run app.py

Note de reproductibilité : le générateur par défaut est le `TemplateProvider`
hors ligne (déterministe, aucune clé API, aucun Ollama requis) afin que la
démo fonctionne partout. Un onglet latéral permet de basculer vers Ollama si
disponible.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Chargement des modules backend (répertoire files/)
# ---------------------------------------------------------------------------

FILES_DIR = Path(__file__).resolve().parent / "files"
CORPUS_FILENAME = "chunks.json"

sys.path.insert(0, str(FILES_DIR))

from hybrid_search import EngineConfig, get_engine  # noqa: E402
from rerank_results import CrossEncoderReranker, rerank_results  # noqa: E402
from generate_answer import MAX_SOURCES, build_provider, generate_answer, load_chunk_text_index, locate_corpus  # noqa: E402
from citation_verifier import MNLICitationVerifier, verify_citations  # noqa: E402
# ---------------------------------------------------------------------------
# Ressources mises en cache (chargées une seule fois par session Streamlit)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Chargement de l'index des chunks…")
def get_chunk_index() -> dict[str, dict[str, Any]]:
    """Charge et met en cache le mapping chunk_id -> {texte, thème, pages}."""
    corpus_path = locate_corpus(CORPUS_FILENAME)
    if corpus_path is None:
        corpus_path = FILES_DIR / "corpus" / CORPUS_FILENAME
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus introuvable : {corpus_path}")
    index, _meta = load_chunk_text_index(corpus_path)
    return index


@st.cache_resource(show_spinner="Chargement du moteur hybride…")
def get_engine_resource(pool_size: int) -> Any:
    """Construit et met en cache le moteur de recherche hybride."""
    return get_engine(EngineConfig(fetch_k=pool_size))


@st.cache_resource(show_spinner="Chargement du reranker…")
def get_reranker() -> CrossEncoderReranker:
    """Construit et met en cache le reranker cross-encoder."""
    return CrossEncoderReranker(device="auto")


@st.cache_resource(show_spinner="Chargement du vérificateur NLI…")
def get_verifier() -> MNLICitationVerifier:
    """Construit et met en cache le vérificateur de citations (roberta-large-mnli)."""
    return MNLICitationVerifier(device="auto")


# ---------------------------------------------------------------------------
# Orchestration du pipeline (zéro logique métier dupliquée)
# ---------------------------------------------------------------------------


def run_pipeline(
    query: str,
    *,
    pool_size: int,
    top_k: int,
    provider_name: str,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    """Exécute le pipeline complet et retourne un dictionnaire de rendu.

    Args:
        query: question en langage naturel de l'utilisateur.
        pool_size: nombre de chunks extraits par la recherche hybride.
        top_k: nombre de chunks retenus par le reranker (<= MAX_SOURCES).
        provider_name: "template" (hors ligne) ou "ollama".
        model_name: modèle LLM (utilisé uniquement si Ollama).
        temperature: température de génération.

    Returns:
        Dictionnaire contenant la réponse, les étapes, les citations
        vérifiées et le temps d'exécution total.
    """
    started = time.perf_counter()

    chunk_index = get_chunk_index()
    engine = get_engine_resource(pool_size)
    reranker = get_reranker()
    verifier = get_verifier()
    provider = build_provider(provider_name, model_name)

    # 1) Hybrid search
    hybrid = engine.search(query=query, top_k=pool_size)

    # 2) Reranking
    reranked = rerank_results(
        query=query,
        hybrid_results=hybrid.results,
        chunk_index=chunk_index,
        reranker=reranker,
        top_k=top_k,
        pool_size=pool_size,
    )

    # 3) Génération (réponse citée)
    answer_response = generate_answer(
        query=query,
        reranked_results=reranked.results,
        chunk_index=chunk_index,
        provider=provider,
        temperature=temperature,
    )

    # 4) Sources pour la vérification : mêmes slices / ordre que la génération.
    #    On ajoute chunk_id + theme à chaque source pour l'affichage.
    sources: list[dict[str, Any]] = []
    source_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for result in reranked.results[:MAX_SOURCES]:
        if result.chunk_id not in chunk_index:
            continue
        src = {
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "page_start": result.page_start,
            "page_end": result.page_end,
            "theme": result.theme,
        }
        sources.append(src)
        source_lookup[(result.document_id, result.page_start, result.page_end)] = src

    # 5) Vérification des citations
    verified, _segments = verify_citations(
        query=query,
        answer=answer_response.answer,
        sources=sources,
        chunk_index=chunk_index,
        verifier=verifier,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "query": query,
        "answer": answer_response.answer,
        "claims": list(answer_response.claims),
        "verified": verified,
        "source_lookup": source_lookup,
        "reranked_results": list(reranked.results),
        "hybrid_latency_ms": hybrid.latencies_ms.total_ms,
        "rerank_latency_ms": reranked.metrics.total_ms,
        "generation_latency_ms": answer_response.generation_latency,
        "total_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Rendu Streamlit
# ---------------------------------------------------------------------------


def render_result(result: dict[str, Any]) -> None:
    """Affiche la réponse, les citations vérifiées et les métadonnées."""
    answer = result["answer"]
    verified = result["verified"]
    lookup = result["source_lookup"]

    st.markdown("### 💬 Réponse générée")
    st.markdown(answer)

    st.markdown("### 🔎 Citations vérifiées")
    if not verified:
        st.info("Aucune citation vérifiable dans la réponse (réponse de refus).")
    else:
        rows = []
        for v in verified:
            src = lookup.get((v.document_id, v.page_start, v.page_end), {})
            theme = src.get("theme", "—")
            rows.append({
                "Claim": v.claim_text,
                "Support score": f"{v.support_score:.2f}",
                "Verdict": v.verdict,
                "Document": v.document_id,
                "Pages": f"{v.page_start}-{v.page_end}",
                "Thème": theme,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### ⏱️ Temps d'exécution")
    lat_cols = st.columns(4)
    lat_cols[0].metric("Hybrid search", f"{result['hybrid_latency_ms']:.0f} ms")
    lat_cols[1].metric("Reranking", f"{result['rerank_latency_ms']:.0f} ms")
    lat_cols[2].metric("Génération", f"{result['generation_latency_ms']:.0f} ms")
    lat_cols[3].metric("Total", f"{result['total_ms']:.0f} ms")
def main() -> None:
    """Point d'entrée de l'interface Streamlit."""
    st.set_page_config(
        page_title="RAG Citation Validator",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 RAG Citation Validator")
    st.caption(
        "Hybrid Search → Reranker → Génération → Vérification des citations "
        "(roberta-large-mnli)."
    )

    # Paramètres latéraux
    with st.sidebar:
        st.header("⚙️ Paramètres")
        provider_name = st.radio(
            "Fournisseur LLM",
            options=["template", "ollama"],
            index=0,
            help="template = hors ligne (déterministe, aucune clé API).",
        )
        model_name = st.text_input("Modèle (Ollama)", value="qwen2.5:3b")
        temperature = st.slider("Température", 0.0, 1.0, 0.2, 0.05)
        pool_size = st.slider("Pool (hybrid search)", 10, 50, 20, 5)
        top_k = st.slider("Top-K (reranker)", 3, MAX_SOURCES, 5, 1)
        st.caption(f"Top-K limité à MAX_SOURCES = {MAX_SOURCES} (borné par le pipeline).")

    # Zone question
    query = st.text_input(
        "Votre question",
        placeholder="Ex. : Qu'est-ce que le RAG ?",
    )
    run = st.button("🚀 Exécuter le pipeline", type="primary")

    if run and query.strip():
        with st.spinner("Exécution du pipeline RAG…"):
            try:
                result = run_pipeline(
                    query.strip(),
                    pool_size=int(pool_size),
                    top_k=int(top_k),
                    provider_name=provider_name,
                    model_name=model_name,
                    temperature=float(temperature),
                )
            except Exception as exc:  # pragma: no cover - affichage d'erreur
                st.error(f"Erreur lors de l'exécution : {exc}")
                return
        render_result(result)
    elif run:
        st.warning("Veuillez saisir une question.")


if __name__ == "__main__":
    main()