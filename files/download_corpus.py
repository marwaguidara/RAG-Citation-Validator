"""
Script de téléchargement automatique du corpus RAG (papers arXiv).

Utilise l'API officielle arXiv (gratuite, sans clé) pour chercher et télécharger
des PDF classés par thème : RAG, Agents, Fine-tuning.

Usage :
    python download_corpus.py

Résultat :
    corpus/
      rag/            (~15-17 PDF)
      agents/          (~15-17 PDF)
      fine_tuning/     (~15-17 PDF)
      manifest.json    (métadonnées de tous les papers téléchargés)

IMPORTANT : ce script doit être exécuté sur TA machine locale, pas dans un
environnement sandbox sans accès à arxiv.org.
"""

import os
import re
import time
import json
import ssl
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    print("ATTENTION : le paquet 'certifi' n'est pas installé (pip install certifi).")
    print("Les requêtes HTTPS risquent d'échouer avec une erreur de certificat SSL.\n")
    SSL_CONTEXT = ssl.create_default_context()

# --- Configuration ---

OUTPUT_DIR = "corpus"
TARGET_PER_THEME = 16          # nombre de papers visés par thème
REQUEST_DELAY_SECONDS = 3      # arXiv demande un délai raisonnable entre requêtes
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Plusieurs requêtes par thème pour couvrir des sous-sujets variés
# et éviter les doublons/le bruit d'une seule requête trop générique.
THEMES = {
    "rag": [
        "retrieval augmented generation",
        "dense passage retrieval question answering",
        "hybrid retrieval reranking RAG",
        "RAG survey knowledge intensive NLP",
    ],
    "agents": [
        "large language model agents reasoning",
        "tool use language models",
        "multi-agent LLM collaboration",
        "ReAct reasoning acting language model",
    ],
    "fine_tuning": [
        "parameter efficient fine-tuning language models",
        "LoRA low rank adaptation",
        "instruction tuning large language models",
        "reinforcement learning human feedback fine-tuning",
    ],
}

NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def sanitize_filename(title: str, arxiv_id: str) -> str:
    """Transforme un titre en nom de fichier propre."""
    short_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    short_title = short_title[:60]
    return f"{arxiv_id}_{short_title}.pdf"


def search_arxiv(query: str, max_results: int = 10):
    """Interroge l'API arXiv et retourne une liste de résultats (dict)."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=30, context=SSL_CONTEXT) as response:
        raw_xml = response.read()

    root = ET.fromstring(raw_xml)
    entries = []
    for entry in root.findall("atom:entry", NAMESPACE):
        arxiv_url = entry.find("atom:id", NAMESPACE).text.strip()
        arxiv_id = arxiv_url.split("/abs/")[-1]
        title = entry.find("atom:title", NAMESPACE).text.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        summary = entry.find("atom:summary", NAMESPACE).text.strip().replace("\n", " ")
        authors = [
            a.find("atom:name", NAMESPACE).text
            for a in entry.findall("atom:author", NAMESPACE)
        ]
        pdf_url = None
        for link in entry.findall("atom:link", NAMESPACE):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None:
            # fallback : construire l'URL PDF depuis l'ID
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "summary": summary,
            "pdf_url": pdf_url,
        })
    return entries


def download_pdf(pdf_url: str, dest_path: str):
    """Télécharge un PDF vers dest_path."""
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
        data = response.read()
    with open(dest_path, "wb") as f:
        f.write(data)


def build_corpus():
    manifest = []

    for theme, queries in THEMES.items():
        theme_dir = os.path.join(OUTPUT_DIR, theme)
        os.makedirs(theme_dir, exist_ok=True)

        seen_ids = set()
        collected = []

        for query in queries:
            if len(collected) >= TARGET_PER_THEME:
                break

            print(f"[{theme}] Recherche : '{query}'")
            try:
                results = search_arxiv(query, max_results=10)
            except Exception as e:
                print(f"  Erreur lors de la recherche : {e}")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            for r in results:
                if len(collected) >= TARGET_PER_THEME:
                    break
                if r["arxiv_id"] in seen_ids:
                    continue
                seen_ids.add(r["arxiv_id"])
                collected.append(r)

            time.sleep(REQUEST_DELAY_SECONDS)  # respect du rate limit arXiv

        # Téléchargement des PDF collectés pour ce thème
        for r in collected:
            filename = sanitize_filename(r["title"], r["arxiv_id"])
            dest_path = os.path.join(theme_dir, filename)

            if os.path.exists(dest_path):
                print(f"  Déjà téléchargé : {filename}")
            else:
                print(f"  Téléchargement : {r['title'][:70]}...")
                try:
                    download_pdf(r["pdf_url"], dest_path)
                except Exception as e:
                    print(f"    Échec du téléchargement : {e}")
                    continue
                time.sleep(REQUEST_DELAY_SECONDS)

            manifest.append({
                "theme": theme,
                "arxiv_id": r["arxiv_id"],
                "title": r["title"],
                "authors": r["authors"],
                "summary": r["summary"],
                "pdf_url": r["pdf_url"],
                "local_path": dest_path.replace(os.sep, "/"),  # chemin portable (JOUR 2)
            })

        print(f"[{theme}] Total collecté : {len(collected)} papers\n")

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Corpus complet. Manifest sauvegardé dans : {manifest_path}")
    print(f"Total de documents téléchargés : {len(manifest)}")


if __name__ == "__main__":
    build_corpus()