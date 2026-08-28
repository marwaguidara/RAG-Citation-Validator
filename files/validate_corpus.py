"""
Validation et nettoyage du corpus RAG (JOUR 2).

Rôles :
1. Vérifier l'intégrité des PDF téléchargés au JOUR 1
   (signature, taille, marqueur EOF, ouverture PyMuPDF).
2. Valider la cohérence du manifest : thème, arxiv_id, chemins, doublons.
3. Détecter les papiers « hors-sujet » (heuristique simple sur le titre) → ils sont
   signalés pour relecture manuelle, jamais supprimés silencieusement.
4. Contrôler la qualité du texte extrait : pages vides, encodage cassé (mojibake),
   PDF scannés sans couche texte et texte bruité.
5. Mettre en quarantaine les fichiers invalides / doublons (option --quarantine).
6. Régénérer un manifest propre et normalisé + un rapport de validation.

Usage :
    python validate_corpus.py [--quarantine] [--quarantine-offtopic]

Sorties :
    corpus/manifest.json           manifest nettoyé (entrées valides uniquement)
    corpus/validation_report.json  rapport machine (réutilisable pour J3+)
    corpus/validation_report.md    rapport lisible
    corpus/_quarantine/            fichiers invalides déplacés (si --quarantine)

Par défaut : dry-run (aucun fichier déplacé). Le manifest nettoyé et le rapport
sont TOUJOURS écrits : ce sont les artefacts canoniques pour J3+.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THEMES = {"rag", "agents", "fine_tuning"}

MIN_PDF_SIZE_BYTES = 50_000   # en dessous : téléchargement probablement corrompu
EOF_LOOKBACK_BYTES = 2048     # le marqueur %%EOF doit être dans les derniers octets
PDF_MAGIC = b"%PDF"
EOF_MARKER = b"%%EOF"

# --- Contrôle qualité du texte extrait (étape 4) ---
MIN_EMPTY_PAGE_CHARS = 50        # en dessous : page considérée vide (figure, remerciements…)
MIN_TOTAL_CHARS = 4000           # en dessous : couche texte absente (scan / OCR manquant)
MIN_AVG_CHARS_PER_PAGE = 400     # moyenne basse → couche texte partielle (image + texte)
GARBAGE_LETTER_RATIO_MIN = 0.40  # ratio de lettres parmi le texte hors espaces -> sinon bruit
MOJIBAKE_RE = re.compile(
    r"[\ufffd]"                      # caractère de remplacement U+FFFD
    r"|Ã[\x80-\xbf]"                 # é -> "Ã©"
    r"|â[\x80-\xbf][\x80-\xbf]"      # ' -> "â€™"
    r"|Â[\x80-\xbf]"                 # espace insécable -> "Â "
)
# Un vrai mojibake cp1252 produit des centaines de C1 ; quelques isolés
# (artefacts de glyphes PDF) ne sont PAS du mojibake.
C1_HEAVY_THRESHOLD = 25

# Mots-clés par thème (heuristique simple et transparente ; à réviser en J5 avec
# un vrai classifieur si le besoin s'en fait). Correspondance sur le titre.
THEME_KEYWORDS: dict[str, list[str]] = {
    "rag": [
        "rag", "retriev", "passage", "rerank", "query",
        "generation", "knowledge", "retriever", "corpus", "index", "fact",
    ],
    "agents": [
        "agent", "tool", "plan", "reason", "react", "llm",
        "multi-agent", "environment", "action", "collaborat",
    ],
    "fine_tuning": [
        "fine-tun", "finetun", "lora", "peft", "adapter", "instruction tun",
        "parameter-efficient", "pretrain", "transfer", "gradient",
    ],
}


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """Une anomalie détectée sur une entrée du corpus."""
    entry_id: str
    title: str
    kind: str
    message: str
    action: str                 # "reported" | "quarantined"
    path: Optional[Path] = None  # fichier concerné (pour la mise en quarantaine)


@dataclass
class ManifestEntry:
    """Entrée de manifest normalisée (artefact stable pour J3+)."""
    id: str
    theme: str
    title: str
    authors: list[str]
    summary: str
    pdf_url: str
    pages: int
    file_path: str              # chemin portable (relatif au corpus)
    status: str                 # valid | invalid | duplicate | flagged | dropped
    relevance: dict = field(default_factory=dict)
    text_quality: dict = field(default_factory=dict)
    path: Optional[Path] = None  # chemin absolu résolu (usage interne)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_arxiv_id(raw: str) -> str:
    """'2506.06962v3' -> '2506.06962' (supprime le suffixe de version)."""
    return re.sub(r"v\d+$", "", raw.strip().lower())


def version_sort_key(arxiv_id: str) -> tuple[int, int, int]:
    """'2401.12345v3' -> (2401, 12345, 3) pour comparer/trier les versions."""
    m = re.fullmatch(r"(\d{4})\.(\d{4,5})(?:v(\d+))?", arxiv_id.strip().lower())
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))


def normalize_title(raw: Any) -> str:
    """Titre nettoyé : espaces multiples écrasés, sans retour à la ligne."""
    return re.sub(r"\s+", " ", str(raw or "")).strip()


def normalize_authors(raw: Any) -> list[str]:
    """Auteurs nettoyés : espaces en tête/queue retirés, doublons supprimés."""
    seen: set[str] = set()
    authors: list[str] = []
    for a in raw or []:
        name = str(a).strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            authors.append(name)
    return authors


def theme_relevance(title: str, theme: str) -> list[str]:
    """Mots-clés du thème présents dans le titre (heuristique)."""
    t = title.lower()
    return [kw for kw in THEME_KEYWORDS.get(theme, []) if kw in t]


def best_theme(title: str) -> tuple[str, int]:
    """(thème, score) le mieux représenté dans le titre, tous thèmes confondus."""
    scores = [(theme, len(theme_relevance(title, theme))) for theme in THEMES]
    theme, score = max(scores, key=lambda x: x[1])
    return theme, score


def locate_corpus() -> tuple[Path, Path]:
    """Retourne (corpus_root, manifest_path) via plusieurs ancrages (script / CWD)."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "corpus" / "manifest.json",
        Path.cwd() / "corpus" / "manifest.json",
        script_dir / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent, candidate
    raise SystemExit(
        "manifest.json introuvable. Lancez le script depuis files/ "
        "ou après avoir généré le corpus (JOUR 1)."
    )


def resolve_path(raw_path: str, corpus_root: Path) -> Optional[Path]:
    """Résout un chemin de fichier PDF de façon robuste.

    Le manifest hérité du JOUR 1 contient des chemins en backslashes relatifs
    à files/. On essaie plusieurs ancrages, puis une recherche par nom de fichier
    (les noms sont uniques grâce au préfixe arxiv_id).
    """
    filename = Path(raw_path).name
    candidates = [
        Path(raw_path),
        Path.cwd() / raw_path,
        corpus_root / filename,
        Path.cwd() / "files" / raw_path,
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            pass
    for hit in corpus_root.rglob(filename):
        if hit.is_file():
            return hit.resolve()
    return None


def inspect_pdf(path: Path, raw_id: str, title: str, issues: list[ValidationIssue]) -> tuple[bool, int]:
    """Contrôle léger (signature, taille, %%EOF) puis ouverture PyMuPDF.

    Retourne (valide, nombre_de_pages). PyMuPDF permet de valider la structure
    sans extraire le texte (l'extraction complète arrive au JOUR 3).
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                      f"stat impossible : {exc}", "reported", path))
        return False, 0

    if size < MIN_PDF_SIZE_BYTES:
        issues.append(ValidationIssue(raw_id, title, "too_small",
                                      f"fichier anormalement petit ({size} o)", "reported", path))

    try:
        with path.open("rb") as handle:
            head = handle.read(8)
            handle.seek(0, 2)
            total = handle.tell()
            handle.seek(max(0, total - EOF_LOOKBACK_BYTES))
            tail = handle.read()
    except OSError as exc:
        issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                      f"lecture impossible : {exc}", "reported", path))
        return False, 0

    ok = True
    if not head.startswith(PDF_MAGIC):
        issues.append(ValidationIssue(raw_id, title, "bad_signature",
                                      "signature %PDF absente", "reported", path))
        ok = False
    if EOF_MARKER not in tail:
        issues.append(ValidationIssue(raw_id, title, "no_eof",
                                      "marqueur %%EOF absent", "reported", path))
        ok = False

    pages = 0
    try:
        with fitz.open(str(path)) as doc:
            if not doc.is_pdf:
                issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                              "non reconnu comme PDF", "reported", path))
                return False, 0
            if doc.needs_pass:
                issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                              "PDF chiffré", "reported", path))
                return False, 0
            pages = doc.page_count
    except Exception as exc:  # noqa: BLE001 — toute erreur d'ouverture = PDF illisible
        issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                      f"illisible par PyMuPDF : {exc}", "reported", path))
        return False, 0

    if pages <= 0:
        issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                      "0 page détectée", "reported", path))
        issues.append(ValidationIssue(raw_id, title, "corrupt_pdf",
                                      "0 page détectée", "reported", path))
        return False, 0
    return ok, pages


def inspect_text_quality(path: Path, raw_id: str, title: str, issues: list[ValidationIssue]) -> dict:
    """Extrait le texte du PDF et contrôle sa qualité (étape 4).

    Détecte : PDF scannés sans couche texte (`no_text` / `too_low_text`),
    pages vides, texte bruité (`garbage_text`) et encodage cassé (`mojibake`).

    Ne modifie rien ; alimente le champ `text_quality` du manifest et des
    `ValidationIssue` rapportées (voir rapport de validation).
    """
    per_page_chars: list[int] = []

    try:
        with fitz.open(str(path)) as doc:
            full_text_parts: list[str] = []
            for page in doc:
                page_text = page.get_text("text")
                per_page_chars.append(len(page_text))
                full_text_parts.append(page_text)
    except Exception as exc:  # noqa: BLE001 — toute erreur d'extraction = texte inexploitable
        issues.append(ValidationIssue(raw_id, title, "text_unreadable",
                                      f"extraction du texte impossible : {exc}",
                                      "reported", path))
        return {"extractable": False}

    full_text = "\n".join(full_text_parts)
    chars_no_ws = re.sub(r"\s+", "", full_text)
    total_chars = len(chars_no_ws)
    words = len(full_text.split())
    letters = sum(1 for ch in chars_no_ws if ch.isalpha())
    letter_ratio = letters / total_chars if total_chars else 0.0
    c1_count = sum(1 for ch in chars_no_ws if 0x80 <= ord(ch) <= 0x9F)
    empty_pages = [i for i, n in enumerate(per_page_chars, 1) if n < MIN_EMPTY_PAGE_CHARS]
    avg_chars = total_chars / len(per_page_chars) if per_page_chars else 0.0

    mojibake = bool(MOJIBAKE_RE.search(full_text)) or c1_count >= C1_HEAVY_THRESHOLD
    quality = {
        "extractable": True,
        "chars": total_chars,
        "words": words,
        "letter_ratio": round(letter_ratio, 3),
        "pages_with_text": len(per_page_chars) - len(empty_pages),
        "empty_pages": len(empty_pages),
        "empty_pages_idx": empty_pages[:20],
        "avg_chars_per_page": round(avg_chars, 1),
        "c1_controls": c1_count,
        "mojibake": mojibake,
    }

    if total_chars < MIN_TOTAL_CHARS:
        issues.append(ValidationIssue(
            raw_id, title, "no_text",
            f"presque aucun texte extrait ({total_chars} chars) — PDF scanné ou sans couche texte ?",
            "reported", path))
    elif avg_chars < MIN_AVG_CHARS_PER_PAGE:
        issues.append(ValidationIssue(
            raw_id, title, "too_low_text",
            f"{avg_chars:.0f} chars/page en moyenne — couche texte partielle ?",
            "reported", path))
    if empty_pages:
        issues.append(ValidationIssue(
            raw_id, title, "empty_pages",
            f"{len(empty_pages)} page(s) sans texte (index {empty_pages[:10]})",
            "reported", path))
    if letter_ratio < GARBAGE_LETTER_RATIO_MIN:
        issues.append(ValidationIssue(
            raw_id, title, "garbage_text",
            f"ratio de lettres {letter_ratio:.2f} — texte bruit ?",
            "reported", path))
    if quality["mojibake"]:
        issues.append(ValidationIssue(
            raw_id, title, "mojibake",
            "encodage cassé détecté (mojibake) dans le texte extrait",
            "reported", path))
    return quality


def validate_entry(raw: dict, corpus_root: Path) -> tuple[ManifestEntry, list[ValidationIssue]]:
    """Valide une entrée brute du manifest et la normalise (id, fichier, thème)."""
    issues: list[ValidationIssue] = []

    # --- Identité ---
    raw_id = str(raw.get("arxiv_id") or raw.get("id") or "").strip()
    if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", raw_id):
        issues.append(ValidationIssue(raw_id, normalize_title(raw.get("title")),
                                      "bad_arxiv_id", f"arxiv_id invalide : {raw_id!r}",
                                      "reported"))

    theme = str(raw.get("theme") or "").strip().lower()
    if theme not in THEMES:
        issues.append(ValidationIssue(raw_id, normalize_title(raw.get("title")),
                                      "bad_theme", f"thème invalide : {theme!r}",
                                      "reported"))

    title = normalize_title(raw.get("title"))
    authors = normalize_authors(raw.get("authors"))

    # --- Fichier PDF ---
    raw_path = str(raw.get("local_path") or raw.get("file_path") or "").strip()
    path = resolve_path(raw_path, corpus_root)
    pages = 0
    text_quality: dict = {}
    if path is None:
        issues.append(ValidationIssue(raw_id, title, "missing_file",
                                      f"PDF introuvable : {raw_path!r}", "reported"))
    else:
        if not path.name.startswith(raw_id + "_"):
            issues.append(ValidationIssue(raw_id, title, "name_mismatch",
                                          f"fichier incohérent avec l'id {raw_id}: {path.name}",
                                          "reported", path))
        if path.parent.name != theme:
            issues.append(ValidationIssue(raw_id, title, "folder_mismatch",
                                          f"fichier dans {path.parent.name}/ au lieu de {theme}/",
                                          "reported", path))
        _pdf_ok, pages = inspect_pdf(path, raw_id, title, issues)
        if pages > 0:
            text_quality = inspect_text_quality(path, raw_id, title, issues)

    # --- Pertinence thématique (heuristique sur le titre) ---
    b_theme, b_score = best_theme(title)
    relevance = {
        "own_score": len(theme_relevance(title, theme)),
        "hits": theme_relevance(title, theme),
        "best_theme": b_theme,
        "best_score": b_score,
    }
    if theme in THEMES and relevance["own_score"] == 0:
        issues.append(ValidationIssue(raw_id, title, "off_topic",
                                      f"aucun mot-clé du thème '{theme}' dans le titre",
                                      "reported", path))
    if theme in THEMES and b_theme != theme and 0 < relevance["own_score"] <= b_score:
        issues.append(ValidationIssue(raw_id, title, "ambiguous_theme",
                                      f"le titre évoque surtout le thème '{b_theme}'",
                                      "reported", path))

    # --- Statut ---
    fatal = {"missing_file", "too_small", "bad_signature", "no_eof", "corrupt_pdf",
                 "no_text", "text_unreadable"}
    if any(i.kind in fatal for i in issues):
        status = "invalid"
    elif any(i.kind == "off_topic" for i in issues):
        status = "flagged"
    else:
        status = "valid"

    file_path = path.relative_to(corpus_root).as_posix() if path else raw_path.replace("\\", "/")
    entry = ManifestEntry(
        id=raw_id,
        theme=theme,
        title=title,
        authors=authors,
        summary=str(raw.get("summary") or "").strip(),
        pdf_url=str(raw.get("pdf_url") or ""),
        pages=pages,
        file_path=file_path,
        status=status,
        relevance=relevance,
        text_quality=text_quality,
        path=path,
    )
    return entry, issues


def dedupe_entries(entries: list[ManifestEntry]) -> tuple[list[ManifestEntry], list[ValidationIssue]]:
    """Supprime les doublons sur la base arXiv (même papier, versions différentes).

    La version la plus élevée (v1 < v2 < v3) est conservée.
    """
    groups: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        groups.setdefault(base_arxiv_id(entry.id), []).append(entry)

    kept: list[ManifestEntry] = []
    issues: list[ValidationIssue] = []
    for _, group in groups.items():
        if len(group) == 1:
            kept.extend(group)
            continue
        best = max(group, key=lambda e: version_sort_key(e.id))
        for entry in group:
            if entry is not best:
                entry.status = "duplicate"
                issues.append(ValidationIssue(
                    entry.id, entry.title, "duplicate",
                    f"même papier que {best.id} (version retenue) : {best.file_path}",
                    "reported", entry.path,
                ))
        kept.append(best)
    return kept, issues


def scan_orphans(corpus_root: Path, entries: list[ManifestEntry]) -> list[ValidationIssue]:
    """Fichiers PDF présents sur disque mais absents du manifest (orchestre des J1)."""
    referenced = {e.path.resolve() for e in entries if e.path is not None}
    issues: list[ValidationIssue] = []
    for pdf in corpus_root.rglob("*.pdf"):
        if "_quarantine" in pdf.parts:
            continue
        if pdf.resolve() not in referenced:
            issues.append(ValidationIssue("?", pdf.name, "orphan",
                                          "PDF présent sur disque mais absent du manifest",
                                          "reported", pdf))
    return issues


def move_files(corpus_root: Path, paths: list[Path]) -> list[tuple[Path, Path]]:
    """Déplace les fichiers vers corpus/_quarantine/ (nom unique en cas de collision)."""
    quarantine_dir = corpus_root / "_quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for src in paths:
        if src in seen or not src.exists():
            continue
        seen.add(src)
        dest = quarantine_dir / src.name
        counter = 1
        while dest.exists():
            dest = quarantine_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        shutil.move(str(src), str(dest))
        moved.append((src.resolve(), dest.resolve()))
    return sorted(moved, key=lambda pair: pair[0].name)


def write_manifest(corpus_root: Path, entries: list[ManifestEntry]) -> None:
    """Écrit le manifest propre (artefact canonique utilisé par J3+)."""
    entries = sorted(entries, key=lambda e: (e.theme, e.id))
    payload = [
        {
            "id": e.id,
            "theme": e.theme,
            "title": e.title,
            "authors": e.authors,
            "summary": e.summary,
            "num_pages": e.pages,
            "pdf_url": e.pdf_url,
            "file_path": e.file_path,      # portable : "rag/xxx.pdf"
            "relevance": e.relevance,
            "text_quality": e.text_quality,
            "status": e.status,
        }
        for e in entries
    ]
    dest = corpus_root / "manifest.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_issue(issue: ValidationIssue) -> dict:
    return {
        "entry_id": issue.entry_id,
        "title": issue.title,
        "kind": issue.kind,
        "message": issue.message,
        "action": issue.action,
        "file": str(issue.path) if issue.path else None,
    }


def write_report(corpus_root: Path, report: dict) -> None:
    """Écrit validation_report.json (machine) et validation_report.md (lisible)."""
    json_path = corpus_root / "validation_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Rapport de validation du corpus (JOUR 2)",
        "",
        f"- Généré le : {report['generated_at']}",
        f"- Corpus : `{report['corpus_root']}`",
        "",
        "## Indicateurs",
        "",
        "| Indicateur | Valeur |",
        "| --- | --- |",
        f"| Entrées dans le manifest (J1) | {report['stats']['manifest_entries']} |",
        f"| PDFs sur disque | {report['stats']['pdfs_on_disk']} |",
        f"| Documents publiés (manifest propre) | {report['stats']['kept']} |",
        f"| Pages totales (documents publiés) | {report['stats']['total_pages']} |",
        f"| Invalides (corrompus / manquants) | {report['stats']['invalid']} |",
        f"| Doublons de contenu | {report['stats']['duplicates']} |",
        f"| Retraits manuels (--drop) | {report['stats']['dropped']} |",
        f"| Hors-sujet (signalés, à vérifier) | {report['stats']['off_topic']} |",
        f"| Fichiers orphelins | {report['stats']['orphans']} |",
        f"| Qualité texte (no_text / mojibake / bruit) | {report['stats']['text_quality']} |",
        "",
        "| Thème | Entrées J1 | Publiés |",
        "| --- | --- | --- |",
    ]
    for theme in sorted(THEMES):
        info = report["per_theme"][theme]
        lines.append(
            f"| {theme} | {info['total']} | {info['kept']} |"
        )
    lines.append("")

    if report["issues"]:
        lines += ["## Problèmes détectés", ""]
        for i, issue in enumerate(report["issues"], 1):
            lines.append(
                f"{i}. `[{issue['entry_id']}]` **{issue['kind']}** — {issue['message']} "
                f"_(action : {issue['action']})_"
            )
        lines.append("")

    if report["quarantined"]:
        lines += ["## Fichiers mis en quarantaine", ""]
        for pair in report["quarantined"]:
            lines.append(f"- `{pair['from']}` → `{pair['to']}`")
        lines.append("")

    (corpus_root / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

INVALID_KINDS = {"missing_file", "too_small", "bad_signature", "no_eof", "corrupt_pdf",
                 "no_text", "text_unreadable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation et nettoyage du corpus RAG (JOUR 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python validate_corpus.py                    # rapport seul (dry-run)\n"
            "  python validate_corpus.py --quarantine       # + déplace fichiers invalides/doublons\n"
            "  python validate_corpus.py --quarantine --quarantine-offtopic  # + hors-sujets\n"
        ),
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="déplacer les fichiers invalides et doublons vers corpus/_quarantine/",
    )
    parser.add_argument(
        "--quarantine-offtopic",
        action="store_true",
        help="déplacer aussi les papiers détectés hors-sujet (requiert --quarantine)",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        metavar="ARXIV_ID",
        help="retirer manuellement ces papers (ex. 1411.4510v1 2504.14689v1). "
             "Quarantaine forcée, même sans --quarantine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quarantine_offtopic and not args.quarantine:
        raise SystemExit("--quarantine-offtopic requiert --quarantine")

    corpus_root, manifest_path = locate_corpus()
    with manifest_path.open(encoding="utf-8") as handle:
        raw_entries = json.load(handle)

    # --- 1. Validation individuelle de chaque entrée ---
    entries: list[ManifestEntry] = []
    all_issues: list[ValidationIssue] = []
    for raw in raw_entries:
        entry, issues = validate_entry(raw, corpus_root)
        entries.append(entry)
        all_issues.extend(issues)

    # --- 2. Doublons de contenu (même papier, versions différentes) ---
    entries, dup_issues = dedupe_entries(entries)
    all_issues.extend(dup_issues)

    # --- 3. Fichiers orphelins (PDF non déclarés dans le manifest) ---
    orphan_issues = scan_orphans(corpus_root, entries)
    all_issues.extend(orphan_issues)

    # --- 4. Décision de quarantaine + retraits manuels (--drop) ---
    quarantine_targets: list[Path] = []
    if args.quarantine:
        for issue in all_issues:
            if issue.kind in INVALID_KINDS and issue.path is not None:
                quarantine_targets.append(issue.path)
            if issue.kind in {"duplicate", "orphan"} and issue.path is not None:
                quarantine_targets.append(issue.path)
            if args.quarantine_offtopic and issue.kind == "off_topic" and issue.path is not None:
                quarantine_targets.append(issue.path)

    if args.drop:
        drop_keys = {base_arxiv_id(x) for x in args.drop}
        known_bases = {base_arxiv_id(e.id) for e in entries}
        unknown = [x for x in args.drop if base_arxiv_id(x) not in known_bases]
        if unknown:
            print(f"  AVERTISSEMENT --drop inconnus (ignorés) : {', '.join(unknown)}")
        for entry in entries:
            if base_arxiv_id(entry.id) in drop_keys and entry.status not in {"invalid", "duplicate"}:
                entry.status = "dropped"
                all_issues.append(ValidationIssue(
                    entry.id, entry.title, "manual_drop",
                    "retiré manuellement via --drop", "quarantined", entry.path))
                if entry.path is not None:
                    quarantine_targets.append(entry.path)

    quarantined: list[tuple[Path, Path]] = []
    moved_src: set[Path] = set()
    if args.quarantine or args.drop:
        quarantined = move_files(corpus_root, quarantine_targets)
        moved_src = {src for src, _ in quarantined}
        for issue in all_issues:
            if issue.path in moved_src:
                issue.action = "quarantined"

    # --- 5. Manifest propre (valid + flagged ; on exclut ce qui a été déplacé) ---
    clean_entries = [
        e for e in entries
        if e.status in {"valid", "flagged"}
        and (e.path is None or e.path.resolve() not in moved_src)
    ]
    write_manifest(corpus_root, clean_entries)

    # --- 6. Statistiques + rapport ---
    counts = Counter(e.status for e in entries)
    off_topic = sum(1 for i in all_issues if i.kind == "off_topic")
    total_pages = sum(e.pages for e in clean_entries)
    pdfs_on_disk = [p for p in corpus_root.rglob("*.pdf") if "_quarantine" not in p.parts]
    text_issue_kinds = {"no_text", "text_unreadable", "too_low_text", "empty_pages",
                        "garbage_text", "mojibake"}
    text_issues = sum(1 for i in all_issues if i.kind in text_issue_kinds)

    per_theme = {}
    for theme in sorted(THEMES):
        theme_all = [e for e in entries if e.theme == theme]
        theme_kept = [e for e in clean_entries if e.theme == theme]
        per_theme[theme] = {
            "total": len(theme_all),
            "kept": len(theme_kept),
            "pages": sum(e.pages for e in theme_kept),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_root": str(corpus_root),
        "stats": {
            "manifest_entries": len(raw_entries),
            "pdfs_on_disk": len(pdfs_on_disk),
            "kept": len(clean_entries),
            "total_pages": total_pages,
            "invalid": counts.get("invalid", 0),
            "duplicates": counts.get("duplicate", 0),
            "dropped": counts.get("dropped", 0),
            "off_topic": off_topic,
            "orphans": len(orphan_issues),
            "text_quality": text_issues,
            "quarantined": len(quarantined),
        },
        "per_theme": per_theme,
        "issues": [format_issue(i) for i in all_issues],
        "quarantined": [{"from": str(src), "to": str(dst)} for src, dst in quarantined],
    }
    write_report(corpus_root, report)

    # --- 7. Résumé console ---
    print("[JOUR 2] Validation du corpus terminée")
    print(f"  · Manifest lu         : {len(raw_entries)} entrées")
    print(f"  · Documents publiés   : {len(clean_entries)} ({total_pages} pages)")
    print(f"  · Invalides/corrompus : {counts.get('invalid', 0)}")
    print(f"  · Doublons retirés    : {counts.get('duplicate', 0)}")
    print(f"  · Hors-sujet signalés : {off_topic}")
    print(f"  · Orphelins           : {len(orphan_issues)}")
    print(f"  · Texte anormal       : {text_issues} (pages vides / mojibake / bruit)")
    print(f"  · Retraits manuels    : {counts.get('dropped', 0)}")
    if quarantined:
        print(f"  · Quarantaine         : {len(quarantined)} fichier(s) -> corpus/_quarantine/")
    print("=> manifest.json, validation_report.json, validation_report.md régénérés")


if __name__ == "__main__":
    main()