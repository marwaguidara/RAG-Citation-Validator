"""Lecture seule : extrait generate_claim_answer et run_verification + verify_citations."""
import re

for path, names in [
    (r"c:/marwaguidara/RAG-Citation-Validator/files/generate_evaluation_artifacts.py",
     ["generate_claim_answer", "run_verification", "_build_config_metrics"]),
    (r"c:/marwaguidara/RAG-Citation-Validator/files/citation_verifier.py",
     None),  # toutes les def de top niveau
]:
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    print("#" * 70)
    print("FILE:", path)
    if names is None:
        # lister les fonctions avec leur ligne
        for i, l in enumerate(lines, 1):
            if re.match(r"^(def|class) ", l):
                print(f"  L{i}: {l.strip()[:100]}")
        continue
    for name in names:
        m = re.search(rf"^def {name}\(", src, re.M)
        if not m:
            print(f"-- {name} NOT FOUND")
            continue
        start = m.start()
        nxt = re.search(r"^def \w+\(", src[m.end():], re.M)
        end = m.end() + nxt.start() if nxt else len(src)
        chunk = src[start:end]
        line_no = src[:start].count("\n") + 1
        print("=" * 30, f"{name} (L{line_no}, {len(chunk)} chars)", "=" * 30)
        print(chunk[:5500])
