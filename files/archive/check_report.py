"""Vérifie l'intégrité structurelle du rapport PFE."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1] if len(sys.argv) > 1 else "RAPPORT_PFE_RAG_Citation_Validator.md"
t = open(path, encoding="utf-8").read()
lines = t.splitlines()

secs = [l for l in lines if re.match(r"^## \d+\.", l)]
print("LIGNES TOTALES:", len(lines))
print("CARACTERES:", len(t))
print("SECTIONS TROUVEES:", len(secs))
for s in secs:
    print(" ", s)

# Vérifie que les 31 sections attendues sont présentes
attendu = []
for i in range(1, 30):
    attendu.append(f"## {i}.")
for i in (30, 31):
    attendu.append(f"## {i}. ")
manquantes = [a for a in attendu if not any(s.startswith(a) for s in secs)]
print("SECTIONS MANQUANTES:", manquantes if manquantes else "aucune")

# Vérifie l'absence de marqueurs de continuation de template
for artefact in ("{{", "}}", "TODO", "placeholder", "<<<"):
    if artefact in t:
        print("ARTEFACT TROUVE:", artefact)