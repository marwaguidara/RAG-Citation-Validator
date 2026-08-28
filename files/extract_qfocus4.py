import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1] if len(sys.argv) > 1 else "files/audit_qfocus_AFTER2.txt"
try:
    t = open(path, encoding="utf-8").read()
except UnicodeDecodeError:
    t = open(path, encoding="utf-16", errors="replace").read()

# Decoupe sur les entetes "# QUESTION N : ..." (fichier LF-only)
pos = [m.start() for m in re.finditer(r"^#*\s*QUESTION \d :", t, re.M)] + [len(t)]
print(f"[{len(pos)-1} questions trouvees]")
for k in range(len(pos) - 1):
    body = t[pos[k]:pos[k + 1]]
    print(f"\n{'='*70}")
    print(body.splitlines()[0])
    print('='*70)
    for label in ("3) REPONSE BRUTE", "4) CLAIMS EXTRAITS", "5) REPONSE FINALE"):
        m = re.search(rf"^-+ {re.escape(label)}.*?(?=^-+ |\Z)", body, re.M | re.S)
        if m:
            block = m.group(0)
            print(block[:1500] + (" ...[tronque]" if len(block) > 1500 else ""))
