"""Localise le passage fautif dans le dump d'audit."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
t = open("files/audit_full_trace_output.txt", encoding="utf-8").read()

MARKER = "real-world images as additional"
pos = t.find(MARKER)
print("pos =", pos)

src_pat = re.compile(r"----- SOURCE \[(\d+)\] doc=(\S+) p\.(\S+) -----")
srcs = [(m.start(), m.group(1), m.group(2), m.group(3)) for m in src_pat.finditer(t)]
print("sources trouvées:", [(s[1], s[2], s[3]) for s in srcs])

if pos >= 0:
    before = [s for s in srcs if s[0] < pos]
    last = before[-1] if before else None
    print("\n>>> PASSAGE FAUTIF DANS:", last)
    start = max(0, pos - 600)
    print("\n--- contexte extrait ---")
    print(t[start:pos + 300])
