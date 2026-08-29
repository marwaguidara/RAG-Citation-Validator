"""Extraction ciblée : sections RAW LLM / CLAIMS / GATE / REPONSE FINALE (+contenu)."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1] if len(sys.argv) > 1 else "files/audit_qfocus_AFTER.txt"
raw = open(path, "rb").read()
t = raw.decode("utf-16") if b"\x00" in raw[:200] else raw.decode("utf-8", errors="replace")
lines = [l.strip() for l in t.splitlines()]

LABEL = re.compile(
    r"^(# QUESTION|--- \d\) |"
    r".*RAW LLM.*|"
    r".*CLAIMS APRES PARSING.*|"
    r".*CLAIMS FINAUX.*|"
    r".*GATE.*|"
    r".*REPONSE FINALE.*|"
    r".*grounding.*|"
    r".*cov.*OK|.*REJECT.*|"
    r".*refus.*|"
    r".*support=.*)$",
    re.IGNORECASE,
)
i = 0
while i < len(lines):
    s = lines[i]
    if s and LABEL.match(s):
        print(s[:500])
        # pour un label, imprimer aussi la/les lignes de contenu qui suivent
        if s.startswith("---") or "RAW LLM" in s or "FINALE" in s:
            j = i + 1
            shown = 0
            while j < len(lines) and shown < 4:
                nxt = lines[j]
                if nxt and not LABEL.match(nxt):
                    print("   >>", nxt[:500])
                    shown += 1
                    j += 1
                else:
                    break
    i += 1
