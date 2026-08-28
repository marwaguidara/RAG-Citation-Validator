"""Dump lisible d'un run audit_qfocus (UTF-16, redirection PS)."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1] if len(sys.argv) > 1 else "files/audit_qfocus_AFTER.txt"
max_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 120
raw = open(path, "rb").read()
if b"\x00" in raw[:200]:
    t = raw.decode("utf-16")
else:
    t = raw.decode("utf-8", errors="replace")
lines = [l.rstrip() for l in t.splitlines()]
printed = 0
for i, line in enumerate(lines):
    s = line.strip()
    if not s:
        continue
    print(f"{i:4d} | {s[:180]}")
    printed += 1
    if printed >= max_lines:
        print(f"... ({len(lines)} lignes au total)")
        break
