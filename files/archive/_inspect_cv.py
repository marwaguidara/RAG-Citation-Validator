"""Extract lines from citation_verifier.py for inspection."""
import io
import sys

PATH = r"c:/marwaguidara/RAG-Citation-Validator/files/citation_verifier.py"
lines = open(PATH, encoding="utf-8").read().splitlines()

ranges = [(186, 246), (300, 416)]
for start, end in ranges:
    print(f"===== LINES {start}-{end} =====")
    for i in range(start, end):
        print(f"{i}: {lines[i]}")
sys.stdout.flush()