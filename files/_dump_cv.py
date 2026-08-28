# -*- coding: utf-8 -*-
"""Outil temporaire d'inspection : dump de sections de citation_verifier.py."""
import sys

PATH = r"c:/marwaguidara/RAG-Citation-Validator/files/citation_verifier.py"

def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    lines = open(PATH, encoding="utf-8").read().splitlines()
    print("TOTAL", len(lines))
    for i, line in enumerate(lines, start=1):
        if start <= i <= end:
            print(f"{i:4d}| {line}")

if __name__ == "__main__":
    main()
