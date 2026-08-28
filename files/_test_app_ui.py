"""Test UI AppTest : lance app.main en mode headless et vérifie le rendu."""
import sys

sys.path.insert(0, r"c:/marwaguidara/RAG-Citation-Validator")
import os
os.chdir(r"c:/marwaguidara/RAG-Citation-Validator")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(r"c:/marwaguidara/RAG-Citation-Validator/app.py")
at.run(timeout=60)
print("EXCEPTIONS:", [str(e.value) for e in at.exception])
print("TITLES:", [t.value for t in at.title])
print("CAPTIONS:", [c.value[:60] for c in at.caption])
print("TEXT_INPUTS:", [ti.label for ti in at.text_input])
print("BUTTONS:", [b.label for b in at.button])
print("RADIOS:", [r.label for r in at.radio])
print("SLIDERS:", [s.label for s in at.slider])
print("=== UI OK ===")