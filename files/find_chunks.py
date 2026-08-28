"""Find the exact RAG chunks for proper testing."""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

with open(r"files\corpus\chunks.json", encoding="utf-8") as f:
    data = json.load(f)
chunks = data["chunks"]

# Find chunks that actually contain "RAG" or "Retrieval-Augmented" as the concept
print("=== CHUNKS WITH 'RAG' (not just subwords) ===")
found = []
for i, c in enumerate(chunks):
    text = c.get("text", "")
    if "RAG " in text or "RAG." in text or "RAG," in text or "(RAG)" in text or "RAG)" in text:
        found.append((i, c))

for i, c in found[:10]:
    print(f"[{i}] chunk_id={c.get('chunk_id')[:16]} theme={c.get('theme')}")
    print(f"   doc={c.get('document_id')[:50]}")
    print(f"   text[:500]={repr(c['text'][:500])}")
    print()

# Also find chunks with "retrieval-augmented" as a compound word
print("=== CHUNKS WITH 'retrieval-augmented' (compound) ===")
aug = []
for i, c in enumerate(chunks):
    text = c.get("text", "").lower()
    if "retrieval-augmented" in text:
        aug.append((i, c))
        if len(aug) >= 10:
            break

for i, c in aug:
    print(f"[{i}] chunk_id={c.get('chunk_id')[:16]} theme={c.get('theme')}")
    print(f"   doc={c.get('document_id')[:50]}")
    print(f"   text[:500]={repr(c['text'][:500])}")
    print()

print(f"\nTotal RAG-context chunks: {len(found)}")
print(f"Total retrieval-augmented chunks: {len(aug)}")


