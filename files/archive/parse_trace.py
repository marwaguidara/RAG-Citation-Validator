"""Extract the exact raw Ollama output from the trace file."""
import sys, re, json
sys.stdout.reconfigure(encoding="utf-8")

with open(r"files\audit_stderr.txt", "rb") as f:
    raw_bytes = f.read()

text = raw_bytes.decode("utf-8", errors="replace")
text = re.sub(r'\x00', '', text)

# Find the STEP2 RAW_LLM section
lines = text.split("\n")
in_raw = False
raw_lines = []
for line in lines:
    if "STEP2 RAW_LLM" in line:
        in_raw = True
        continue
    if in_raw:
        if "STEP3" in line:
            in_raw = False
            break
        raw_lines.append(line)

raw_output = "\n".join(raw_lines).strip()
# Remove >>> markers
raw_output = raw_output.replace(">>>", "").replace("<<<", "").strip()

print("=== EXTRACTED RAW OLLAMA OUTPUT ===")
print(repr(raw_output))
print()
print(f"Length: {len(raw_output)} chars")
print()

# Try to parse it as JSON
candidates_str = raw_output.find("{\"claims\"")
if candidates_str >= 0:
    json_start = candidates_str
    json_end = raw_output.rfind("}")
    if json_end > json_start:
        json_str = raw_output[json_start:json_end+1]
        print(f"JSON substring: {repr(json_str[:200])}...")
        try:
            parsed = json.loads(json_str)
            print(f"\nParsed JSON: {parsed}")
            print(f"Claims count: {len(parsed.get('claims', []))}")
            for c in parsed.get("claims", []):
                print(f"  text={c.get('text','')[:80]} cits={c.get('citations')}")
        except json.JSONDecodeError as e:
            print(f"\nJSON PARSE ERROR: {e}")
            print("This is why parse_claim_json returns []!")
            # Show the problematic citations
            claims_match = re.findall(r'"citations":\[([^\]]+)\]', json_str)
            print(f"Citation arrays found: {claims_match}")
            for arr in claims_match:
                parts = arr.split(",")
                print(f"  Parts: {parts}")
                for p in parts:
                    p = p.strip()
                    try:
                        int(p)
                        print(f"    {p} -> valid int")
                    except ValueError:
                        print(f"    {p} -> INVALID (not an int, breaks JSON!)")

