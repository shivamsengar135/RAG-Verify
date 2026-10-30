import json, sys
sys.path.insert(0, "code")
from common import read_jsonl
for m in ["rag_llama", "rag", "ragverify"]:
    old = {r["id"]: r for r in read_jsonl(f"results/raw/hotpotqa/{m}/seed0.jsonl")}
    new = read_jsonl(f"results/raw/hotpotqa/{m}/seed0_probe_ollama04012.jsonl")
    txt = sum(old[r["id"]]["output_text"] == r["output_text"] for r in new)
    ans = sum((old[r["id"]]["prediction"], old[r["id"]]["abstained"]) == (r["prediction"], r["abstained"]) for r in new)
    print(f"{m:10s} n={len(new)} identical_text={txt} identical_final_answer={ans}")
