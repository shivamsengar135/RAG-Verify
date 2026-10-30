"""Environment check: GPU, LLMs via Ollama, NLI verifier and embedder co-resident in VRAM.

Usage: .venv\\Scripts\\python.exe code\\env_check.py
Writes logs/env_check.json.
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import ollama
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LLMS = ["qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M"]
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMB_MODEL = "BAAI/bge-small-en-v1.5"


def gpu_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


def main():
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "vram_idle_mib": gpu_used_mib(),
    }

    # LLMs: one greedy generation each, timed by Ollama's own counters.
    report["llms"] = {}
    for m in LLMS:
        r = ollama.generate(
            model=m,
            prompt="Answer in one short sentence: what is the capital of Australia?",
            options={"temperature": 0, "seed": 42},
            keep_alive="10m",
        )
        report["llms"][m] = {
            "answer": r["response"].strip(),
            "gen_tok_per_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1),
        }
    # Keep the main backbone resident so the VRAM reading below is the real combined peak.
    ollama.generate(model=LLMS[0], prompt="ok", options={"num_predict": 1}, keep_alive="10m")
    report["vram_with_llm_mib"] = gpu_used_mib()

    # NLI verifier.
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, dtype=torch.float16).cuda().eval()
    premise = "Pride and Prejudice is an 1813 novel by the English author Jane Austen."
    hyps = ["Jane Austen wrote Pride and Prejudice.", "Charles Dickens wrote Pride and Prejudice.",
            "Jane Austen was born in Hampshire."]
    enc = tok([premise] * len(hyps), hyps, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        probs = torch.softmax(nli(**enc).logits.float(), dim=-1).cpu()
    labels = [nli.config.id2label[i] for i in range(probs.shape[1])]
    report["nli"] = {
        "model": NLI_MODEL,
        "labels": labels,
        "predictions": [
            {"hypothesis": h, "pred": labels[int(p.argmax())], "probs": [round(float(x), 3) for x in p]}
            for h, p in zip(hyps, probs)
        ],
    }

    # Embedder.
    emb = SentenceTransformer(EMB_MODEL, device="cuda")
    t0 = time.perf_counter()
    vecs = emb.encode([premise] * 256, batch_size=64, normalize_embeddings=True)
    report["embedder"] = {
        "model": EMB_MODEL,
        "dim": int(vecs.shape[1]),
        "passages_per_s": round(256 / (time.perf_counter() - t0), 1),
    }

    report["vram_peak_all_loaded_mib"] = gpu_used_mib()
    report["vram_total_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20

    out = ROOT / "logs" / "env_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
