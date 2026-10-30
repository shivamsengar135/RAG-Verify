"""Shared helpers: config, paths, JSONL I/O, seeding, Ollama client with cost accounting."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_config(*extra):
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    for path in extra:
        if path:
            deep_update(cfg, yaml.safe_load(Path(path).read_text()))
    return cfg


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # last line cut off by a crash / power loss: drop it so the example is simply re-run
            path.with_suffix(".truncated").write_text(line, encoding="utf-8")
            path.write_text("".join(lines[:-1]), encoding="utf-8")
    return rows


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def text_hash(*parts):
    return hashlib.sha1("␟".join(parts).encode("utf-8")).hexdigest()[:16]


class CostMeter:
    """Accumulates LLM calls / tokens for one example."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.gen_tokens = 0
        self.llm_seconds = 0.0
        self.prompt_eval_s = 0.0  # Ollama-reported time spent reading the prompt
        self.gen_s = 0.0          # Ollama-reported time spent generating tokens

    def as_dict(self):
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "llm_seconds": round(self.llm_seconds, 3),
            "prompt_eval_s": round(self.prompt_eval_s, 3),
            "gen_s": round(self.gen_s, 3),
        }


class LLM:
    """Thin wrapper over the Ollama chat API with deterministic options and cost tracking."""

    def __init__(self, model, num_ctx=8192, max_new_tokens=256, temperature=0.0, seed=0):
        import ollama

        self.client = ollama.Client()
        self.model = model
        self.base_options = {
            "num_ctx": num_ctx,
            "num_predict": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
        }

    def chat(self, user, system=None, meter=None, **overrides):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        options = {**self.base_options, **overrides}
        t0 = time.perf_counter()
        r = self.client.chat(model=self.model, messages=messages, options=options, keep_alive="30m")
        dt = time.perf_counter() - t0
        if meter is not None:
            meter.calls += 1
            meter.prompt_tokens += r.get("prompt_eval_count") or 0
            meter.gen_tokens += r.get("eval_count") or 0
            meter.llm_seconds += dt
            meter.prompt_eval_s += (r.get("prompt_eval_duration") or 0) / 1e9
            meter.gen_s += (r.get("eval_duration") or 0) / 1e9
        return r["message"]["content"].strip()

    def warmup(self):
        """Untimed call so model loading is never charged to the first example."""
        self.chat("Reply with OK.", num_predict=2)

    def unload(self):
        self.client.generate(model=self.model, prompt="", keep_alive=0)


def gpu_temp_c():
    import subprocess

    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                      text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:  # never let monitoring break a run
        return None


def thermal_pause(log=None, hot=83, cool=72, max_wait_s=900):
    """Called between examples (outside the timed region): if the GPU is at/above `hot` degC, idle until it
    drops to `cool` degC, so long laptop runs do not sit at their thermal limit. Returns seconds paused."""
    t = gpu_temp_c()
    if t is None or t < hot:
        return 0.0
    t0 = time.perf_counter()
    while t is not None and t > cool and time.perf_counter() - t0 < max_wait_s:
        time.sleep(5)
        t = gpu_temp_c()
    waited = time.perf_counter() - t0
    if log is not None:
        log.info("thermal pause: %.0fs (GPU now %s C)", waited, t)
    return waited


def unload_all_ollama():
    import ollama

    client = ollama.Client()
    for m in client.ps().get("models", []):
        client.generate(model=m["model"], prompt="", keep_alive=0)
