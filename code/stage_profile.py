"""Reviewer 1 (camera-ready): processing time and GPU memory of each RAG-Verify stage.

Same protocol as the latency benchmark (code/run_extras.py): first 50 dev questions per dataset, frozen configuration
(hybrid verifier, tau 0.3, rho 0), Qwen2.5-7B, nothing else on the GPU. The pipeline is not changed; its own
functions are wrapped with timers:
  retrieval      Retriever.search
  answer         first top-level LLM call (the draft)
  decomposition  methods.decompose (one LLM call per round)
  answer check   LLMJudgeVerifier.score_answer (question-aware check of the answer claim, one call per round)
  nli            NLIVerifier.score (supporting claims)
  revision       second top-level LLM call (only when a claim was flagged)
A call made inside another stage (the LLM call of decomposition or of the answer check) counts in that stage.
GPU memory: device memory before anything is loaded, Ollama's resident size of the generator (api/ps), torch
allocations of the retriever and NLI models, the torch peak inside each stage, and device-wide use sampled every
50 ms (torch.cuda.mem_get_info) and attributed to the stage running at that moment.

Usage: .venv\\Scripts\\python.exe code\\stage_profile.py
Output: results/profile/stages_<dataset>.jsonl, results/profile/summary.json
"""
import collections
import contextlib
import subprocess
import threading
import time

import httpx
import numpy as np
import torch

import methods
from common import (LLM, ROOT, CostMeter, append_jsonl, load_config, read_jsonl, set_seed, thermal_pause,
                    unload_all_ollama, write_json)
from run import load_examples, split_seed

OUT = ROOT / "results" / "profile"
MIB = 2 ** 20
STAGES = ["retrieval", "answer", "decomposition", "answer_check", "nli", "revision"]


def smi(query):
    out = subprocess.check_output(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], text=True,
                                  timeout=20)
    return [l.strip() for l in out.splitlines() if l.strip()]


class Profiler:
    def __init__(self, period=0.05):
        self.active, self.dev_peak = None, collections.defaultdict(int)
        self.reset()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, args=(period,), daemon=True)
        self._thread.start()

    def reset(self):
        self.t, self.n, self.torch_peak = collections.defaultdict(float), collections.Counter(), {}

    def _sample(self, period):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            key = self.active or "between stages"
            self.dev_peak[key] = max(self.dev_peak[key], (total - free) // MIB)
            time.sleep(period)

    def stop(self):
        self._stop.set()
        self._thread.join()

    @contextlib.contextmanager
    def stage(self, name):
        if self.active is not None:  # nested call: counted in the outer stage
            yield
            return
        self.active = name
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.t[name] += time.perf_counter() - t0
            self.n[name] += 1
            self.torch_peak[name] = max(self.torch_peak.get(name, 0), torch.cuda.max_memory_allocated() // MIB)
            self.active = None


def staged(prof, name, fn):
    def wrapper(*a, **k):
        with prof.stage(name):
            return fn(*a, **k)
    return wrapper


def main():
    unload_all_ollama()
    time.sleep(5)
    others = smi("compute-apps=pid,process_name")
    if others:
        raise SystemExit(f"GPU is in use by {others}; stop other GPU jobs first so the measurement is clean")
    cfg = load_config()
    seed = split_seed("dev")
    set_seed(seed)
    mem = {"device_total_mib": int(smi("gpu=memory.total")[0]), "before_anything_mib": int(smi("gpu=memory.used")[0])}
    torch.zeros(1, device="cuda")
    mem["after_cuda_context_mib"] = int(smi("gpu=memory.used")[0])

    llm = LLM(cfg["llm"]["backbone"], cfg["llm"]["num_ctx"], cfg["llm"]["max_new_tokens"], cfg["llm"]["temperature"],
              seed=seed)
    llm.warmup()
    ps = next(m for m in llm.client.ps()["models"] if m["model"] == cfg["llm"]["backbone"])
    mem.update(ollama_size_vram_mib=ps["size_vram"] // MIB, ollama_size_mib=ps["size"] // MIB,
               after_llm_mib=int(smi("gpu=memory.used")[0]))

    from retriever import Retriever
    from verifiers import LLMJudgeVerifier, NLIVerifier

    a0 = torch.cuda.memory_allocated()
    nli = NLIVerifier(cfg)
    nli.score(["warm-up claim"], ["warm-up premise"])
    mem.update(nli_weights_mib=(torch.cuda.memory_allocated() - a0) // MIB, after_nli_mib=int(smi("gpu=memory.used")[0]))
    judge = LLMJudgeVerifier(llm)
    info = {"ollama_version": httpx.get("http://localhost:11434/api/version").json()["version"],
            "gpu": smi("gpu=name")[0], "backbone": cfg["llm"]["backbone"], "split": "dev", "n_per_dataset": 50}

    prof = Profiler()
    llm_chat, decompose = llm.chat, methods.decompose

    def chat(*a, **k):
        if prof.active is None:  # top-level call inside ragverify(): the draft, then (if flagged) the revision
            with prof.stage("answer" if prof.n["answer"] == 0 else "revision"):
                return llm_chat(*a, **k)
        return llm_chat(*a, **k)

    llm.chat = chat
    methods.decompose = staged(prof, "decomposition", decompose)
    nli.score = staged(prof, "nli", nli.score)
    judge.score_answer = staged(prof, "answer_check", judge.score_answer)
    flags = {"verifier": cfg["verifier"]["kind"]}
    summary = {"info": info, "memory": mem, "datasets": {}}
    OUT.mkdir(parents=True, exist_ok=True)

    for ds in ["hotpotqa", "triviaqa"]:
        a0 = torch.cuda.memory_allocated()
        retriever = Retriever(ds, cfg)
        retriever.search("warm-up query", 1)
        mem[f"retriever_weights_mib_{ds}"] = (torch.cuda.memory_allocated() - a0) // MIB
        retriever.search = staged(prof, "retrieval", retriever.search)
        from types import SimpleNamespace

        ctx = SimpleNamespace(cfg=cfg, seed=seed, meter=None, retriever=retriever, nli=nli, judge=judge, llm=llm)
        saved = {r["id"]: r["output_text"] for r in read_jsonl(ROOT / "results/raw" / ds / "ragverify/dev_latency.jsonl")}
        path = OUT / f"stages_{ds}.jsonl"
        path.unlink(missing_ok=True)
        rows = []
        for i, ex in enumerate(load_examples(ds, "dev", 50), 1):
            thermal_pause()  # outside the timed region
            prof.reset()
            ctx.meter = CostMeter()
            t0 = time.perf_counter()
            res = methods.ragverify(ex, ctx, flags)
            total = time.perf_counter() - t0
            row = {"id": ex["id"], "total_s": round(total, 4), "stage_s": {k: round(v, 4) for k, v in prof.t.items()},
                   "stage_calls": dict(prof.n), "torch_peak_mib": dict(prof.torch_peak), "llm_calls": ctx.meter.calls,
                   "same_output_as_latency_run": res["output_text"] == saved.get(ex["id"])}
            append_jsonl(path, row)
            rows.append(row)
            if i % 10 == 0:
                print(f"  {ds} {i}/50 last={total:.1f}s", flush=True)
        tot = np.array([r["total_s"] for r in rows])
        st = {}
        for s in STAGES:
            v = np.array([r["stage_s"].get(s, 0.0) for r in rows])
            ran = v[v > 0]
            st[s] = {"mean_s_per_question": float(v.mean()), "share_of_mean_total": float(v.mean() / tot.mean()),
                     "questions_with_stage": int(len(ran)), "median_s_when_run": float(np.median(ran)) if len(ran) else None,
                     "calls_per_question": float(np.mean([r["stage_calls"].get(s, 0) for r in rows])),
                     "torch_peak_mib": max(r["torch_peak_mib"].get(s, 0) for r in rows)}
        other = tot - sum(np.array([r["stage_s"].get(s, 0.0) for r in rows]) for s in STAGES)
        summary["datasets"][ds] = {"n": len(rows), "total_median_s": float(np.median(tot)), "total_mean_s": float(tot.mean()),
                                   "other_mean_s": float(other.mean()), "stages": st,
                                   "same_output_as_latency_run": int(sum(r["same_output_as_latency_run"] for r in rows))}
        del retriever, ctx
        torch.cuda.empty_cache()
    prof.stop()
    mem["device_peak_by_stage_mib"] = dict(prof.dev_peak)
    mem["device_peak_mib"] = max(prof.dev_peak.values())
    write_json(OUT / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
