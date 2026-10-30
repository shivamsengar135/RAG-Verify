"""Run one system on one dataset split. Resumable: already-finished example ids are skipped.

Usage:
  .venv\\Scripts\\python.exe code\\run.py --dataset hotpotqa --method ragverify --split seed0
  .venv\\Scripts\\python.exe code\\run.py --dataset triviaqa --method cove --split dev --n 20
Output: results/raw/<dataset>/<method>/<split>.jsonl ; log: logs/run_<dataset>_<method>_<split>.log
"""
import argparse
import copy
import json
import logging
import time
from types import SimpleNamespace

import torch
import yaml

from common import (LLM, ROOT, CostMeter, append_jsonl, deep_update, load_config, read_jsonl, set_seed,
                    thermal_pause)


def load_examples(dataset, split, n=None):
    qs = {q["id"]: q for q in read_jsonl(ROOT / "data" / dataset / "questions.jsonl")}
    ids = json.loads((ROOT / "data" / dataset / "splits.json").read_text())[split]
    exs = [qs[i] for i in ids]
    return exs[:n] if n else exs


def split_seed(split):
    return int(split[4:]) if split.startswith("seed") else 99


def build_context(dataset, method_cfg, cfg, seed):
    from methods import METHODS  # noqa: F401  (import check)

    fn, flags = method_cfg["fn"], method_cfg.get("flags", {})
    ctx = SimpleNamespace(cfg=cfg, seed=seed, meter=None, retriever=None, nli=None, judge=None)
    ctx.llm = LLM(cfg["llm"]["backbone"], cfg["llm"]["num_ctx"], cfg["llm"]["max_new_tokens"],
                  cfg["llm"]["temperature"], seed=seed)
    needs_retrieval = fn == "rag" or (fn == "cove" and flags.get("retrieval")) or \
        (fn == "ragverify" and flags.get("retrieval", True))
    if needs_retrieval:
        from retriever import Retriever

        ctx.retriever = Retriever(dataset, cfg)
    if fn == "ragverify" and flags.get("verifier", "nli") in ("nli", "hybrid"):
        from verifiers import NLIVerifier

        ctx.nli = NLIVerifier(cfg)
    if fn == "ragverify" and flags.get("verifier") in ("llm_judge", "hybrid"):
        from verifiers import LLMJudgeVerifier

        ctx.judge = LLMJudgeVerifier(ctx.llm)
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "triviaqa"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="seed0")
    ap.add_argument("--n", type=int, default=None, help="limit to first n examples of the split")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. smoke)")
    args = ap.parse_args()

    registry = yaml.safe_load((ROOT / "configs" / "methods.yaml").read_text())
    mcfg = registry[args.method]
    if mcfg["fn"] == "selfrag":
        raise SystemExit("Self-RAG uses code/selfrag_run.py")
    cfg = deep_update(copy.deepcopy(load_config()), mcfg.get("cfg", {}))
    if mcfg["fn"] == "ragverify":  # default verifier comes from the frozen config unless the variant overrides it
        mcfg["flags"] = {"verifier": cfg["verifier"]["kind"], **mcfg.get("flags", {})}
    seed = split_seed(args.split)
    set_seed(seed)

    name = f"{args.split}{'_' + args.tag if args.tag else ''}"
    out = ROOT / "results" / "raw" / args.dataset / args.method / f"{name}.jsonl"
    log_path = ROOT / "logs" / f"run_{args.dataset}_{args.method}_{name}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger()

    examples = load_examples(args.dataset, args.split, args.n)
    done = {r["id"] for r in read_jsonl(out)}
    todo = [e for e in examples if e["id"] not in done]
    log.info("start method=%s dataset=%s split=%s n=%d todo=%d cfg=%s", args.method, args.dataset, args.split,
             len(examples), len(todo), json.dumps(mcfg))
    print(f"{args.method} | {args.dataset} | {name}: {len(todo)}/{len(examples)} to run")

    from methods import METHODS

    ctx = build_context(args.dataset, mcfg, cfg, seed)
    fn, flags = METHODS[mcfg["fn"]], mcfg.get("flags", {})
    ctx.llm.warmup()
    if ctx.retriever is not None:
        ctx.retriever.search("warm-up query", 1)
    if ctx.nli is not None:
        ctx.nli.score(["warm-up claim"], ["warm-up premise"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i, ex in enumerate(todo, 1):
        thermal_pause(log)  # outside the timed region
        ctx.meter = CostMeter()
        t0 = time.perf_counter()
        try:
            res = fn(ex, ctx, flags)
            err = None
        except Exception as e:  # record the failure and keep going; failures are reported in evaluation
            log.exception("example %s failed", ex["id"])
            if "CUDA" in repr(e) or "out of memory" in repr(e).lower():
                # Environment failure, not a model outcome: stop without recording; rerun resumes here.
                raise SystemExit(f"CUDA failure on {ex['id']}; stopping so it is not scored as an abstention")
            res, err = {"output_text": "", "prediction": "", "abstained": True}, repr(e)
        rec = {"id": ex["id"], "question": ex["question"], "answers": ex["answers"], "method": args.method,
               "dataset": args.dataset, "split": args.split, "backbone": cfg["llm"]["backbone"],
               **res, "error": err, "latency_s": round(time.perf_counter() - t0, 3), **ctx.meter.as_dict()}
        append_jsonl(out, rec)
        log.info("[%d/%d] %s pred=%r abst=%s %.1fs calls=%d", i, len(todo), ex["id"], rec["prediction"][:60],
                 rec["abstained"], rec["latency_s"], rec["llm_calls"])
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} last={rec['latency_s']:.1f}s")
    if torch.cuda.is_available():
        log.info("peak_torch_vram_mib=%d", torch.cuda.max_memory_allocated() // 2**20)
    log.info("done")


if __name__ == "__main__":
    main()
