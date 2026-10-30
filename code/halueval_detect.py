"""RQ5: can RAG-Verify's verifiers tell a correct answer from a hallucinated one, given the gold knowledge?

HaluEval-QA (data/halueval/items.jsonl: 1,000 sampled questions x {correct, hallucinated} answer = 2,000 items).
For each item the answer is turned into the same question-grounded answer claim RAG-Verify uses, and scored by
  - NLI (frozen DeBERTa-v3 model): hallucination score = 1 - P(entailment | knowledge, claim)
  - LLM answer judge (frozen Qwen2.5-7B backbone, the hybrid verifier's answer check): flag unless SUPPORTED
Reports AUROC (NLI score, and the judge's binary flag) and flag accuracy / precision / recall / F1 at tau = 0.3,
with 95% bootstrap CIs (1,000 resamples over questions, seed 0). Resumable (per-item cache).

Usage: .venv\\Scripts\\python.exe code\\halueval_detect.py [--n 20]
Output: results/halueval/scores.jsonl, results/halueval/summary.json
"""
import argparse
import json

import numpy as np

from common import LLM, ROOT, append_jsonl, load_config, read_jsonl, write_json
from verifiers import LLMJudgeVerifier, NLIVerifier

CFG = load_config()
OUT = ROOT / "results" / "halueval"


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties; labels 1 = hallucinated."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def flag_metrics(flags, labels):
    f, y = np.asarray(flags, bool), np.asarray(labels, bool)
    tp, fp, fn = (f & y).sum(), (f & ~y).sum(), (~f & y).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": float((f == y).mean()), "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit to first n items (smoke test)")
    args = ap.parse_args()
    items = read_jsonl(ROOT / "data" / "halueval" / "items.jsonl")[: args.n]
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / ("scores.jsonl" if args.n is None else f"scores_n{args.n}.jsonl")
    done = {r["id"] for r in read_jsonl(cache_path)} if cache_path.exists() else set()
    todo = [x for x in items if x["id"] not in done]
    print(f"halueval: {len(items)} items, {len(todo)} to score", flush=True)
    if todo:
        nli = NLIVerifier(CFG)
        judge = LLMJudgeVerifier(LLM(CFG["llm"]["backbone"], CFG["llm"]["num_ctx"], CFG["llm"]["max_new_tokens"],
                                     CFG["llm"]["temperature"]))
        for i, x in enumerate(todo, 1):
            claim = f'The answer to the question "{x["question"]}" is {x["answer"].strip().rstrip(".")}.'
            n = nli.score([claim], [x["knowledge"]])[0]
            j = judge.score_answer(x["question"], x["answer"], [{"text": x["knowledge"]}])
            append_jsonl(cache_path, {"id": x["id"], "label": x["label"], "nli_ent": n["ent"], "nli_con": n["con"],
                                      "judge_raw": j["raw"], "judge_supported": bool(j["ent"])})
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    rows = read_jsonl(cache_path)
    y = np.array([r["label"] == "hallucinated" for r in rows], int)
    tau = CFG["verifier"]["tau"]
    systems = {  # name -> (continuous hallucination score, binary flag)
        "nli": (np.array([1 - r["nli_ent"] for r in rows]), np.array([r["nli_ent"] < tau for r in rows])),
        "llm_judge": (np.array([0.0 if r["judge_supported"] else 1.0 for r in rows]),
                      np.array([not r["judge_supported"] for r in rows])),
    }
    qid = np.array([r["id"].rsplit("_", 1)[0] for r in rows])
    groups = {q: np.flatnonzero(qid == q) for q in np.unique(qid)}
    keys = list(groups)
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))]) for _ in range(1000)]
    summary = {"n_items": len(rows), "n_hallucinated": int(y.sum()), "tau": tau}
    for name, (score, flag) in systems.items():
        res = {"auroc": float(auroc(score, y)), **flag_metrics(flag, y)}
        bs = np.array([[auroc(score[b], y[b]), flag_metrics(flag[b], y[b])["f1"]] for b in boot_idx])
        res["auroc_ci"] = [float(v) for v in np.percentile(bs[:, 0], [2.5, 97.5])]
        res["f1_ci"] = [float(v) for v in np.percentile(bs[:, 1], [2.5, 97.5])]
        summary[name] = res
        print(f"{name:9s} AUROC={res['auroc']:.3f} [{res['auroc_ci'][0]:.3f}, {res['auroc_ci'][1]:.3f}] "
              f"acc={res['accuracy']:.3f} P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f}")
    write_json(OUT / ("summary.json" if args.n is None else f"summary_n{args.n}.json"), summary)
    print(json.dumps(summary)[:300])


if __name__ == "__main__":
    main()
