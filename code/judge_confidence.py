"""Reviewer 2 (camera-ready): can a calibrated verifier confidence cut unnecessary abstentions while still rejecting
wrong answers? Post-hoc analysis added after review; it is not part of the pre-registered protocol.

The hybrid verifier's answer check (frozen Qwen2.5-7B, same prompt, same seed per block) is re-run on every first
draft with token log-probabilities. Its confidence c is P(SUPPORTED) of the first generated token, normalised over
the three verdicts (refusal drafts get c = 0). First drafts and self-revision outputs come from the saved
RAG + self-revision runs, whose first drafts equal RAG-Verify's on all 1,800 test questions. Two policies:
  gate     answer the first draft if c >= t, else abstain        (RAG-Verify's gate with a continuous score, no revision)
  revise   keep the first draft if c >= t, else take the self-revision output   (revise only when unsure)
For each policy one threshold t is chosen on the dev block (100 questions per dataset) by the original selection rule
(maximise mean Phi_1 over the two datasets; within 0.01 of the best, prefer higher coverage) and applied once to the
three test blocks. Paired bootstrap against RAG-Verify and RAG + self-revision as in code/aggregate.py.

Usage: .venv\\Scripts\\python.exe code\\judge_confidence.py            (scores, resumable, then analysis)
       .venv\\Scripts\\python.exe code\\judge_confidence.py --analyze  (analysis only)
Output: results/judge_conf/<dataset>_<split>.jsonl, results/judge_conf/summary.json
"""
import argparse
import json
import math

import httpx
import numpy as np

from common import ROOT, append_jsonl, load_config, read_jsonl, write_json
from evaluate import answer_scores
from halueval_detect import auroc
from methods import is_idk, parse_response
from run import split_seed
from verifiers import ANSWER_JUDGE_PROMPT, _verdict

CFG = load_config()
OUT = ROOT / "results" / "judge_conf"
DATASETS, SPLITS = ["hotpotqa", "triviaqa"], {"dev": "dev_full", "seed0": "seed0", "seed1": "seed1", "seed2": "seed2"}


def judge_confidence(client, prompt, seed):
    body = {"model": CFG["llm"]["backbone"], "messages": [{"role": "user", "content": prompt}], "stream": False,
            "keep_alive": "30m", "logprobs": True, "top_logprobs": 10,
            "options": {"num_ctx": CFG["llm"]["num_ctx"], "num_predict": 8, "temperature": CFG["llm"]["temperature"],
                        "seed": seed}}
    r = client.post("http://localhost:11434/api/chat", json=body, timeout=600).json()
    lp = r.get("logprobs") or r["message"].get("logprobs")
    mass = {"S": 0.0, "C": 0.0, "N": 0.0}
    for t in lp[0]["top_logprobs"]:  # Qwen splits the verdict, e.g. "S" + "UPPORTED": match on prefixes
        tok = t["token"].strip().upper()
        for verdict in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"):
            if tok and (verdict.startswith(tok) or tok.startswith(verdict)):
                mass[verdict[0]] += math.exp(t["logprob"])
    tot = sum(mass.values())
    return {"raw": r["message"]["content"].strip(), "first_token": lp[0]["token"], "mass": mass,
            "c": mass["S"] / tot if tot else None}


def score_all():
    client = httpx.Client()
    for ds in DATASETS:
        corpus = {p["pid"]: p["text"] for p in read_jsonl(ROOT / "data" / ds / "corpus.jsonl")}
        for sp, f in SPLITS.items():
            path = OUT / f"{ds}_{sp}.jsonl"
            done = {r["id"] for r in read_jsonl(path)}
            saved = {}
            if sp != "dev":  # RAG-Verify's own verdict on the same draft, for an agreement check
                for r in read_jsonl(ROOT / "results/raw" / ds / "ragverify" / f"{sp}.jsonl"):
                    saved[r["id"]] = _verdict(r["trace"]["claims0"][0].get("raw", ""))
            todo = [r for r in read_jsonl(ROOT / "results/raw" / ds / "rv_no_verifier" / f"{f}.jsonl") if r["id"] not in done]
            print(f"{ds} {sp}: {len(todo)} to score", flush=True)
            for i, r in enumerate(todo, 1):
                a0 = parse_response(r["trace"]["initial"])[0]
                rec = {"id": r["id"], "refusal": is_idk(a0), "saved_verdict": saved.get(r["id"])}
                if not rec["refusal"]:
                    ptxt = "\n".join(f"[{j + 1}] {corpus[p]}" for j, p in enumerate(r["trace"]["passages"]))
                    rec.update(judge_confidence(client, ANSWER_JUDGE_PROMPT.format(passages=ptxt, question=r["question"],
                                                                                   answer=a0), split_seed(sp)))
                append_jsonl(path, rec)
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)


def load_block(ds, sp):
    conf = {r["id"]: r for r in read_jsonl(OUT / f"{ds}_{sp}.jsonl")}
    raws = read_jsonl(ROOT / "results/raw" / ds / "rv_no_verifier" / f"{SPLITS[sp]}.jsonl")
    sr = {x["id"]: x for x in json.loads((ROOT / "results/metrics" / ds / "rv_no_verifier" / f"{SPLITS[sp]}.json")
                                         .read_text(encoding="utf-8"))["per_example"]}
    rows = []
    for r in raws:
        a0 = parse_response(r["trace"]["initial"])[0]
        s = answer_scores(a0, r["answers"])
        k = conf[r["id"]]
        rows.append({"id": r["id"], "c": 0.0 if k["refusal"] or k.get("c") is None else k["c"],
                     "draft": 0 if k["refusal"] else (1 if s["em"] or s["match"] else -1),  # +1 right, -1 wrong, 0 refusal
                     "rev": 1 if sr[r["id"]]["correct"] else (-1 if sr[r["id"]]["answered"] else 0)})
    return rows


def policy(rows, t, kind):
    out = []
    for r in rows:
        if r["c"] >= t and r["draft"] != 0:
            out.append(r["draft"])
        else:
            out.append(0 if kind == "gate" else r["rev"])
    return np.array(out)


def phi_ids(ds, m):
    s = {}
    for sp in ["seed0", "seed1", "seed2"]:
        for x in json.loads((ROOT / "results/metrics" / ds / m / f"{sp}.json").read_text(encoding="utf-8"))["per_example"]:
            s[x["id"]] = 1 if x["correct"] else (-1 if x["answered"] else 0)
    return s


def analyze():
    dev = {ds: load_block(ds, "dev") for ds in DATASETS}
    test = {ds: {sp: load_block(ds, sp) for sp in ["seed0", "seed1", "seed2"]} for ds in DATASETS}
    summary = {"note": "post-hoc analysis added after review", "agreement": {}, "auroc": {}}
    for ds in DATASETS:  # does the re-run reproduce the saved verdicts, and does c separate right from wrong drafts?
        recs = [r for sp in ["seed0", "seed1", "seed2"] for r in read_jsonl(OUT / f"{ds}_{sp}.jsonl") if not r["refusal"]]
        summary["agreement"][ds] = {"n": len(recs), "same_verdict": sum(_verdict(r["raw"]) == r["saved_verdict"] for r in recs)}
        rows = [r for sp in test[ds].values() for r in sp if r["draft"] != 0]
        summary["auroc"][ds] = float(auroc([r["c"] for r in rows], [r["draft"] == 1 for r in rows]))
    grid = sorted({r["c"] for ds in DATASETS for r in dev[ds]} | {1.01})
    rng = np.random.default_rng(0)
    base = {ds: {m: phi_ids(ds, m) for m in ["ragverify", "rv_no_verifier"]} for ds in DATASETS}
    summary["lost_correct_drafts_baselines"] = {  # correct first drafts that do not end correct, same definition as below
        ds: {m: sum(r["draft"] == 1 and base[ds][m][r["id"]] != 1 for sp in test[ds].values() for r in sp)
             for m in base[ds]} for ds in DATASETS}
    for kind in ["gate", "revise"]:
        cand = []
        for t in grid:
            ph = [policy(dev[ds], t, kind) for ds in DATASETS]
            cand.append((float(np.mean([p.mean() for p in ph])), float(np.mean([(p != 0).mean() for p in ph])), t))
        best = max(c[0] for c in cand)
        phi_dev, cov_dev, t = max((c for c in cand if c[0] >= best - 0.01), key=lambda c: (c[1], c[0]))
        res = {"threshold": t, "dev_phi1": phi_dev, "dev_coverage": cov_dev}
        for ds in DATASETS:
            per = {sp: policy(rows, t, kind) for sp, rows in test[ds].items()}
            phis = [p.mean() for p in per.values()]
            pooled = np.concatenate(list(per.values()))
            ids = [r["id"] for sp in test[ds].values() for r in sp]
            drafts = np.array([r["draft"] for sp in test[ds].values() for r in sp])
            d = {"phi1_mean": float(np.mean(phis)), "phi1_std": float(np.std(phis, ddof=1)),
                 "abstention": float((pooled == 0).mean()), "hallucination": float((pooled == -1).mean()),
                 "accuracy": float((pooled == 1).mean()),
                 "lost_correct_drafts": int(((drafts == 1) & (pooled != 1)).sum()),
                 "wrong_drafts_answered": int(((drafts == -1) & (pooled == -1)).sum())}
            for m in ["ragverify", "rv_no_verifier"]:
                diff = pooled - np.array([base[ds][m][i] for i in ids], dtype=float)
                boots = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                d[f"vs_{m}"] = {"delta": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}
            res[ds] = d
        summary[kind] = res
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.analyze:
        score_all()
    analyze()


if __name__ == "__main__":
    main()
