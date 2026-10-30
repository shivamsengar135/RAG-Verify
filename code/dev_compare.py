"""Dev-block comparison of all systems on effective reliability (Phi_1), with paired bootstrap CIs.

Per question: +1 correct answer, -1 wrong answer, 0 abstention. Phi_1 = mean score.
The selected RAG-Verify configuration (results/tuning/selected.json) is re-scored from its saved traces with
the selected rho, exactly as code/tune_dev.py does.

Usage: .venv\\Scripts\\python.exe code\\dev_compare.py   -> results/tuning/dev_compare.csv
"""
import json

import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from evaluate import answer_scores
from methods import is_idk, parse_response
from verifiers import label_from_scores

SYSTEMS = ["rag", "rag_abstain", "rv_no_verifier", "rv_no_decomp", "ragverify", "rv_llm_judge"]
DATASETS = ["hotpotqa", "triviaqa"]


def score(r, tau=None, rho=None):
    answer = parse_response(r["output_text"])[0]
    if r.get("error") or is_idk(answer):
        return 0
    if tau is None:
        if r["abstained"]:
            return 0
    else:
        claims = r.get("trace", {}).get("claims1") or r.get("trace", {}).get("claims0") or []
        labels = [label_from_scores(c["ent"], c["con"], tau) for c in claims]
        ratio = sum(l == "supported" for l in labels) / max(len(labels), 1)
        if not labels or labels[0] != "supported" or ratio < rho:
            return 0
    s = answer_scores(answer, r["answers"])
    return 1 if (s["em"] or s["match"]) else -1


def main():
    sel = json.loads((ROOT / "results" / "tuning" / "selected.json").read_text())
    per = {}
    for ds in DATASETS:
        for m in SYSTEMS:
            recs = sorted(read_jsonl(ROOT / "results" / "raw" / ds / m / "dev_full.jsonl"), key=lambda r: r["id"])
            per[(m, ds)] = np.array([score(r) for r in recs])
        recs = sorted(read_jsonl(ROOT / "results" / "raw" / ds / sel["method"] / "dev_full.jsonl"), key=lambda r: r["id"])
        per[("SELECTED", ds)] = np.array([score(r, sel["tau"], sel["rho"]) for r in recs])

    rng = np.random.default_rng(0)
    pooled = {m: np.concatenate([per[(m, d)] for d in DATASETS]) for m in SYSTEMS + ["SELECTED"]}
    n = len(pooled["SELECTED"])
    idx = rng.integers(0, n, size=(10000, n))
    rows = []
    for m in SYSTEMS + ["SELECTED"]:
        diff = pooled["SELECTED"] - pooled[m]
        boots = diff[idx].mean(axis=1)
        rows.append({
            "system": m,
            "phi1_hotpotqa": per[(m, "hotpotqa")].mean(),
            "phi1_triviaqa": per[(m, "triviaqa")].mean(),
            "phi1_mean": (per[(m, "hotpotqa")].mean() + per[(m, "triviaqa")].mean()) / 2,
            "abstention": float((pooled[m] == 0).mean()),
            "selected_minus_this": diff.mean(),
            "ci95_low": np.percentile(boots, 2.5),
            "ci95_high": np.percentile(boots, 97.5),
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tuning" / "dev_compare.csv", index=False)
    print(f"SELECTED = {sel['method']} tau={sel['tau']} rho={sel['rho']}  (n={n} pooled dev questions)")
    print(df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
