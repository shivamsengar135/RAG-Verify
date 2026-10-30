"""Aggregate per-seed metrics into mean +/- std tables and paired bootstrap tests (main grid).

Effective reliability Phi_1 = accuracy - hallucination rate (+1 correct, -1 wrong answer, 0 abstain).
Std is the sample std over the three disjoint 300-question test blocks (seed0-2).
Paired bootstrap (10k resamples, seed 0) over the 900 pooled test questions of each dataset.

Usage: .venv\\Scripts\\python.exe code\\aggregate.py
Output: results/tables/main_summary.csv, results/tables/main_tests.csv, paper/tables/*.tex
"""
import csv
import json

import numpy as np

from common import ROOT

DATASETS = ["hotpotqa", "triviaqa"]
SEEDS = ["seed0", "seed1", "seed2"]
SYSTEMS = [  # (method id, display name, backbone)
    ("closed_book", "Closed-book", "Qwen2.5-7B"),
    ("rag", "Standard RAG", "Qwen2.5-7B"),
    ("rag_abstain", "RAG + abstain prompt", "Qwen2.5-7B"),
    ("rv_no_verifier", "RAG + self-revision", "Qwen2.5-7B"),
    ("cove", "CoVe (closed-book)", "Qwen2.5-7B"),
    ("cove_rag", "CoVe + RAG", "Qwen2.5-7B"),
    ("ragverify", "RAG-Verify (ours)", "Qwen2.5-7B"),
    ("closed_book_llama", "Closed-book", "Llama-3.1-8B"),
    ("rag_llama", "Standard RAG", "Llama-3.1-8B"),
    ("rag_abstain_llama", "RAG + abstain prompt", "Llama-3.1-8B"),
    ("rv_no_verifier_llama", "RAG + self-revision", "Llama-3.1-8B"),
    ("ragverify_llama", "RAG-Verify (ours)", "Llama-3.1-8B"),
    ("selfrag", "Self-RAG", "Llama-2-7B (NF4)"),
]
METRICS = ["accuracy", "f1", "abstention_rate", "hallucination_rate", "selective_accuracy", "phi1", "llm_calls"]
PAIRS = [("ragverify", b) for b in ["rag", "rag_abstain", "rv_no_verifier", "cove_rag", "closed_book"]] + \
        [("rv_no_verifier", "rag_abstain"), ("ragverify_llama", "rag_llama"), ("ragverify_llama", "rag_abstain_llama"),
         ("ragverify_llama", "rv_no_verifier_llama")]


def load(ds, m, sp):
    p = ROOT / "results" / "metrics" / ds / m / f"{sp}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if d["summary"]["n"] != 300 or d["summary"]["errors"]:
        return None  # incomplete or errored run: treated as pending
    return d


def phi_scores(d):
    return {x["id"]: (1 if x["correct"] else (-1 if x["answered"] else 0)) for x in d["per_example"]}


def main():
    rows, avail = [], {}
    for ds in DATASETS:
        for m, name, bb in SYSTEMS:
            runs = [load(ds, m, sp) for sp in SEEDS]
            if any(r is None for r in runs):
                rows.append({"dataset": ds, "method": m, "system": name, "backbone": bb, "status": "pending"})
                continue
            avail[(ds, m)] = runs
            row = {"dataset": ds, "method": m, "system": name, "backbone": bb, "status": "done"}
            for k in METRICS:
                vals = [r["summary"]["accuracy"] - r["summary"]["hallucination_rate"] if k == "phi1"
                        else r["summary"][k] for r in runs]
                row[k + "_mean"], row[k + "_std"] = float(np.mean(vals)), float(np.std(vals, ddof=1))
            rows.append(row)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "method", "system", "backbone", "status"), k))
    with (out / "main_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(0)
    tests = []
    for ds in DATASETS:
        for a, b in PAIRS:
            if (ds, a) not in avail or (ds, b) not in avail:
                tests.append({"dataset": ds, "a": a, "b": b, "status": "pending"})
                continue
            sa, sb = {}, {}
            for ra, rb in zip(avail[(ds, a)], avail[(ds, b)]):
                sa.update(phi_scores(ra))
                sb.update(phi_scores(rb))
            ids = sorted(sa)
            diff = np.array([sa[i] - sb[i] for i in ids], dtype=float)
            boots = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            tests.append({"dataset": ds, "a": a, "b": b, "status": "done", "n": len(ids), "delta_phi1": diff.mean(),
                          "ci_lo": lo, "ci_hi": hi, "p_two_sided": min(p, 1.0)})
    with (out / "main_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "a", "b", "status", "n", "delta_phi1", "ci_lo", "ci_hi", "p_two_sided"])
        w.writeheader()
        w.writerows(tests)
    write_latex(rows, tests)
    for r in rows:
        if r["status"] == "done":
            print(f"{r['dataset']:9s} {r['method']:22s} acc={r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} "
                  f"abst={r['abstention_rate_mean']:.3f} hall={r['hallucination_rate_mean']:.3f} "
                  f"phi1={r['phi1_mean']:.3f}±{r['phi1_std']:.3f}")
        else:
            print(f"{r['dataset']:9s} {r['method']:22s} PENDING")
    for t in tests:
        if t["status"] == "done":
            print(f"{t['dataset']:9s} {t['a']} - {t['b']}: dPhi1={t['delta_phi1']:+.3f} "
                  f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}] p={t['p_two_sided']:.4f}")


def pm(r, k, scale=100):
    return f"{scale * r[k + '_mean']:.1f}".replace("-", "$-$") + f"$\\pm${scale * r[k + '_std']:.1f}"


def write_latex(rows, tests):
    by = {(r["dataset"], r["method"]): r for r in rows}
    lines = []
    last_bb = None
    for m, name, bb in SYSTEMS:
        if bb != last_bb:
            if last_bb is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{Backbone: {bb}}}}}\\\\")
            last_bb = bb
        cells = [name.replace("(ours)", "\\textbf{(ours)}")]
        for ds in DATASETS:
            r = by[(ds, m)]
            if r["status"] != "done":
                cells += ["\\pending{}"] * 4
            else:
                cells += [pm(r, "accuracy"), pm(r, "abstention_rate"), pm(r, "hallucination_rate"), pm(r, "phi1")]
        lines.append(" & ".join(cells) + " \\\\")
    short = {"closed_book": "Closed-book", "rag": "RAG", "rag_abstain": "Abstain prompt", "rv_no_verifier": "Self-revision",
             "cove_rag": "CoVe+RAG", "ragverify": "RAG-Verify"}
    names = {m: short[m.replace("_llama", "")] + (" (L)" if m.endswith("_llama") else " (Q)") for m, _, _ in SYSTEMS
             if m.replace("_llama", "") in short}
    tl = []
    for a, b in PAIRS:
        cells = [f"{names[a].replace(' (ours)', '')} vs.\\ {names[b]}"]
        for ds in DATASETS:
            t = next(x for x in tests if x["dataset"] == ds and x["a"] == a and x["b"] == b)
            if t["status"] != "done":
                cells.append("\\pending{}")
            else:
                cells.append(f"{100 * t['delta_phi1']:+.1f} [{100 * t['ci_lo']:+.1f}, {100 * t['ci_hi']:+.1f}]".replace("-", "$-$")
                             + ("$^{*}$" if t["ci_lo"] > 0 or t["ci_hi"] < 0 else ""))
        tl.append(" & ".join(cells) + " \\\\")
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "tables" / "tests_rows.tex").write_text(
        "% AUTO-GENERATED by code/aggregate.py -- do not edit by hand\n" + "\n".join(tl) + "\n", encoding="utf-8")
    (ROOT / "paper" / "tables" / "main_rows.tex").write_text(
        "% AUTO-GENERATED by code/aggregate.py from results/metrics -- do not edit by hand\n" + "\n".join(lines) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
