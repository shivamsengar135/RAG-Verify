"""Human validation of the automatic claim judge (MiniCheck on gold evidence), 100 claims.

make : sample 100 (claim, gold evidence, MiniCheck verdict) rows from the seed0 claim-precision outputs of
       RAG and RAG-Verify, 25 per (dataset, system), seed 0. Writes results/human_check/sheet.csv (verdict hidden)
       and results/human_check/key.csv (MiniCheck verdicts, not for annotators).
score: read the filled sheet (columns annotator_1, annotator_2: S = supported, N = not supported) and report
       inter-annotator agreement and agreement of MiniCheck with the human labels (Cohen's kappa).

Usage: .venv\\Scripts\\python.exe code\\human_check.py make | score
"""
import csv
import json
import sys

import numpy as np

from common import ROOT
from evaluate import ClaimJudge

OUT = ROOT / "results" / "human_check"
CELLS = [(ds, m) for ds in ("hotpotqa", "triviaqa") for m in ("rag", "ragverify")]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def make():
    rng = np.random.default_rng(0)
    rows = []
    for ds, m in CELLS:
        metrics = json.loads((ROOT / "results" / "metrics" / ds / m / "seed0.json").read_text(encoding="utf-8"))
        raw = {r["id"]: r for r in map(json.loads, (ROOT / "results" / "raw" / ds / m / "seed0.jsonl").open(encoding="utf-8"))}
        pool = [(x["id"], c) for x in metrics["per_example"] for c in x.get("claims", [])]
        if len(pool) < 25:
            sys.exit(f"{ds}/{m}: only {len(pool)} judged claims -- run claim precision on seed0 first")
        judge = ClaimJudge(ds)
        for k in rng.choice(len(pool), 25, replace=False):
            qid, c = pool[k]
            docs = judge.gold_docs(qid, c["claim"])
            rows.append({"dataset": ds, "system": m, "qid": qid, "question": raw[qid]["question"],
                         "claim": c["claim"], "evidence": "\n---\n".join(docs), "minicheck_p": c["p_support"],
                         "minicheck": "S" if c["supported"] else "N"})
    order = rng.permutation(len(rows))  # mix datasets/systems so annotators do not see the source
    rows = [dict(rows[i], item=n + 1) for n, i in enumerate(order)]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["item", "question", "claim", "evidence", "annotator_1", "annotator_2", "notes"])
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in w.fieldnames} for r in rows)
    with (OUT / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "dataset", "system", "qid", "minicheck_p", "minicheck"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
    print(f"wrote {len(rows)} items to {OUT / 'sheet.csv'}")


def score():
    sheet = {r["item"]: r for r in csv.DictReader((OUT / "sheet.csv").open(encoding="utf-8-sig"))}
    key = {r["item"]: r for r in csv.DictReader((OUT / "key.csv").open(encoding="utf-8"))}
    norm = lambda v: v.strip().upper()[:1]  # noqa: E731
    items = [i for i in key if norm(sheet[i]["annotator_1"]) in "SN" and norm(sheet[i]["annotator_2"]) in "SN"
             and sheet[i]["annotator_1"].strip() and sheet[i]["annotator_2"].strip()]
    a1 = [norm(sheet[i]["annotator_1"]) for i in items]
    a2 = [norm(sheet[i]["annotator_2"]) for i in items]
    mc = [key[i]["minicheck"] for i in items]
    agreed = [i for i, x, y in zip(items, a1, a2) if x == y]
    res = {"n_labelled": len(items), "annotator_agreement": float(np.mean(np.array(a1) == np.array(a2))),
           "annotator_kappa": kappa(a1, a2), "n_agreed": len(agreed),
           "minicheck_vs_agreed_accuracy": float(np.mean([key[i]["minicheck"] == norm(sheet[i]["annotator_1"]) for i in agreed])),
           "minicheck_vs_agreed_kappa": kappa([key[i]["minicheck"] for i in agreed], [norm(sheet[i]["annotator_1"]) for i in agreed]),
           "minicheck_vs_annotator_1_kappa": kappa(mc, a1), "minicheck_vs_annotator_2_kappa": kappa(mc, a2)}
    (OUT / "agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
