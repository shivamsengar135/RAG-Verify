"""Reviewer 1 (camera-ready): a finer classification of the verifier's decisions, and the effect of the 12-claim cap.
From the saved RAG-Verify traces (Qwen2.5-7B, 900 test questions per dataset); no model is run.

Each first draft gets exactly one verdict class, in this order:
  refused               the model already answered "I don't know"
  contradicted          the answer check says the passages point to a different answer (judge: CONTRADICTED)
  no evidence           the answer check finds no evidence for the answer (judge: NOT_ENOUGH_INFO)
  detail contradicted   answer accepted, but NLI labels a supporting claim contradicted
  partially supported   answer accepted, but NLI labels a supporting claim unsupported (none contradicted)
  fully supported       every claim supported, so the draft is answered unchanged
For each class: drafts, and how many were already correct (for the flagged classes these are false alarms).
Wrong drafts that were fully supported are split into missing evidence (no gold answer string in the retrieved
passages), near miss (token F1 >= 0.5 with a gold answer) and wrong entity (gold answer retrieved, another answer given).
Claims per draft: distribution of m and how many drafts reached the 12-claim cap of the decomposition.

Usage: .venv\\Scripts\\python.exe code\\verdict_breakdown.py
Output: results/error_analysis/verdicts.json, paper_ictcs_crc/tables/verdict_rows.tex
"""
import collections

import numpy as np

from common import ROOT, read_jsonl, write_json
from evaluate import answer_scores, normalize
from methods import is_idk, parse_response

CLASSES = [("refused", "Draft is already a refusal"),
           ("contradicted", "Answer check: evidence points elsewhere"),
           ("no evidence", "Answer check: no evidence for the answer"),
           ("detail contradicted", "Answer accepted, a detail contradicted"),
           ("partially supported", "Answer accepted, a detail unsupported"),
           ("fully supported", "All claims supported (answered as is)")]


def verdict(t):
    if is_idk(parse_response(t["initial"])[0]):
        return "refused"
    claims = t["claims0"]
    assert claims[0].get("judge") == "answer"
    if claims[0]["label"] == "contradicted":
        return "contradicted"
    if claims[0]["label"] != "supported":
        return "no evidence"
    rest = [c["label"] for c in claims[1:]]
    if "contradicted" in rest:
        return "detail contradicted"
    if "unsupported" in rest:
        return "partially supported"
    return "fully supported"


def main():
    out, rows = {}, []
    for ds in ["hotpotqa", "triviaqa"]:
        corpus = {p["pid"]: p["text"] for p in read_jsonl(ROOT / "data" / ds / "corpus.jsonl")}
        n, correct = collections.Counter(), collections.Counter()
        passed_wrong, m0, m1 = collections.Counter(), [], []
        for sp in ["seed0", "seed1", "seed2"]:
            for r in read_jsonl(ROOT / "results" / "raw" / ds / "ragverify" / f"{sp}.jsonl"):
                t = r["trace"]
                v = verdict(t)
                a0 = parse_response(t["initial"])[0]
                s = answer_scores(a0, r["answers"])
                ok = v != "refused" and bool(s["em"] or s["match"])
                n[v] += 1
                correct[v] += ok
                m0.append(len(t["claims0"]))
                m1 += [len(t["claims1"])] if "claims1" in t else []
                if v == "fully supported" and not ok:
                    ptext = " " + normalize(" ".join(corpus[p] for p in t["passages"])) + " "
                    if not any(normalize(g) and f" {normalize(g)} " in ptext for g in r["answers"]):
                        passed_wrong["missing evidence"] += 1
                    elif s["f1"] >= 0.5:
                        passed_wrong["near miss"] += 1
                    else:
                        passed_wrong["wrong entity"] += 1
        m_all = np.array(m0 + m1)
        out[ds] = {"n": sum(n.values()), "drafts": {k: n[k] for k, _ in CLASSES},
                   "correct": {k: correct[k] for k, _ in CLASSES},
                   "flagged": sum(n[k] for k in ("contradicted", "no evidence", "detail contradicted", "partially supported")),
                   "flagged_correct": sum(correct[k] for k in ("contradicted", "no evidence", "detail contradicted",
                                                               "partially supported")),
                   "passed_wrong": dict(passed_wrong),
                   "claims": {"decompositions": int(len(m_all)), "median": float(np.median(m_all)),
                              "p95": float(np.percentile(m_all, 95)), "max": int(m_all.max()),
                              "at_cap": int((m_all >= 12).sum())}}
    for k, label in CLASSES:
        cells = [label]
        for ds in ["hotpotqa", "triviaqa"]:
            cells += [str(out[ds]["drafts"][k]), "--" if k == "refused" else str(out[ds]["correct"][k])]
        rows.append(" & ".join(cells) + " \\\\")
    write_json(ROOT / "results" / "error_analysis" / "verdicts.json", out)
    dst = ROOT / "paper_ictcs_crc" / "tables"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "verdict_rows.tex").write_text("% AUTO-GENERATED by code/verdict_breakdown.py -- do not edit by hand\n"
                                          + "\n".join(rows) + "\n", encoding="utf-8")
    for ds, d in out.items():
        print(ds, d)


if __name__ == "__main__":
    main()
