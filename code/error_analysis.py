"""Error analysis of RAG-Verify (Qwen2.5-7B, main grid, 900 test questions per dataset), from saved traces.

Two failure types:
  wrong answer    answered and scored incorrect
  lost answer     abstained although the first draft was correct
Each failure gets one rule-based cause (first rule that applies, in this order):
  wrong answer:  retrieval miss  (no gold answer string in any retrieved passage)
                 near miss       (token F1 >= 0.5 with a gold answer: partial / paraphrase, may be a scoring artefact)
                 verifier accept (the answer claim was judged supported)
                 other           (answer claim flagged, but a wrong answer survived revision)
  lost answer:   answer-claim flag   (the question-aware judge rejected the correct answer)
                 support-claim flag  (only NLI-checked supporting claims were flagged; model then refused)
                 retrieval miss among lost answers is reported as a separate count (gold not in passages).
A random sample of 50 failures (25 per dataset, seed 0) is written out in full for manual reading.

Usage: .venv\\Scripts\\python.exe code\\error_analysis.py
Output: results/error_analysis/summary.json, results/error_analysis/sample50.md
"""
import collections
import json

import numpy as np

from common import ROOT, read_jsonl, write_json
from evaluate import answer_scores, normalize
from methods import is_idk, parse_response

OUT = ROOT / "results" / "error_analysis"


def answer_claim(claims):
    return next((c for c in claims if c.get("judge") == "answer"), None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary, sample_md = {}, ["# 50 sampled RAG-Verify failures (Qwen2.5-7B, seed0-2)\n"]
    for ds in ["hotpotqa", "triviaqa"]:
        corpus = {p["pid"]: p["text"] for p in read_jsonl(ROOT / "data" / ds / "corpus.jsonl")}
        metrics = {}
        for sp in ["seed0", "seed1", "seed2"]:
            for x in json.loads((ROOT / "results" / "metrics" / ds / "ragverify" / f"{sp}.json").read_text(encoding="utf-8"))["per_example"]:
                metrics[x["id"]] = x
        fails, counts = [], collections.Counter()
        for sp in ["seed0", "seed1", "seed2"]:
            for r in read_jsonl(ROOT / "results" / "raw" / ds / "ragverify" / f"{sp}.jsonl"):
                m, t = metrics[r["id"]], r["trace"]
                ptext = " " + normalize(" ".join(corpus[p] for p in t["passages"])) + " "
                gold_in = any(normalize(g) and f" {normalize(g)} " in ptext for g in r["answers"])
                init_ans = parse_response(t["initial"])[0]
                init_ok = (not is_idk(init_ans)) and any(answer_scores(init_ans, r["answers"])[k] for k in ("em", "match"))
                ac0 = answer_claim(t.get("claims0", []))
                if m["hallucinated"]:
                    kind = "wrong answer"
                    final_claims = t.get("claims1") or t.get("claims0", [])
                    ac = answer_claim(final_claims)
                    if not gold_in:
                        cause = "retrieval miss"
                    elif m["f1"] >= 0.5:
                        cause = "near miss"
                    elif ac is not None and ac["label"] == "supported":
                        cause = "verifier accept"
                    else:
                        cause = "other"
                elif not m["answered"] and init_ok:
                    kind = "lost answer"
                    cause = "answer-claim flag" if ac0 is not None and ac0["label"] != "supported" else "support-claim flag"
                    counts["lost answer: gold not in passages"] += not gold_in
                else:
                    continue
                counts[kind] += 1
                counts[f"{kind}: {cause}"] += 1
                fails.append({"id": r["id"], "split": sp, "kind": kind, "cause": cause, "question": r["question"],
                              "gold": r["answers"][:3], "initial": t["initial"], "final": r["output_text"],
                              "answer_claim": ac0, "gold_in_passages": gold_in})
        counts["n_questions"] = len(metrics)
        summary[ds] = dict(sorted(counts.items()))
        for k in rng.choice(len(fails), 25, replace=False):
            f = fails[k]
            ac = f["answer_claim"] or {}
            sample_md.append(f"## {ds} {f['split']} {f['id']} -- {f['kind']} / {f['cause']}\n"
                             f"- Q: {f['question']}\n- Gold: {f['gold']} (in passages: {f['gold_in_passages']})\n"
                             f"- First draft: {f['initial']!r}\n- Final: {f['final']!r}\n"
                             f"- Answer-claim verdict: {ac.get('label')} ({ac.get('raw', '')!r})\n")
    write_json(OUT / "summary.json", summary)
    (OUT / "sample50.md").write_text("\n".join(sample_md), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
