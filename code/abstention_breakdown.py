"""Where do RAG-Verify's abstentions come from, and how many discard a correct answer? (main grid, Qwen backbone)

For each test question: did the verifier flag any claim in the first draft, did the revision itself say IDK,
or did the final verifier gate force abstention? For gate-forced abstentions, the withheld answer is scored
against the gold answers to measure over-abstention. Output: results/tables/abstention_breakdown.json
Usage: .venv\Scripts\python.exe code\abstention_breakdown.py
"""
import json

from common import ROOT, read_jsonl, write_json
from evaluate import answer_scores
from methods import is_idk, parse_response


def main():
    out = {}
    for ds in ["hotpotqa", "triviaqa"]:
        c = dict(n=0, flagged_initial=0, idk_by_model=0, idk_after_revision=0, idk_first_draft=0, gate_abstain=0, gate_withheld_correct=0, answered=0,
                 initial_correct=0, flagged_initial_correct=0)
        for sp in ["seed0", "seed1", "seed2"]:
            for r in read_jsonl(ROOT / "results" / "raw" / ds / "ragverify" / f"{sp}.jsonl"):
                t = r["trace"]
                c["n"] += 1
                init_ans = parse_response(t["initial"])[0]
                init_ok = (not is_idk(init_ans)) and any(answer_scores(init_ans, r["answers"])[k] for k in ("em", "match"))
                c["initial_correct"] += init_ok
                flagged = any(x["label"] != "supported" for x in t.get("claims0", []))
                c["flagged_initial"] += flagged
                c["flagged_initial_correct"] += flagged and init_ok
                final_ans = parse_response(t.get("revised", t["initial"]))[0]
                if t.get("verifier_abstained"):
                    c["gate_abstain"] += 1
                    s = answer_scores(final_ans, r["answers"])
                    c["gate_withheld_correct"] += bool(s["em"] or s["match"])
                elif r["abstained"]:
                    c["idk_by_model"] += 1
                    if "revised" in t and is_idk(parse_response(t["revised"])[0]) and not is_idk(init_ans):
                        c["idk_after_revision"] += 1
                    else:
                        c["idk_first_draft"] += 1
                else:
                    c["answered"] += 1
        out[ds] = c
        print(ds, json.dumps(c))
    write_json(ROOT / "results" / "tables" / "abstention_breakdown.json", out)


if __name__ == "__main__":
    main()
