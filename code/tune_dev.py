"""Select RAG-Verify's verifier design, tau (threshold) and rho (min. supported-claim ratio) on the DEV block only.

Pre-declared rule (written before any dev results were seen, 27 Oct 2026):
  criterion  = effective reliability Phi_1 = accuracy - hallucination_rate
               (correct = +1, wrong answer = -1, abstain = 0; Whitehead et al., ECCV 2022)
  choice     = (tau, rho) maximising Phi_1 averaged over HotpotQA-dev and TriviaQA-dev (100 questions each)
  tie-break  = within 0.01 of the best, prefer higher coverage (fewer abstentions), then tau closer to 0.5
Extension (27 Oct 2026, after the NLI answer-claim diagnosis, BEFORE any hybrid / LLM-judge dev run):
  the search also covers the verifier design {nli, hybrid, llm_judge}; tie-break order becomes
  coverage, then fewer LLM calls per query, then tau closer to 0.5.
tau needs one run per value (it changes which claims trigger revision); rho only changes the final abstention
decision, so it is swept offline from the saved per-claim scores. The LLM judge gives binary verdicts, so tau
has no effect on it and it is run once.

Usage: .venv\\Scripts\\python.exe code\\tune_dev.py      -> results/tuning/dev_tau_rho.csv, prints the choice
"""
import itertools

import pandas as pd

from common import ROOT, read_jsonl, write_json
from evaluate import answer_scores
from methods import is_idk, parse_response
from verifiers import label_from_scores

DESIGNS = {
    "nli": {0.3: "rv_tau03", 0.5: "ragverify", 0.7: "rv_tau07", 0.9: "rv_tau09"},
    "hybrid": {0.3: "rv_hybrid_tau03", 0.5: "rv_hybrid", 0.7: "rv_hybrid_tau07", 0.9: "rv_hybrid_tau09"},
    "llm_judge": {0.5: "rv_llm_judge"},
}
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
DATASETS = ["hotpotqa", "triviaqa"]


def simulate(recs, tau, rho):
    n = correct = wrong = 0
    for r in recs:
        n += 1
        t = r.get("trace", {})
        answer = parse_response(r["output_text"])[0]
        if r.get("error") or is_idk(answer):
            continue  # model itself declined -> abstention regardless of rho
        claims = t.get("claims1") or t.get("claims0") or []
        labels = [label_from_scores(c["ent"], c["con"], tau) for c in claims]
        ratio = sum(l == "supported" for l in labels) / max(len(labels), 1)
        answer_ok = bool(labels) and labels[0] == "supported"
        if (not answer_ok) or ratio < rho:
            continue  # verifier abstains
        s = answer_scores(answer, r["answers"])
        if s["em"] or s["match"]:
            correct += 1
        else:
            wrong += 1
    acc, hall = correct / n, wrong / n
    return {"accuracy": acc, "hallucination_rate": hall, "coverage": (correct + wrong) / n, "phi1": acc - hall,
            "llm_calls": sum(r["llm_calls"] for r in recs) / n}


def main():
    rows = []
    for design, runs in DESIGNS.items():
        for (tau, method), rho, ds in itertools.product(runs.items(), RHOS, DATASETS):
            recs = read_jsonl(ROOT / "results" / "raw" / ds / method / "dev_full.jsonl")
            assert len(recs) == 100, f"{ds}/{method}: {len(recs)} records"
            assert not any(r.get("error") for r in recs), f"{ds}/{method}: runtime errors"
            rows.append({"design": design, "method": method, "dataset": ds, "tau": tau, "rho": rho,
                         **simulate(recs, tau, rho)})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dev_design_tau_rho.csv", index=False)

    cols = ["phi1", "accuracy", "hallucination_rate", "coverage", "llm_calls"]
    avg = df.groupby(["design", "method", "tau", "rho"])[cols].mean().reset_index()
    best = avg["phi1"].max()
    cand = avg[avg["phi1"] >= best - 0.01].copy()
    cand["dist"] = (cand["tau"] - 0.5).abs()
    choice = cand.sort_values(["coverage", "llm_calls", "dist"], ascending=[False, True, True]).iloc[0]
    print(avg.sort_values("phi1", ascending=False).to_string(index=False, float_format="%.3f"))
    print(f"\nSELECTED design={choice.design} ({choice.method}) tau={choice.tau} rho={choice.rho} "
          f"(avg Phi1={choice.phi1:.3f}, best={best:.3f})")
    write_json(out / "selected.json", {
        "design": choice.design, "method": choice.method, "tau": float(choice.tau), "rho": float(choice.rho),
        "avg_phi1": float(choice.phi1), "best_phi1": float(best),
        "rule": "max mean Phi1 over dev; tie (<=0.01) -> higher coverage, fewer LLM calls, tau near 0.5"})


if __name__ == "__main__":
    main()
