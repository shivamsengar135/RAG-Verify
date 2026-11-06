"""Phase-6 style checks on a paper version: every number quoted in the prose is recomputed from results/,
every label is referenced, every citation key is in the verified bibliography, plus the venue's format rules
(paper/ = IEEE conference version: abstract 150-250 words, at most 6 pages; paper_ictcs/ = ICTCS 2026, Springer:
abstract at most 250 words with no citations or first person, 4-6 keywords, main text >= 3500 words or >= 10 pages,
at most 10 pages to avoid the over-length charge).

Usage: .venv\\Scripts\\python.exe code\\check_paper.py [--paper paper_ictcs]      (exit code 1 if any check fails)
"""
import argparse
import csv
import json
import re
import sys

from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--paper", default="paper", choices=["paper", "paper_ictcs", "paper_ictcs_crc"])
PAPER = ROOT / ap.parse_args().paper
ICTCS = PAPER.name.startswith("paper_ictcs")
CRC = PAPER.name == "paper_ictcs_crc"  # camera-ready copy: adds the analyses asked for by the reviewers
tex = (PAPER / "main.tex").read_text(encoding="utf-8")
S = {(r["dataset"], r["method"]): r for r in csv.DictReader((ROOT / "results/tables/main_summary.csv").open(encoding="utf-8"))}
T = {(r["dataset"], r["a"], r["b"]): r for r in csv.DictReader((ROOT / "results/tables/main_tests.csv").open(encoding="utf-8"))}
B = json.loads((ROOT / "results/tables/abstention_breakdown.json").read_text(encoding="utf-8"))
SEL = json.loads((ROOT / "results/tuning/selected.json").read_text(encoding="utf-8"))
H, Q = "hotpotqa", "triviaqa"
fails = []


def pct(ds, m, k):
    return f"{100 * float(S[(ds, m)][k + '_mean']):.1f}"


def delta(ds, a, b):
    return f"{100 * float(T[(ds, a, b)]['delta_phi1']):.1f}"


def expect(text, why):
    if text not in tex and text.replace("artefacts", "artifacts") not in tex:  # British (IEEE) vs American (Springer)
        fails.append(f"missing '{text}' ({why})")


# prose numbers -> source values
for m, ks in {"rag": ["hallucination_rate", "accuracy"], "ragverify": ["hallucination_rate", "accuracy", "abstention_rate"],
              "closed_book": ["accuracy"], "cove": ["accuracy"]}.items():
    for k in ks:
        for ds in (H, Q):
            expect(pct(ds, m, k) + "\\%", f"{ds} {m} {k}")
for ds in (H, Q):
    expect(f"{100 * float(S[(ds, 'ragverify')]['selective_accuracy_mean']):.1f}\\%", f"{ds} RV selective acc")
    expect(f"{100 * float(S[(ds, 'rag')]['selective_accuracy_mean']):.1f}\\%", f"{ds} RAG selective acc")
    for m in ["rag_abstain", "rv_no_verifier", "cove_rag", "rag"]:
        expect(pct(ds, m, "phi1").replace("-", "$-$"), f"{ds} {m} phi1")
for ds, a, b in [(H, "ragverify", "rag"), (Q, "ragverify", "rag"), (H, "ragverify", "rv_no_verifier"),
                 (Q, "ragverify", "rv_no_verifier"), (H, "rv_no_verifier", "rag_abstain"), (Q, "rv_no_verifier", "rag_abstain")]:
    expect(delta(ds, a, b).lstrip("-"), f"{ds} {a}-{b} delta")
expect(pct(Q, "closed_book_llama", "accuracy") + "\\%", "llama closed-book TQ")
expect(pct(Q, "rag_llama", "accuracy") + "\\%", "llama RAG TQ")
for ds in (H, Q):
    expect(pct(ds, "rag_abstain_llama", "abstention_rate") + "\\%", f"{ds} llama abstain rate")
    expect(pct(ds, "rag_abstain", "abstention_rate") + "\\%", f"{ds} qwen abstain rate")
    expect(f"{float(S[(ds, 'ragverify')]['llm_calls_mean']):.1f}", f"{ds} RV calls")
for ds in (H, Q):
    expect(pct(ds, "ragverify_llama", "hallucination_rate") + "\\%", f"{ds} llama RV hall")
    if CRC:  # the camera-ready text only says Self-RAG abstains least and hallucinates most among retrieval-based systems
        retr = [r for (d, m), r in S.items() if d == ds and r.get("status") == "done"
                and not m.startswith("closed_book") and m != "cove"]
        if min(retr, key=lambda r: float(r["abstention_rate_mean"]))["method"] != "selfrag" or \
                max(retr, key=lambda r: float(r["hallucination_rate_mean"]))["method"] != "selfrag":
            fails.append(f"{ds}: Self-RAG is not the lowest-abstention, highest-hallucination retrieval system")
    else:
        expect(pct(ds, "selfrag", "hallucination_rate") + "\\%", f"{ds} selfrag hall")
        expect(pct(ds, "selfrag", "abstention_rate") + "\\%", f"{ds} selfrag abst")
    for m in ["ragverify_llama", "rv_no_verifier_llama"] + ([] if CRC else ["selfrag"]):
        expect(pct(ds, m, "phi1").replace("-", "$-$"), f"{ds} {m} phi1")
    expect(delta(ds, "ragverify_llama", "rag_llama"), f"{ds} llama RV-RAG delta")
    expect(f"{float(S[(ds, 'ragverify_llama')]['llm_calls_mean']):.1f} ", f"{ds} llama RV calls")
    for a, b in [("ragverify_llama", "rv_no_verifier_llama"), ("ragverify_llama", "rag_abstain_llama")]:
        t = T[(ds, a, b)]
        if float(t["ci_lo"]) > 0 or float(t["ci_hi"]) < 0:
            fails.append(f"{ds} {a}-{b} is significant but the prose says tied")
for ds, m in [(H, "rv_no_verifier"), (Q, "rv_no_verifier")]:
    if 2 / float(S[(ds, "ragverify")]["llm_calls_mean"]) > 0.6:
        fails.append("'no more than 60% of its model calls' is false")
# ablations (single block) and error analysis
A = {(r["dataset"], r["method"]): r for r in csv.DictReader((ROOT / "results/tables/ablations.csv").open(encoding="utf-8"))}
for ds, m in [(H, "rv_no_retrieval"), (Q, "rv_no_retrieval"), (H, "rv_no_verifier")]:
    expect(f"{100 * abs(float(A[(ds, m)]['delta_phi1'])):.1f}", f"{ds} {m} ablation delta")
for ds in (H, Q):
    expect(f"{100 * float(A[(ds, 'rv_no_abstain')]['hallucination_rate']):.1f}\\%", f"{ds} no-gate hall")
    expect(f"{100 * float(A[(ds, 'rag')]['hallucination_rate']):.1f}\\%", f"{ds} seed0 RAG hall")
    for m in ["rv_no_decomp", "rv_no_revision"]:
        if abs(float(A[(ds, m)]["delta_phi1"])) > 0.01:
            fails.append(f"{ds} {m}: 'at most one point' is false")
E = json.loads((ROOT / "results/error_analysis/summary.json").read_text(encoding="utf-8"))
M = json.loads((ROOT / "results/error_analysis/manual_summary.json").read_text(encoding="utf-8"))
expect(f"there are {E[H]['lost answer']} on HotpotQA and {E[Q]['lost answer']} on TriviaQA", "lost answers")
expect(f"in {E[H]['lost answer: answer-claim flag']} and {E[Q]['lost answer: answer-claim flag']} of them", "judge lost")
if not CRC:  # the camera-ready copy splits wrong answers per first draft instead (verdict breakdown below)
    expect(f"Of the {E[H]['wrong answer']} and {E[Q]['wrong answer']} wrong answers, {E[H]['wrong answer: retrieval miss']} "
           f"and {E[Q]['wrong answer: retrieval miss']}", "wrong / retrieval miss")
    expect(f"in {E[H]['wrong answer: verifier accept']} and {E[Q]['wrong answer: verifier accept']} the judge", "verifier accept")
for k, phrase in [("lost answer: judge rejects explicitly supported answer", "{} were rejected although"),
                  ("lost answer: comparison / yes-no needs combining passages", "{} were comparison"),
                  ("lost answer: draft not grounded in passages (prior knowledge)", "in {} the draft"),
                  ("lost answer: scoring artefact (noisy gold alias, draft wrong)", "{} were artefacts"),
                  ("wrong answer: answers a different slot (judge accepts)", "{} named a related"),
                  ("wrong answer: reasoning or reading error over evidence", "{} were reasoning"),
                  ("wrong answer: retrieval miss (answer from wrong passage)", "{} followed a retrieval"),
                  ("wrong answer: scoring artefact (acceptable answer)", "{} were acceptable")]:
    if not ICTCS or k.endswith(("explicitly supported answer", "different slot (judge accepts)")):  # 10-page version quotes two
        expect(phrase.format(M[k]), k)
HE = json.loads((ROOT / "results/halueval/summary.json").read_text(encoding="utf-8"))
HR = [json.loads(l) for l in (ROOT / "results/halueval/scores.jsonl").open(encoding="utf-8")]
expect(f"{HE['nli']['auroc']:.3f} (95\\% CI {HE['nli']['auroc_ci'][0]:.3f}--{HE['nli']['auroc_ci'][1]:.3f})", "HaluEval NLI AUROC")
expect(f"flags only {sum(r['label'] == 'hallucinated' and r['nli_ent'] < HE['tau'] for r in HR)} of the 1{{,}}000 hallucinated", "NLI recall")
expect(f"with {sum(r['label'] == 'correct' and r['nli_ent'] < HE['tau'] for r in HR)} false alarms", "NLI FP")
expect(f"accuracy {HE['llm_judge']['accuracy']:.3f}, F1 {HE['llm_judge']['f1']:.3f} vs.\\ {HE['nli']['f1']:.3f}", "judge acc/F1")
expect(f"flags {sum(r['label'] == 'correct' and not r['judge_supported'] for r in HR)} of the 1{{,}}000 correct", "judge FP")
LC = {(r["dataset"], r["method"]): r for r in csv.DictReader((ROOT / "results/tables/latency_claims.csv").open(encoding="utf-8"))}
for ds in (H, Q):
    for m in (["rag", "rv_no_verifier", "ragverify"] if ICTCS else ["closed_book", "rag", "rag_abstain", "rv_no_verifier", "cove_rag", "ragverify"]):
        expect(f"{100 * float(LC[(ds, m)]['claim_precision']):.1f}\\%", f"{ds} {m} claim precision")
    for m in ["rag", "rv_no_verifier", "cove_rag", "ragverify"]:
        expect(f"{float(LC[(ds, m)]['latency_median']):.1f}\\,s", f"{ds} {m} median latency")
b = B[H]
expect(f"{100 * b['flagged_initial_correct'] / b['initial_correct']:.1f}\\%", "HP correct drafts flagged")
expect(f"{100 * B[Q]['flagged_initial_correct'] / B[Q]['initial_correct']:.1f}\\%", "TQ correct drafts flagged")
for ds in (H, Q):
    c = B[ds]
    expect(f"{c['idk_after_revision']} of {c['idk_by_model'] + c['gate_abstain']}", f"{ds} refusals of abstentions")
    expect(f"{c['idk_after_revision']} of {c['flagged_initial']}", f"{ds} refusals of flagged")
    expect(f"{round(100 * c['idk_after_revision'] / c['flagged_initial'])}\\%", f"{ds} % flagged -> refusal")
expect(f"{B[H]['gate_withheld_correct']} of {B[H]['gate_abstain']}", "gate withheld correct")
expect(f"{SEL.get('avg_phi1', SEL.get('mean_phi1', 0.2)):.3f}", "dev selected phi1")


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


if CRC:  # reviewer analyses (camera-ready)
    V = json.loads((ROOT / "results/error_analysis/verdicts.json").read_text(encoding="utf-8"))
    d, c, w = ({ds: V[ds][k] for ds in (H, Q)} for k in ("drafts", "correct", "passed_wrong"))
    expect(f"contradicting evidence for {d[H]['contradicted']} and {d[Q]['contradicted']} drafts", "verdict contradicted")
    expect(f"no evidence for {d[H]['no evidence']} and {d[Q]['no evidence']}", "verdict no evidence")
    expect(f"yet {c[H]['contradicted'] + c[H]['no evidence']} and {c[Q]['contradicted'] + c[Q]['no evidence']} of these",
           "answer-check false alarms")
    det = {ds: d[ds]["detail contradicted"] + d[ds]["partially supported"] for ds in (H, Q)}
    expect(f"({det[H]} and {det[Q]} drafts)", "detail flags")
    expect(f"({c[Q]['detail contradicted'] + c[Q]['partially supported']} of {det[Q]})", "detail false alarms TQ")
    expect(f"{w[H]['missing evidence']} and {w[Q]['missing evidence']} had no gold answer", "passed wrong: missing evidence")
    expect(f"{w[H]['wrong entity']} and {w[Q]['wrong entity']} gave another entity or value", "passed wrong: entity")
    expect(f"{w[H]['near miss']} and {w[Q]['near miss']} were near misses", "passed wrong: near miss")
    for ds in (H, Q):
        if V[ds]["flagged_correct"] != B[ds]["flagged_initial_correct"]:
            fails.append(f"{ds}: verdict breakdown disagrees with abstention breakdown")
    cl = {ds: V[ds]["claims"] for ds in (H, Q)}
    expect(f"median of {cl[H]['median']:.0f} (HotpotQA) and {cl[Q]['median']:.0f} (TriviaQA)", "claims per decomposition")
    expect(f"{cl[H]['at_cap'] + cl[Q]['at_cap']} of {thousands(cl[H]['decompositions'] + cl[Q]['decompositions'])} reached",
           "12-claim cap")
    PR = json.loads((ROOT / "results/profile/summary.json").read_text(encoding="utf-8"))
    st, mem = {ds: PR["datasets"][ds]["stages"] for ds in (H, Q)}, PR["memory"]
    expect(f"Ollama release ({PR['info']['ollama_version']}; mean {PR['datasets'][H]['total_mean_s']:.1f}\\,s and "
           f"{PR['datasets'][Q]['total_mean_s']:.1f}\\,s per question)", "profile totals")
    for stage, phrase in [("answer", "drafting takes"), ("decomposition", "decomposition"), ("answer_check", "the answer check"),
                          ("nli", "NLI"), ("revision", "revision")]:
        expect(f"{phrase} {st[H][stage]['mean_s_per_question']:.2f} and {st[Q][stage]['mean_s_per_question']:.2f}\\,s",
               f"profile {stage}")
    expect(f"({st[H]['revision']['median_s_when_run']:.1f} and {st[Q]['revision']['median_s_when_run']:.1f}\\,s when triggered)",
           "revision when triggered")
    if {f"{st[ds]['retrieval']['mean_s_per_question']:.2f}" for ds in (H, Q)} != {"0.02"}:
        fails.append("retrieval time is not 0.02 s on both datasets")
    expect(f"resident generator ({mem['ollama_size_vram_mib'] / 1024:.1f}\\,GB", "generator memory")
    expect(f"adds {mem['nli_weights_mib'] / 1024:.1f}\\,GB of weights", "NLI weights")
    expect(f"peak at {max(st[ds]['nli']['torch_peak_mib'] for ds in (H, Q)) / 1024:.1f}\\,GB during verification", "NLI peak")
    expect(f"device peak was {mem['device_peak_mib'] / 1024:.1f} of 8\\,GB", "device peak")
    J = json.loads((ROOT / "results/judge_conf/summary.json").read_text(encoding="utf-8"))
    ag = {ds: 100 * J["agreement"][ds]["same_verdict"] / J["agreement"][ds]["n"] for ds in (H, Q)}
    expect(f"({ag[H]:.0f}\\% and {ag[Q]:.0f}\\% of verdicts reproduced)", "judge re-run agreement")
    expect(f"AUROC {J['auroc'][H]:.2f} and {J['auroc'][Q]:.2f} for draft correctness", "confidence AUROC")
    g, rv = J["gate"], J["revise"]
    expect(f"confident drafts gives $\\Phi_1$ of {100 * g[H]['phi1_mean']:.1f} and {100 * g[Q]['phi1_mean']:.1f}, level with",
           "gate phi1")
    for ds in (H, Q):
        if g[ds]["vs_ragverify"]["ci_lo"] > 0 or g[ds]["vs_ragverify"]["ci_hi"] < 0:
            fails.append(f"{ds}: calibrated gate differs significantly from RAG-Verify, but the text says level")
    expect(f"self-revision gives {100 * rv[H]['phi1_mean']:.1f} and {100 * rv[Q]['phi1_mean']:.1f}", "revise phi1")
    lb = J["lost_correct_drafts_baselines"]
    expect(f"({rv[H]['lost_correct_drafts']} and {rv[Q]['lost_correct_drafts']}, against {lb[H]['ragverify']} and "
           f"{lb[Q]['ragverify']} for RAG-Verify)", "lost drafts")
    if not all(rv[ds]["hallucination"] > g[ds]["hallucination"] for ds in (H, Q)):
        fails.append("'lets more wrong ones through' is not supported")

    def sgn(x):
        return f"{100 * x:+.1f}".replace("-", "$-$").replace("+", "$+$")

    def ci(t):
        return f"{sgn(t['delta'])} [{sgn(t['ci_lo'])}, {sgn(t['ci_hi'])}]"

    expect(f"($\\Delta$ {ci(rv[H]['vs_rv_no_verifier'])} and {ci(rv[Q]['vs_rv_no_verifier'])})", "revise vs self-revision")
    for ds in (H, Q):
        if rv[ds]["vs_rv_no_verifier"]["ci_hi"] > 0 and rv[ds]["vs_rv_no_verifier"]["ci_lo"] > 0:
            fails.append(f"{ds}: confidence-guided revision beats self-revision, but the text says it does not")
    K = json.loads((ROOT / "results/claim_judge_check/summary.json").read_text(encoding="utf-8"))
    hm, sj = K["halueval_minicheck"], K["second_judge_nli"]
    expect(f"MiniCheck is right on {100 * hm['accuracy']:.1f}\\% of answers (AUROC {hm['auroc']:.3f})", "MiniCheck on HaluEval")
    expect(f"agrees with it on {100 * sj[H]['agreement']:.1f}\\% and {100 * sj[Q]['agreement']:.1f}\\% (Cohen's $\\kappa$ "
           f"{sj[H]['kappa']:.2f} and {sj[Q]['kappa']:.2f})", "second judge agreement")
    for ds in (H, Q):
        sy = sj[ds]["systems"]
        if sorted(sy, key=lambda m: sy[m]["claim_precision_minicheck"]) != sorted(sy, key=lambda m: sy[m]["claim_precision_nli"]):
            fails.append(f"{ds}: NLI judge does not rank the systems in the same order")
        if any(f"{100 * sy[m]['claim_precision_minicheck']:.1f}" != f"{100 * float(LC[(ds, m)]['claim_precision']):.1f}" for m in sy):
            fails.append(f"{ds}: re-read MiniCheck precision differs from the reported one")

# structure
for lab in re.findall(r"\\label\{([^}]+)\}", tex):
    if f"ref{{{lab}}}" not in tex:
        fails.append(f"label never referenced: {lab}")
bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
for keys in re.findall(r"\\cite\{([^}]+)\}", tex):
    for k in keys.split(","):
        if f"{{{k.strip()},".replace("{", "{", 1) not in bib:
            fails.append(f"citation not in verified bib: {k}")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
kw = re.search(r"\\keywords\{(.*)\}\s*$", abstract.strip(), re.S)
abstract_text = abstract.strip()[:kw.start()] if kw else abstract
words = len(re.sub(r"\\[a-z]+|[{}$\\]", " ", abstract_text).split())
if not 150 <= words <= 250:
    fails.append(f"abstract has {words} words")
log = (PAPER / "main.log").read_text(encoding="latin-1")
pages = int(re.search(r"Output written on main\.pdf \((\d+) pages", log).group(1))
body_words = None
if ICTCS:
    if "\\cite" in abstract_text or re.search(r"\b(we|our|us)\b", abstract_text, re.I):
        fails.append("abstract has a citation or first-person wording")
    if not kw or not 4 <= len(kw.group(1).split("$\\cdot$")) <= 6:
        fails.append("need 4-6 keywords")
    body = tex[tex.index("\\section{Introduction}"):tex.index("\\bibliographystyle")]
    body = re.sub(r"\\begin\{(table|algorithm)\}.*?\\end\{\1\}|(?<!\\)%[^\n]*|\$[^$]*\$|\\(cite|ref|label|url)\{[^}]*\}", " ",
                  body, flags=re.S)
    body_words = len([w for w in re.sub(r"\\[a-zA-Z]+\*?|[{}\[\]]", " ", body).split() if re.search(r"[A-Za-z0-9]", w)])
    if body_words < 3500 and pages < 10:
        fails.append(f"main text has {body_words} words and {pages} pages (ICTCS needs >= 3500 words or >= 10 pages)")
    if pages > 10:
        fails.append(f"{pages} pages (ICTCS charges INR 1800 per page beyond 10)")
elif pages > 6:
    fails.append(f"{pages} pages")
for bad in ["undefined", "Overfull"]:
    if bad in log:
        fails.append(f"LaTeX log contains '{bad}'")
pending = len(re.findall(r"\\todo\{|\\pending\{\}", tex)) + (PAPER / "tables/main_rows.tex").read_text().count("\\pending")
if CRC:  # the organisers want the final PDF without page numbers, headers or footers, and nothing may be pending
    if not re.search(r"\\documentclass\[[^]]*norunningheads[^]]*\]\{svproc\}", tex):
        fails.append("camera-ready copy must use the norunningheads option (no headers or page numbers)")
    if pending:
        fails.append(f"{pending} PENDING markers left")

print(f"{PAPER.name}: abstract words: {words} | pages: {pages}" + (f" | main-text words: {body_words}" if ICTCS else "")
      + f" | citations checked | PENDING markers: {pending}")
print("FAILED:\n  " + "\n  ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
