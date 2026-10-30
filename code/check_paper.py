"""Phase-6 style checks on a paper version: every number quoted in the prose is recomputed from results/,
every label is referenced, every citation key is in the verified bibliography, plus the venue's format rules
(paper/ = IEEE conference version: abstract 150-250 words, at most 6 pages; paper_ictcs/ = ICTCS 2026, Springer:
abstract at most 250 words with no citations or first person, 4-6 keywords, main text >= 3500 words or >= 10 pages).

Usage: .venv\\Scripts\\python.exe code\\check_paper.py [--paper paper_ictcs]      (exit code 1 if any check fails)
"""
import argparse
import csv
import json
import re
import sys

from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--paper", default="paper", choices=["paper", "paper_ictcs"])
PAPER = ROOT / ap.parse_args().paper
ICTCS = PAPER.name == "paper_ictcs"
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
    expect(pct(ds, "selfrag", "hallucination_rate") + "\\%", f"{ds} selfrag hall")
    expect(pct(ds, "selfrag", "abstention_rate") + "\\%", f"{ds} selfrag abst")
    for m in ["ragverify_llama", "rv_no_verifier_llama", "selfrag"]:
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
    for m in ["closed_book", "rag", "rag_abstain", "rv_no_verifier", "cove_rag", "ragverify"]:
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
elif pages > 6:
    fails.append(f"{pages} pages")
for bad in ["undefined", "Overfull"]:
    if bad in log:
        fails.append(f"LaTeX log contains '{bad}'")
pending = len(re.findall(r"\\todo\{|\\pending\{\}", tex)) + (PAPER / "tables/main_rows.tex").read_text().count("\\pending")

print(f"{PAPER.name}: abstract words: {words} | pages: {pages}" + (f" | main-text words: {body_words}" if ICTCS else "")
      + f" | citations checked | PENDING markers: {pending}")
print("FAILED:\n  " + "\n  ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
