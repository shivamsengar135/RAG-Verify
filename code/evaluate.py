"""Score raw outputs: answer accuracy, abstention, hallucination rate, FActScore-style claim precision, cost.

Claim precision: claims are extracted from each answered output by a FIXED extractor LLM (same for every system),
then judged by MiniCheck-Flan-T5-Large against GOLD evidence (HotpotQA supporting paragraphs; for TriviaQA the
top-5 chunks of the gold Wikipedia pages per claim, as in FActScore). The pipeline's own verifier is never used.

Usage:
  .venv\\Scripts\\python.exe code\\evaluate.py --dataset hotpotqa --method ragverify --split seed0
  .venv\\Scripts\\python.exe code\\evaluate.py --all            (every raw file without metrics)
  add --no-claims to skip claim precision (fast)
"""
import argparse
import collections
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch

import prompts as P
from common import LLM, ROOT, append_jsonl, load_config, read_jsonl, text_hash, write_json
from methods import is_idk, parse_claims, parse_response

CFG = load_config()


# ------------------------------------------------------------- answer metrics (SQuAD normalisation)
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_scores(pred, golds):
    if not pred:
        return {"em": 0.0, "f1": 0.0, "match": 0.0}
    npred = normalize(pred)
    return {
        "em": float(any(npred == normalize(g) for g in golds)),
        "f1": max(token_f1(pred, g) for g in golds),
        "match": float(any(normalize(g) and f" {normalize(g)} " in f" {npred} " for g in golds)),
    }


# ------------------------------------------------------------- claim precision
class ClaimJudge:
    def __init__(self, dataset):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        ev = CFG["evaluation"]
        self.dataset = dataset
        self.extractor = LLM(ev["claim_extractor"], CFG["llm"]["num_ctx"], 320, 0.0, seed=0)
        self.tok = AutoTokenizer.from_pretrained(ev["minicheck_model"])
        self.mc = AutoModelForSeq2SeqLM.from_pretrained(ev["minicheck_model"], dtype=torch.bfloat16).cuda().eval()
        self.claim_cache_path = ROOT / "results" / "cache" / f"claims_{dataset}.jsonl"
        self.mc_cache_path = ROOT / "results" / "cache" / f"minicheck_{dataset}.jsonl"
        self.claim_cache = {r["key"]: r["claims"] for r in read_jsonl(self.claim_cache_path)}
        self.mc_cache = {r["key"]: r["p"] for r in read_jsonl(self.mc_cache_path)}
        self.questions = {q["id"]: q for q in read_jsonl(ROOT / "data" / dataset / "questions.jsonl")}
        if dataset == "triviaqa":
            from sentence_transformers import SentenceTransformer

            self.emb = SentenceTransformer(CFG["retriever"]["model"], device="cuda")
            self.pages = collections.defaultdict(list)
            for p in read_jsonl(ROOT / "data" / dataset / "corpus.jsonl"):
                self.pages[p["title"]].append(p["text"])

    def claims(self, question, output_text):
        key = text_hash(CFG["evaluation"]["claim_extractor"], question, output_text)
        if key not in self.claim_cache:
            raw = self.extractor.chat(P.DECOMPOSE.format(question=question, response=output_text))
            self.claim_cache[key] = parse_claims(raw)
            append_jsonl(self.claim_cache_path, {"key": key, "claims": self.claim_cache[key]})
        return self.claim_cache[key]

    @torch.no_grad()
    def minicheck(self, docs, claims):
        """P(supported) for each (doc, claim) pair, cached."""
        keys = [text_hash(d, c) for d, c in zip(docs, claims)]
        todo = [i for i, k in enumerate(keys) if k not in self.mc_cache]
        for s in range(0, len(todo), 8):
            idx = todo[s:s + 8]
            texts = ["predict: " + self.tok.eos_token.join([docs[i], claims[i]]) for i in idx]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
            dec = torch.zeros((len(idx), 1), dtype=torch.long, device="cuda")
            logits = self.mc(**enc, decoder_input_ids=dec).logits[:, 0, :]
            probs = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1].cpu().tolist()
            for i, p in zip(idx, probs):
                self.mc_cache[keys[i]] = p
                append_jsonl(self.mc_cache_path, {"key": keys[i], "p": p})
        return [self.mc_cache[k] for k in keys]

    def gold_docs(self, qid, claim):
        q = self.questions[qid]
        if self.dataset == "hotpotqa":
            return ["\n".join(q["gold_evidence"])]
        chunks = [c for t in q["gold_titles"] for c in self.pages.get(t, [])]
        if not chunks:
            return []
        n = CFG["evaluation"]["triviaqa_gold_chunks_per_claim"]
        cv = self.emb.encode([claim], normalize_embeddings=True, show_progress_bar=False)
        dv = self.emb.encode(chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        top = np.argsort(-(dv @ cv[0]))[:n]
        return [chunks[i] for i in top]

    def precision(self, rec):
        claims = self.claims(rec["question"], rec["output_text"])
        verdicts = []
        for c in claims:
            docs = self.gold_docs(rec["id"], c)
            p = max(self.minicheck(docs, [c] * len(docs))) if docs else 0.0
            verdicts.append({"claim": c, "p_support": round(p, 4), "supported": p >= 0.5})
        return verdicts


# ------------------------------------------------------------- per-run evaluation
def evaluate_file(raw_path, judge=None):
    recs = read_jsonl(raw_path)
    rows = []
    for r in recs:
        # Re-apply the (shared) refusal detector so every system is judged by the same, current rule.
        if not r["abstained"] and is_idk(parse_response(r["output_text"])[0]):
            r["abstained"], r["prediction"] = True, ""
        s = answer_scores(r["prediction"], r["answers"])
        correct = bool(s["em"] or s["match"])
        row = {"id": r["id"], **s, "correct": correct, "answered": not r["abstained"],
               "hallucinated": (not r["abstained"]) and not correct, "error": r.get("error"),
               "latency_s": r["latency_s"], "llm_calls": r["llm_calls"], "gen_tokens": r["gen_tokens"],
               "prompt_tokens": r["prompt_tokens"]}
        if judge is not None and not r["abstained"] and r["output_text"]:
            v = judge.precision(r)
            row["claims"] = v
            row["claim_precision"] = (sum(x["supported"] for x in v) / len(v)) if v else None
        rows.append(row)

    n = len(rows)
    answered = [x for x in rows if x["answered"]]
    mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
    cp = [x["claim_precision"] for x in rows if x.get("claim_precision") is not None]
    summary = {
        "n": n,
        "errors": sum(x["error"] is not None for x in rows),
        "em": mean([x["em"] for x in rows]),
        "f1": mean([x["f1"] for x in rows]),
        "accuracy": mean([x["correct"] for x in rows]),
        "coverage": len(answered) / n if n else None,
        "abstention_rate": 1 - len(answered) / n if n else None,
        "selective_accuracy": mean([x["correct"] for x in answered]),
        "hallucination_rate": mean([x["hallucinated"] for x in rows]),
        "claim_precision": mean(cp),
        "claims_per_answer": mean([len(x["claims"]) for x in rows if "claims" in x]),
        "latency_s": mean([x["latency_s"] for x in rows]),
        "llm_calls": mean([x["llm_calls"] for x in rows]),
        "gen_tokens": mean([x["gen_tokens"] for x in rows]),
        "prompt_tokens": mean([x["prompt_tokens"] for x in rows]),
    }
    rel = raw_path.relative_to(ROOT / "results" / "raw")
    out = ROOT / "results" / "metrics" / rel.with_suffix(".json")
    write_json(out, {"source": str(rel).replace("\\", "/"), "summary": summary, "per_example": rows})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--split", default="seed0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-claims", action="store_true")
    args = ap.parse_args()

    if args.all:
        paths = sorted((ROOT / "results" / "raw").glob("*/*/*.jsonl"))
    else:
        name = f"{args.split}{'_' + args.tag if args.tag else ''}"
        paths = [ROOT / "results" / "raw" / args.dataset / args.method / f"{name}.jsonl"]
    judges = {}
    for p in paths:
        ds = p.parts[-3]
        judge = None
        if not args.no_claims:
            judges.setdefault(ds, ClaimJudge(ds))
            judge = judges[ds]
        s = evaluate_file(p, judge)
        cp = f"{s['claim_precision']:.3f}" if s["claim_precision"] is not None else "-"
        print(f"{ds:9s} {p.parts[-2]:18s} {p.stem:12s} n={s['n']:3d} acc={s['accuracy']:.3f} "
              f"f1={s['f1']:.3f} abst={s['abstention_rate']:.3f} hall={s['hallucination_rate']:.3f} "
              f"claimP={cp} lat={s['latency_s']:.2f}s calls={s['llm_calls']:.1f} err={s['errors']}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
