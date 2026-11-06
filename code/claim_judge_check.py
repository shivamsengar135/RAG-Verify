"""Reviewer 2 (camera-ready): how far can the MiniCheck claim-precision scores be trusted without human labels?
Two automatic checks (no human validation was done):
  1. Known labels: MiniCheck-Flan-T5-Large, scored exactly as in code/evaluate.py, on HaluEval-QA (2,000 items, gold
     knowledge as evidence, the question-grounded answer claim of code/halueval_detect.py): accuracy against the
     labels, the share of correct answers accepted and of hallucinated answers rejected, and AUROC with a 95%
     bootstrap CI over questions (1,000 resamples, seed 0).
  2. A second judge on the evaluated outputs: every claim behind the claim-precision numbers of standard RAG,
     RAG + self-revision and RAG-Verify (first test block) is re-judged by the NLI model (DeBERTa-v3,
     P(entailment) >= 0.5) against the same gold evidence; agreement with MiniCheck, Cohen's kappa, and claim
     precision under both judges.

Usage: .venv\\Scripts\\python.exe code\\claim_judge_check.py
Output: results/claim_judge_check/summary.json
"""
import collections
import json

import numpy as np
import torch

from common import ROOT, load_config, read_jsonl, unload_all_ollama, write_json
from halueval_detect import auroc
from verifiers import NLIVerifier

CFG = load_config()
OUT = ROOT / "results" / "claim_judge_check"
SYSTEMS = ["rag", "rv_no_verifier", "ragverify"]


@torch.no_grad()
def minicheck(tok, model, docs, claims, bs=8):
    """P(supported) per (doc, claim) pair; same input format and label tokens as evaluate.ClaimJudge.minicheck."""
    out = []
    for s in range(0, len(docs), bs):
        texts = ["predict: " + tok.eos_token.join([d, c]) for d, c in zip(docs[s:s + bs], claims[s:s + bs])]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        dec = torch.zeros((len(texts), 1), dtype=torch.long, device="cuda")
        logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :]
        out += torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1].cpu().tolist()
    return out


def halueval_check(tok, model):
    items = read_jsonl(ROOT / "data" / "halueval" / "items.jsonl")
    claims = [f'The answer to the question "{x["question"]}" is {x["answer"].strip().rstrip(".")}.' for x in items]
    p = np.array(minicheck(tok, model, [x["knowledge"] for x in items], claims))
    hall = np.array([x["label"] == "hallucinated" for x in items])
    sup = p >= 0.5

    def stats(idx):
        return {"accuracy": float((sup[idx] != hall[idx]).mean()), "auroc": float(auroc(1 - p[idx], hall[idx]))}

    qid = np.array([x["id"].rsplit("_", 1)[0] for x in items])
    groups = {q: np.flatnonzero(qid == q) for q in np.unique(qid)}
    keys, rng = list(groups), np.random.default_rng(0)
    boots = [stats(np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])) for _ in range(1000)]
    res = {"n_items": len(items), **stats(np.arange(len(items))),
           "accepted_correct": float(sup[~hall].mean()), "rejected_hallucinated": float((~sup[hall]).mean())}
    for k in ("accuracy", "auroc"):
        res[f"{k}_ci"] = [float(v) for v in np.percentile([b[k] for b in boots], [2.5, 97.5])]
    return res


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def second_judge(nli):
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer(CFG["retriever"]["model"], device="cuda")
    res = {}
    for ds in ["hotpotqa", "triviaqa"]:
        questions = {q["id"]: q for q in read_jsonl(ROOT / "data" / ds / "questions.jsonl")}
        pages = collections.defaultdict(list)
        if ds == "triviaqa":
            for p in read_jsonl(ROOT / "data" / ds / "corpus.jsonl"):
                pages[p["title"]].append(p["text"])
        chunk_vecs = {}

        def gold_docs(qid, claim):  # as evaluate.ClaimJudge.gold_docs; chunk embeddings cached per question
            q = questions[qid]
            if ds == "hotpotqa":
                return ["\n".join(q["gold_evidence"])]
            chunks = [c for t in q["gold_titles"] for c in pages.get(t, [])]
            if not chunks:
                return []
            if qid not in chunk_vecs:
                chunk_vecs[qid] = emb.encode(chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            cv = emb.encode([claim], normalize_embeddings=True, show_progress_bar=False)
            top = np.argsort(-(chunk_vecs[qid] @ cv[0]))[:CFG["evaluation"]["triviaqa_gold_chunks_per_claim"]]
            return [chunks[i] for i in top]

        mc_all, nli_all, per_sys = [], [], {}
        for m in SYSTEMS:
            rows = json.loads((ROOT / "results/metrics" / ds / m / "seed0.json").read_text(encoding="utf-8"))["per_example"]
            prec_mc, prec_nli = [], []
            for r in rows:
                if not r.get("claims"):
                    continue
                mc = [c["supported"] for c in r["claims"]]
                nl = []
                for c in r["claims"]:
                    docs = gold_docs(r["id"], c["claim"])
                    nl.append(bool(docs) and nli.score([c["claim"]], docs)[0]["ent"] >= 0.5)
                prec_mc.append(np.mean(mc))
                prec_nli.append(np.mean(nl))
                mc_all += mc
                nli_all += nl
            per_sys[m] = {"answers": len(prec_mc), "claim_precision_minicheck": float(np.mean(prec_mc)),
                          "claim_precision_nli": float(np.mean(prec_nli))}
            print(ds, m, per_sys[m], flush=True)
        res[ds] = {"claims": len(mc_all), "agreement": float((np.array(mc_all) == np.array(nli_all)).mean()),
                   "kappa": kappa(mc_all, nli_all), "systems": per_sys}
    return res


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    unload_all_ollama()
    OUT.mkdir(parents=True, exist_ok=True)
    name = CFG["evaluation"]["minicheck_model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    summary = {"halueval_minicheck": halueval_check(tok, model)}
    print(summary, flush=True)
    del model
    torch.cuda.empty_cache()
    summary["second_judge_nli"] = second_judge(NLIVerifier(CFG))
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
