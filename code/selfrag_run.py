"""Self-RAG baseline (Asai et al., 2023) with the official selfrag/selfrag_llama2_7b checkpoint.

Re-implements the short-form inference of the authors' retrieval_lm/run_short_form.py
(adaptive retrieval, threshold 0.2, per-passage generation scored by relevance + groundness + utility
+ exp(mean token log-prob), weights 1/1/1) with HF transformers instead of vLLM. Differences from the
original, reported in the paper: 4-bit NF4 weights (8 GB GPU), our retriever/corpus, k=5 passages.

Usage: .venv\\Scripts\\python.exe code\\selfrag_run.py --dataset hotpotqa --split seed0 [--n 20 --tag smoke]
"""
import argparse
import logging
import math
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import ROOT, CostMeter, append_jsonl, load_config, read_jsonl, set_seed, thermal_pause, unload_all_ollama
from methods import is_idk
from retriever import Retriever
from run import load_examples, split_seed

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
RET_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
REL_TOKENS = ["[Irrelevant]", "[Relevant]"]
GROUND_TOKENS = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
UTIL_TOKENS = [f"[Utility:{i}]" for i in range(1, 6)]
UTIL_WEIGHTS = [-1.0, -0.5, 0.0, 0.5, 1.0]
CONTROL_RE = re.compile(r"\[(No Retrieval|Retrieval|Continue to Use Evidence|Irrelevant|Relevant|Fully supported|"
                        r"Partially supported|No support / Contradictory|Utility:\d)\]|</?paragraph>|</s>")


def load_model(cfg):
    # Official main-branch files, SHA-256 verified by code/fetch_selfrag.py.
    name = ROOT / "data" / "models" / "selfrag_llama2_7b_official"
    local = ROOT / "data" / "models" / "selfrag_llama2_7b_nf4"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    if local.exists():
        model = AutoModelForCausalLM.from_pretrained(local, device_map="cuda:0")
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map="cuda:0",
                                                     dtype=torch.float16, use_safetensors=False)
        model.save_pretrained(local)  # cache the quantised weights so later runs load quickly
    return tok, model.eval()


class SelfRAG:
    def __init__(self, cfg, tok, model):
        self.c, self.tok, self.model = cfg["selfrag"], tok, model
        ids = lambda names: [tok.convert_tokens_to_ids(t) for t in names]  # noqa: E731
        self.ret_ids, self.rel_ids = ids(RET_TOKENS), ids(REL_TOKENS)
        self.ground_ids, self.util_ids = ids(GROUND_TOKENS), ids(UTIL_TOKENS)
        assert self.tok.unk_token_id not in self.ret_ids + self.rel_ids + self.ground_ids + self.util_ids

    @torch.no_grad()
    def generate(self, prompts, meter):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.c["max_new_tokens"], do_sample=False,
                                  output_scores=True, return_dict_in_generate=True,
                                  pad_token_id=self.tok.pad_token_id)
        logprobs = torch.stack(out.scores, dim=1).float().log_softmax(-1)  # [B, T, V]
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        meter.calls += 1
        meter.prompt_tokens += int(enc["attention_mask"].sum())
        results = []
        for b in range(gen.shape[0]):
            toks = gen[b].tolist()
            n = toks.index(self.tok.eos_token_id) + 1 if self.tok.eos_token_id in toks else len(toks)
            meter.gen_tokens += n
            results.append({"tokens": toks[:n], "logprobs": logprobs[b, :n],
                            "text": self.tok.decode(toks[:n], skip_special_tokens=False)})
        return results

    @staticmethod
    def _norm(lp, ids):
        p = [math.exp(float(lp[i])) for i in ids]
        s = sum(p)
        return [x / s for x in p] if s > 0 else [0.0] * len(p)

    def score(self, r):
        lp, toks = r["logprobs"], r["tokens"]
        seq = sum(float(lp[t, tok]) for t, tok in enumerate(toks)) / max(len(toks), 1)
        rel = self._norm(lp[0], self.rel_ids)[1]
        ground = util = 0.0
        g = next((t for t, tok in enumerate(toks) if tok in self.ground_ids), None)
        if g is not None:
            p = self._norm(lp[g], self.ground_ids)
            ground = p[0] + 0.5 * p[1]
        u = next((t for t, tok in enumerate(toks) if tok in self.util_ids), None)
        if u is not None:
            util = sum(w * p for w, p in zip(UTIL_WEIGHTS, self._norm(lp[u], self.util_ids)))
        final = self.c["w_rel"] * rel + self.c["w_sup"] * ground + self.c["w_use"] * util
        if self.c["use_seqscore"]:
            final += math.exp(seq)
        return {"final": final, "rel": rel, "ground": ground, "util": util, "seq": seq}

    @torch.no_grad()
    def retrieval_prob(self, prompt):
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        lp = self.model(**enc).logits[0, -1].float().log_softmax(-1)
        p_no, p_ret = math.exp(float(lp[self.ret_ids[0]])), math.exp(float(lp[self.ret_ids[1]]))
        return p_ret / (p_ret + p_no)

    def answer(self, question, retriever, k, meter):
        prompt = PROMPT.format(instruction=question)
        p_ret = self.retrieval_prob(prompt)
        meter.calls += 1
        trace = {"p_retrieval": round(p_ret, 4)}
        if p_ret <= self.c["threshold"]:
            r = self.generate([prompt + "[No Retrieval]"], meter)[0]
            trace["retrieved"] = False
            return r["text"], trace
        passages = retriever.search(question, k)
        prompts = [prompt + "[Retrieval]<paragraph>" + p["title"] + "\n" + p["text"][len(p["title"]) + 2:]
                   + "</paragraph>" for p in passages]
        results = self.generate(prompts, meter)
        scored = [self.score(r) for r in results]
        best = max(range(len(results)), key=lambda i: scored[i]["final"])
        trace.update(retrieved=True, passages=[p["pid"] for p in passages], best=best,
                     scores=[{k2: round(v, 4) for k2, v in s.items()} for s in scored])
        return results[best]["text"], trace


def clean(text):
    return re.sub(r"\s+", " ", CONTROL_RE.sub(" ", text)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "triviaqa"])
    ap.add_argument("--split", default="seed0")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config()
    seed = split_seed(args.split)
    set_seed(seed)
    name = f"{args.split}{'_' + args.tag if args.tag else ''}"
    out = ROOT / "results" / "raw" / args.dataset / "selfrag" / f"{name}.jsonl"
    logging.basicConfig(filename=ROOT / "logs" / f"run_{args.dataset}_selfrag_{name}.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()

    examples = load_examples(args.dataset, args.split, args.n)
    done = {r["id"] for r in read_jsonl(out)}
    todo = [e for e in examples if e["id"] not in done]
    print(f"selfrag | {args.dataset} | {name}: {len(todo)}/{len(examples)} to run")
    if not todo:
        return
    unload_all_ollama()  # free VRAM held by Ollama
    tok, model = load_model(cfg)
    srag = SelfRAG(cfg, tok, model)
    retriever = Retriever(args.dataset, cfg)
    torch.cuda.reset_peak_memory_stats()

    for i, ex in enumerate(todo, 1):
        thermal_pause(log)  # outside the timed region
        meter, t0 = CostMeter(), time.perf_counter()
        try:
            raw, trace = srag.answer(ex["question"], retriever, cfg["retriever"]["k"], meter)
            pred, err = clean(raw), None
        except Exception as e:
            raw, trace, pred, err = "", {}, "", repr(e)
            log.exception("example %s failed", ex["id"])
        meter.llm_seconds = time.perf_counter() - t0
        abst = is_idk(pred)
        rec = {"id": ex["id"], "question": ex["question"], "answers": ex["answers"], "method": "selfrag",
               "dataset": args.dataset, "split": args.split, "backbone": cfg["selfrag"]["model"] + " (nf4)",
               "output_text": pred, "raw_output": raw, "prediction": "" if abst else pred, "abstained": abst,
               "trace": trace, "error": err, "latency_s": round(time.perf_counter() - t0, 3), **meter.as_dict()}
        append_jsonl(out, rec)
        log.info("[%d/%d] %s pred=%r %.1fs", i, len(todo), ex["id"], pred[:60], rec["latency_s"])
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} last={rec['latency_s']:.1f}s")
    log.info("peak_torch_vram_mib=%d", torch.cuda.max_memory_allocated() // 2**20)


if __name__ == "__main__":
    main()
