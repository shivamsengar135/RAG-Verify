"""Claim verifiers: NLI against retrieved evidence, LLM-as-judge, and SelfCheck-style sample consistency."""
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def label_from_scores(ent, con, tau):
    if ent >= tau:
        return "supported"
    if con >= tau:
        return "contradicted"
    return "unsupported"


class NLIVerifier:
    """Support(c) = max over premises of P(entailment); premises are the k passages plus one joint premise."""

    def __init__(self, cfg, device="cuda"):
        v = cfg["verifier"]
        self.tok = AutoTokenizer.from_pretrained(v["nli_model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(v["nli_model"], dtype=torch.float16)
        self.model.to(device).eval()
        self.device, self.max_len, self.joint_top = device, v["nli_max_len"], v["joint_premise_top"]
        labels = {l.lower(): i for i, l in self.model.config.id2label.items()}
        self.i_ent, self.i_con = labels["entailment"], labels["contradiction"]

    @torch.no_grad()
    def probs(self, premises, hypotheses, batch_size=16):
        out = []
        for i in range(0, len(premises), batch_size):
            enc = self.tok(premises[i:i + batch_size], hypotheses[i:i + batch_size], return_tensors="pt",
                           padding=True, truncation="only_first", max_length=self.max_len).to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, 3)

    def premises_from_passages(self, passages):
        texts = [p["text"] for p in passages]
        if self.joint_top and len(texts) > 1:
            texts.append("\n".join(texts[: self.joint_top]))
        return texts

    def score(self, claims, premises):
        """Returns per-claim dicts with entailment/contradiction scores and the best premise index."""
        if not claims or not premises:
            return [{"ent": 0.0, "con": 0.0, "best": None} for _ in claims]
        pairs_p = [p for _ in claims for p in premises]
        pairs_h = [c for c in claims for _ in premises]
        pr = self.probs(pairs_p, pairs_h).view(len(claims), len(premises), -1)
        ent, con = pr[..., self.i_ent], pr[..., self.i_con]
        return [{"ent": float(ent[i].max()), "con": float(con[i].max()), "best": int(ent[i].argmax())}
                for i in range(len(claims))]

    def score_selfcheck(self, claims, samples):
        """SelfCheckGPT-NLI style: support = mean over sampled responses of P(entail)."""
        if not claims or not samples:
            return [{"ent": 0.0, "con": 0.0, "best": None} for _ in claims]
        pairs_p = [s for _ in claims for s in samples]
        pairs_h = [c for c in claims for _ in samples]
        pr = self.probs(pairs_p, pairs_h).view(len(claims), len(samples), -1)
        return [{"ent": float(pr[i, :, self.i_ent].mean()), "con": float(pr[i, :, self.i_con].mean()),
                 "best": None} for i in range(len(claims))]


JUDGE_PROMPT = """You are checking whether a claim is supported by evidence passages.

Passages:
{passages}

Claim: {claim}

Is the claim supported by the passages? Reply with exactly one word: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO."""


ANSWER_JUDGE_PROMPT = """You are checking whether a proposed answer to a question is supported by evidence passages.

Passages:
{passages}

Question: {question}
Proposed answer: {answer}

Do the passages show that the proposed answer is a correct answer to THIS question (not to a related question)?
Reply with exactly one word: SUPPORTED, CONTRADICTED (the passages point to a different answer), or NOT_ENOUGH_INFO."""


def _verdict(text):
    m = re.search(r"SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO", text.upper())
    return m.group(0) if m else "NOT_ENOUGH_INFO"


class LLMJudgeVerifier:
    """One LLM call per claim; maps the verdict to (ent, con) in {0,1} so the same tau logic applies."""

    def __init__(self, llm):
        self.llm = llm

    def score_answer(self, question, answer, passages, meter=None):
        """Question-aware check of the answer itself (used for the answer claim in the hybrid verifier)."""
        ptxt = "\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
        raw = self.llm.chat(ANSWER_JUDGE_PROMPT.format(passages=ptxt, question=question, answer=answer),
                            meter=meter, num_predict=8)
        v = _verdict(raw)
        return {"ent": float(v == "SUPPORTED"), "con": float(v == "CONTRADICTED"), "best": None, "raw": raw,
                "judge": "answer"}

    def score(self, claims, passages, meter=None):
        ptxt = "\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
        out = []
        for c in claims:
            verdict = self.llm.chat(JUDGE_PROMPT.format(passages=ptxt, claim=c), meter=meter, num_predict=8)
            v = _verdict(verdict)
            out.append({"ent": float(v == "SUPPORTED"), "con": float(v == "CONTRADICTED"), "best": None,
                        "raw": verdict})
        return out
