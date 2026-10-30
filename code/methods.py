"""Systems: closed-book, vanilla RAG (+abstain prompt), CoVe (closed / retrieval), RAG-Verify (+ablation flags).

Every method returns a dict with the final `output_text`, parsed `prediction`, `abstained`, and a `trace` of
intermediate steps. Cost counters are filled through the shared CostMeter.
"""
import re

import prompts as P
from verifiers import label_from_scores

IDK_RE = re.compile(r"\b(i don'?t know|unknown|cannot be determined|can ?not answer|not enough information|"
                    r"insufficient information|no information|i'?m sorry|i apologi[sz]e|"
                    r"(?:do not|don'?t) have (?:access|information|enough)|"
                    r"not (?:specified|mentioned|provided|stated|given|available|found|determinable))\b", re.I)


def parse_response(text):
    ans = re.search(r"Answer:\s*(.+)", text)
    expl = re.search(r"Explanation:\s*(.+)", text, re.S)
    answer = ans.group(1).strip() if ans else text.strip().splitlines()[0] if text.strip() else ""
    explanation = expl.group(1).strip() if expl else ""
    return answer, explanation


def is_idk(answer):
    return bool(IDK_RE.search(answer)) or answer.strip() == ""


def finalize(text, abstained=None, **extra):
    answer, _ = parse_response(text)
    abst = is_idk(answer) if abstained is None else (abstained or is_idk(answer))
    return {"output_text": text, "prediction": "" if abst else answer, "abstained": abst, **extra}


def parse_claims(text):
    claims = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip() for l in text.splitlines()]
    return [c for c in claims if len(c.split()) >= 2][:12]


def answer_claim(question, text):
    """Question-grounded answer claim. Without the question, NLI can only check the explanation
    ("The answer is X." is vacuous), which made wrong answers look supported."""
    answer = parse_response(text)[0].strip().rstrip(".")
    return f'The answer to the question "{question}" is {answer}.'


def as_statement(question, text):
    """Whole response as one hypothesis (no-decomposition ablation)."""
    return f"{answer_claim(question, text)} {parse_response(text)[1]}".strip()


# ---------------------------------------------------------------- baselines
def closed_book(ex, ctx, flags):
    return finalize(ctx.llm.chat(P.closed_book(ex["question"]), meter=ctx.meter))


def rag(ex, ctx, flags):
    passages = ctx.retriever.search(ex["question"], ctx.cfg["retriever"]["k"])
    text = ctx.llm.chat(P.rag(ex["question"], passages, abstain=flags.get("abstain_prompt", False)),
                        meter=ctx.meter)
    return finalize(text, trace={"passages": [p["pid"] for p in passages]})


def cove(ex, ctx, flags):
    q, k = ex["question"], ctx.cfg["retriever"]["k"]
    use_ret = flags.get("retrieval", False)
    if use_ret:
        passages = ctx.retriever.search(q, k)
        draft = ctx.llm.chat(P.rag(q, passages), meter=ctx.meter)
    else:
        draft = ctx.llm.chat(P.closed_book(q), meter=ctx.meter)
    n = ctx.cfg["cove"]["max_verification_questions"]
    plan = ctx.llm.chat(P.COVE_PLAN.format(n=n, question=q, draft=draft), meter=ctx.meter)
    vqs = [l.strip(" -*\t") for l in plan.splitlines() if l.strip().endswith("?")][:n]
    qa = []
    for vq in vqs:  # factored: each question answered in a separate call, without the draft
        if use_ret:
            ev = ctx.retriever.search(vq, k)
            a = ctx.llm.chat(P.COVE_EXEC_RAG.format(passages=P.fmt_passages(ev), q=vq), meter=ctx.meter,
                             num_predict=96)
        else:
            a = ctx.llm.chat(P.COVE_EXEC_CLOSED.format(q=vq), meter=ctx.meter, num_predict=96)
        qa.append((vq, a))
    final = ctx.llm.chat(P.cove_final(q, draft, qa), meter=ctx.meter) if qa else draft
    return finalize(final, trace={"draft": draft, "verification": qa})


# ---------------------------------------------------------------- RAG-Verify
def decompose(ctx, question, response, flags):
    if not flags.get("decompose", True):
        return [as_statement(question, response)]
    claims = parse_claims(ctx.llm.chat(P.DECOMPOSE.format(question=question, response=response),
                                       meter=ctx.meter, num_predict=320))
    # The abstention rule relies on claims[0] carrying the answer; do not trust the LLM's ordering.
    answer = parse_response(response)[0].lower().strip(" .")
    hit = next((i for i, c in enumerate(claims) if answer and answer in c.lower()), None)
    if hit is None:
        claims.insert(0, answer_claim(question, response))
    elif hit:
        claims.insert(0, claims.pop(hit))
    return claims


def verify(ctx, claims, passages, samples, flags, question=None, response=None):
    kind = flags.get("verifier", "nli")
    if kind == "llm_judge":
        scores = ctx.judge.score(claims, passages, meter=ctx.meter)
    elif not flags.get("retrieval", True):
        scores = ctx.nli.score_selfcheck(claims, samples)
    else:
        scores = ctx.nli.score(claims, ctx.nli.premises_from_passages(passages))
        if kind == "hybrid" and scores:
            # claims[0] is the answer claim (see decompose): judge it question-aware, keep NLI for the rest
            scores[0] = ctx.judge.score_answer(question, parse_response(response)[0], passages, meter=ctx.meter)
    tau = ctx.cfg["verifier"]["tau"]
    for s in scores:
        s["label"] = label_from_scores(s["ent"], s["con"], tau)
    return scores


def ragverify(ex, ctx, flags):
    q, cfg = ex["question"], ctx.cfg
    use_ret = flags.get("retrieval", True)
    kind = flags.get("verifier", "nli")
    passages, samples = [], []

    if use_ret:
        passages = ctx.retriever.search(q, cfg["retriever"]["k"])
        resp0 = ctx.llm.chat(P.rag(q, passages), meter=ctx.meter)
    else:
        resp0 = ctx.llm.chat(P.closed_book(q), meter=ctx.meter)
        v = cfg["verifier"]
        samples = [ctx.llm.chat(P.closed_book(q), meter=ctx.meter, temperature=v["selfcheck_temperature"],
                                seed=ctx.seed * 1000 + j) for j in range(v["selfcheck_samples"])]
    trace = {"passages": [p["pid"] for p in passages], "initial": resp0}

    if kind == "none":  # ablation: unconditional self-revision, no verification signal
        final = ctx.llm.chat(P.revise_unconditional(q, passages, resp0), meter=ctx.meter) if use_ret else resp0
        trace["revised"] = final
        return finalize(final, trace=trace)

    claims0 = decompose(ctx, q, resp0, flags)
    scores0 = verify(ctx, claims0, passages, samples, flags, q, resp0)
    trace["claims0"] = [{"claim": c, **s} for c, s in zip(claims0, scores0)]
    final, claims, scores = resp0, claims0, scores0

    flagged = [(c, s["label"]) for c, s in zip(claims0, scores0) if s["label"] != "supported"]
    if flagged and flags.get("revise", True):
        allow_idk = flags.get("abstain", True)
        prompt = (P.revise(q, passages, resp0, flagged, allow_idk) if use_ret
                  else P.revise_closed(q, resp0, flagged, allow_idk))
        final = ctx.llm.chat(prompt, meter=ctx.meter)
        trace["revised"] = final
        if not is_idk(parse_response(final)[0]):
            claims = decompose(ctx, q, final, flags)
            scores = verify(ctx, claims, passages, samples, flags, q, final)
            trace["claims1"] = [{"claim": c, **s} for c, s in zip(claims, scores)]

    abstain = False
    if flags.get("abstain", True) and not is_idk(parse_response(final)[0]):
        support_ratio = sum(s["label"] == "supported" for s in scores) / max(len(scores), 1)
        answer_ok = scores[0]["label"] == "supported" if scores else False
        abstain = (not answer_ok) or support_ratio < cfg["verifier"]["rho"]
        trace.update(support_ratio=support_ratio, answer_claim_supported=answer_ok)
    trace["verifier_abstained"] = abstain
    return finalize(final, abstained=abstain, trace=trace)


METHODS = {"closed_book": closed_book, "rag": rag, "cove": cove, "ragverify": ragverify}
