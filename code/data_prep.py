"""Download benchmarks, draw fixed disjoint subsets, and build retrieval corpora.

Usage: .venv\\Scripts\\python.exe code\\data_prep.py
Outputs (per dataset, under data/<name>/):
  questions.jsonl  id, question, answers (list), gold_titles, [gold_evidence for HotpotQA]
  splits.json      {"seed0": [...ids], "seed1": [...], "seed2": [...], "dev": [...]}
  corpus.jsonl     pid, title, text
  stats.json       counts
"""
import json
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from common import ROOT, load_config, write_json

CFG = load_config()
S = CFG["subsets"]


def make_splits(ids):
    """Seeds 0..n-1 are disjoint test blocks; 'dev' is the block right after them."""
    perm = np.random.default_rng(S["permutation_seed"]).permutation(len(ids))
    n, t = S["num_test_seeds"], S["test_size"]
    splits = {f"seed{s}": [ids[i] for i in perm[s * t:(s + 1) * t]] for s in range(n)}
    splits["dev"] = [ids[i] for i in perm[n * t:n * t + S["dev_size"]]]
    return splits, perm


def load_parquets(repo, prefix):
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"No parquet files under {repo}/{prefix}")
    frames = [pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in sorted(files)]
    return pd.concat(frames, ignore_index=True)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prep_hotpotqa():
    out = ROOT / "data" / "hotpotqa"
    df = load_parquets("hotpotqa/hotpot_qa", "distractor/validation")
    questions, corpus = [], OrderedDict()
    for row in df.itertuples():
        titles = list(row.context["title"])
        paras = {t: " ".join(s.strip() for s in sents) for t, sents in zip(titles, row.context["sentences"])}
        for t, text in paras.items():
            corpus.setdefault((t, text), None)
        gold_titles = list(OrderedDict.fromkeys(row.supporting_facts["title"]))
        questions.append({
            "id": row.id,
            "question": row.question,
            "answers": [row.answer],
            "type": row.type,
            "level": row.level,
            "gold_titles": gold_titles,
            "gold_evidence": [f"{t}: {paras[t]}" for t in gold_titles if t in paras],
        })
    corpus_rows = [{"pid": f"hp{i}", "title": t, "text": f"{t}: {text}"} for i, (t, text) in enumerate(corpus)]
    splits, _ = make_splits([q["id"] for q in questions])
    write_jsonl(out / "questions.jsonl", questions)
    write_jsonl(out / "corpus.jsonl", corpus_rows)
    write_json(out / "splits.json", splits)
    stats = {"questions": len(questions), "corpus_passages": len(corpus_rows),
             **{k: len(v) for k, v in splits.items()}}
    write_json(out / "stats.json", stats)
    print("hotpotqa", stats)


def chunk_words(text, n):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n) if words[i:i + n]]


def prep_triviaqa():
    out = ROOT / "data" / "triviaqa"
    df = load_parquets("mandarjoshi/trivia_qa", "rc.wikipedia/validation")
    df = df[df["entity_pages"].apply(lambda p: len(p["title"]) > 0)].reset_index(drop=True)
    ids = df["question_id"].tolist()
    splits, perm = make_splits(ids)
    used = S["num_test_seeds"] * S["test_size"] + S["dev_size"]
    n_dist = CFG["retriever"]["triviaqa_distractor_questions"]
    distractor_ids = [ids[i] for i in perm[used:used + n_dist]]
    keep = set(sum(splits.values(), [])) | set(distractor_ids)

    questions, pages = [], OrderedDict()
    for row in df.itertuples():
        if row.question_id not in keep:
            continue
        titles = list(row.entity_pages["title"])
        for t, ctx in zip(titles, row.entity_pages["wiki_context"]):
            pages.setdefault(t, ctx)
        if row.question_id in distractor_ids:
            continue
        ans = row.answer
        aliases = list(OrderedDict.fromkeys([ans["value"], *ans["aliases"]]))
        questions.append({"id": row.question_id, "question": row.question, "answers": aliases,
                          "gold_titles": titles})
    corpus_rows, n = [], CFG["retriever"]["triviaqa_chunk_words"]
    for t, ctx in pages.items():
        for j, chunk in enumerate(chunk_words(ctx, n)):
            corpus_rows.append({"pid": f"tq{len(corpus_rows)}", "title": t, "text": f"{t}: {chunk}"})
    write_jsonl(out / "questions.jsonl", questions)
    write_jsonl(out / "corpus.jsonl", corpus_rows)
    write_json(out / "splits.json", splits)
    stats = {"validation_with_pages": len(df), "questions_kept": len(questions),
             "distractor_questions": len(distractor_ids), "pages": len(pages),
             "corpus_passages": len(corpus_rows), **{k: len(v) for k, v in splits.items()}}
    write_json(out / "stats.json", stats)
    print("triviaqa", stats)


def prep_halueval():
    out = ROOT / "data" / "halueval"
    df = load_parquets("pminervini/HaluEval", "qa/")
    # qa/ also holds a single-answer "samples" variant whose paired columns are empty; keep the paired QA rows only
    df = df.dropna(subset=["right_answer", "hallucinated_answer"]).reset_index(drop=True)
    idx = np.random.default_rng(S["permutation_seed"]).permutation(len(df))[:1000]
    rows = []
    for i in idx:
        r = df.iloc[int(i)]
        for label, ans in (("correct", r["right_answer"]), ("hallucinated", r["hallucinated_answer"])):
            rows.append({"id": f"he{int(i)}_{label}", "question": r["question"], "knowledge": r["knowledge"],
                         "answer": ans, "label": label})
    write_jsonl(out / "items.jsonl", rows)
    write_json(out / "stats.json", {"source_rows": len(df), "sampled_questions": len(idx), "items": len(rows)})
    print("halueval", len(rows), "items")


if __name__ == "__main__":
    prep_hotpotqa()
    prep_triviaqa()
    prep_halueval()
