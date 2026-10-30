# Phase 2 — Implementation and smoke test report (27 Oct 2026)

**Scope.** Every pipeline was run on the first 20 questions of the held-out **dev** block of each dataset. The dev block is disjoint from the three 300-question test subsets. These runs check that the code works; **they are not results and will not appear in the paper.** With n = 20, one question equals 5 percentage points.

Source files: `results/raw/<dataset>/<method>/dev_smoke.jsonl`, `results/metrics/...`, `logs/eval_smoke_final.log`, `logs/run_*.log`.

## 1. Everything runs

| | |
|---|---|
| Pipelines tested | 15 (closed-book, RAG, RAG+abstain prompt, CoVe, CoVe+retrieval, RAG-Verify, 5 component ablations, LLM-judge verifier, k=10, Llama-3.1 backbone, Self-RAG) |
| Datasets | HotpotQA, TriviaQA |
| Runtime errors | **0 / 600 examples** |
| Self-RAG weights | official `selfrag/selfrag_llama2_7b` main branch, both shards SHA-256 verified; loaded in 4-bit NF4; peak VRAM 5.48 GB |
| Evaluation | answer metrics + MiniCheck claim precision computed for all 30 runs |

## 2. Smoke numbers (dev, n = 20 — sanity check only)

Acc = EM-or-contains match; Abst = abstention rate; Hall = answered-and-wrong rate; ClaimP = MiniCheck claim precision vs gold evidence.

| Method | HotpotQA Acc / Abst / Hall / ClaimP | TriviaQA Acc / Abst / Hall / ClaimP |
|---|---|---|
| closed_book | .15 / .00 / .85 / .25 | .50 / .00 / .50 / .45 |
| rag | .40 / .05 / .55 / .61 | .65 / .00 / .35 / .67 |
| rag_abstain | .35 / .35 / .30 / .72 | .55 / .25 / .20 / .74 |
| cove | .20 / .00 / .80 / .24 | .60 / .00 / .40 / .50 |
| cove_rag | .40 / .05 / .55 / .57 | .60 / .00 / .40 / .57 |
| selfrag | .45 / .05 / .50 / .51 | .65 / .00 / .35 / .50 |
| **ragverify** | .35 / .25 / .40 / .57 | .50 / .25 / .25 / .67 |
| rv_no_retrieval | .05 / .75 / .20 / .44 | .35 / .50 / .15 / .67 |
| rv_no_decomp | .40 / .10 / .50 / .60 | .65 / .00 / .35 / .66 |
| rv_no_verifier | .35 / .35 / .30 / .74 | .50 / .35 / .15 / .80 |
| rv_no_revision | .35 / .25 / .40 / .58 | .55 / .15 / .30 / .68 |
| rv_llm_judge | .25 / .55 / .20 / .66 | .40 / .35 / .25 / .73 |
| rv_k10 | .35 / .15 / .50 / .68 | .70 / .10 / .20 / .60 |
| ragverify_llama | .25 / .40 / .35 / .62 | .50 / .25 / .25 / .65 |

(rv_no_abstain was redesigned after this run; see §3.)

## 3. Problems found and fixed

1. **"Not specified" counted as an answer.** The refusal detector now also covers *not specified / not mentioned / I'm sorry / don't have access*. It is applied identically to every system at evaluation time.
2. **Answer claim not always first after decomposition.** The code now moves the claim that contains the answer to the front, or synthesises one if no claim contains it.
3. **Latency was not trustworthy.** Plain RAG appeared slower than RAG-Verify, which is impossible. Each run now starts with an untimed warm-up call. Ollama's own prompt-processing and generation times are recorded per call. Re-check: RAG 0.87–1.17 s/query, RAG-Verify 2.33 s/query (HotpotQA dev, n=20).
4. **The no-abstention ablation was nearly a no-op.** Abstentions came from the revision prompt allowing "I don't know". With `abstain: false` the revision prompt now requires an answer. Re-check: abstention .25 → .05 and hallucination .40 → .60 on HotpotQA dev.
5. **Self-RAG download.** The HF CDN served 0.22 MB/s per connection, and the library silently fetched a bot-converted safetensors PR. Replaced with `code/fetch_selfrag.py`: parallel range requests from the main branch, with SHA-256 verification.

## 4. Determinism (measured)

The same RAG run was repeated with temperature 0 and a fixed seed:

- **Output text:** identical in 17/20 (chk1 vs chk2) and 16/20 (chk1 vs the earlier smoke run).
- **Final answers:** identical in 20/20.

Ollama's GPU inference is therefore not bit-for-bit reproducible, and the paper will say so. The reported std over seeds captures subset variance, not this effect.

## 5. Observations to discuss before Phase 3 (not conclusions; n = 20)

- **Prompted abstention looks as strong as RAG-Verify.** On these 20 questions, `rag_abstain` (one call, prompt allows "I don't know") reaches equal or lower hallucination than `ragverify`. So does `rv_no_verifier` (unconditional self-revision). If this holds at scale, the verifier adds little over prompted abstention, which is the paper's main risk.
- **`rv_no_decomp` never triggered a revision in 40 examples.** NLI seems to accept a whole-response statement too easily. This needs a check: it might be a real finding (decomposition matters) or a premise/truncation artefact.
- **Self-RAG's F1 is low but its match accuracy is fine.** It answers in full sentences ("X is older than Y…"), which token-F1 penalises. We will use match accuracy as the primary metric and report F1 as secondary.
- **The τ = 0.5 and ρ = 0.5 defaults were fixed in advance and never tuned.** Tuning them on the 100-question dev block (never on test) is legitimate and standard.

## 6. Suggested next step

Run the four key systems (rag, rag_abstain, ragverify, rv_no_verifier) on the **full 100-question dev block** (~15 min), check the `rv_no_decomp` behaviour, and select τ/ρ on dev. Then launch Phase 3 on the test subsets with settings frozen.
