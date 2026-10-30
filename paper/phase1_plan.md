# Phase 1 — Literature scan and experiment plan

Working title: **RAG-Verify: Retrieval-Augmented Generation with Claim-Level Verification for Hallucination Mitigation in LLMs**
Authors: Shivam Sengar, Yash Prabhakar Sonawane, Bansidhar Joshi (DIAT Pune) — order to be confirmed.
Date: 27 Oct 2026. All references below were checked against the arXiv API / Crossref by `code/verify_refs.py`
(output: `data/refs_verified.json`, 35/35 found). Summaries are written from the official abstracts.

---

## 1. Target conference options (dates checked on official sites, 27 Oct 2026)

| | IEEE CONECCT 2027 | IEEE ICRCICN 2027 | IEEE MPCON 2027 |
|---|---|---|---|
| Full name | 13th IEEE Int. Conf. on Electronics, Computing and Communication Technologies | 8th IEEE Int. Conf. on Research in Computational Intelligence and Communication Networks | IEEE Madhya Pradesh Section Conference |
| Where / when | Bangalore, 29–31 Jul 2027 | Kalyani GEC, West Bengal (hybrid), 27–28 Feb 2027 | AITR Indore, 17–19 Apr 2027 |
| Organiser | IEEE Bangalore Section | IEEE conference record #73691; Xplore-compliant ISBN listed | IEEE MP Section; IEEE CPS proceedings |
| **Paper deadline** | **15 Dec 2026** | **30 Nov 2026** | **15 Nov 2026** (abstract deadline 15 Oct 2026 already passed) |
| Notification | 15 Feb 2027 | 1 Feb 2027 | 15 Jan 2027 |
| Page limit | Not yet published (CFP page not live yet) | Not stated on home page | 6 pages, extra pages charged |
| Days left | 49 | 34 | 19 |
| Sources | ieee-conecct.org | icrcicn.in | aitr.ac.in/mpcon2027 |

**Recommendation: CONECCT 2027.** It is the IEEE Bangalore Section's flagship conference, explicitly includes AI/ML, and its 15 Dec deadline leaves about 3 weeks of buffer after the planned finish (see §8). ICRCICN is a feasible backup if we compress Phases 4–5. MPCON is listed only for completeness: its abstract stage has closed and 19 days is not enough to run the experiments honestly.
Caveat: CONECCT's site says its CFP "opens Dec 2026" while giving a 15 Dec deadline, and the CFP/tracks pages currently return 404. I will re-check before we commit; deadlines at these conferences are often extended, but we should not plan on that.
Searches did not find official 2027 dates for ICCCNT or INDICON (their 2027 CFPs are not out yet).

---

## 2. Literature scan (2023–2026)

**Mitigation methods closest to ours**
1. **Self-RAG** — Asai et al., 2023, arXiv:2310.11511 (ICLR 2024). Trains a single LM to emit reflection tokens that decide when to retrieve and to critique the relevance and support of its own outputs. *Requires fine-tuning; we are training-free.*
2. **Chain-of-Verification (CoVe)** — Dhuliawala et al., 2023, arXiv:2309.11495. Drafts an answer, plans verification questions, answers them independently, and produces a revised answer. *Verification relies on the same model's parametric knowledge, not external evidence.*
3. **CRAG** — Yan et al., 2024, arXiv:2401.15884. A light retrieval evaluator scores retrieved documents and triggers corrective actions (refine, discard, or web search). *Checks the evidence, not the generated claims.*
4. **FLARE** — Jiang et al., 2023, arXiv:2305.06983. Actively re-retrieves when an upcoming sentence contains low-confidence tokens.
5. **RARR** — Gao et al., arXiv:2210.08726 (ACL 2023). Post-hoc research-and-revise: finds evidence for a generated text and edits unsupported content while preserving the original.

**Claim-level factuality evaluation / verification**
6. **FActScore** — Min et al., 2023, arXiv:2305.14251 (EMNLP 2023). Splits long-form generations into atomic facts and measures the fraction supported by a knowledge source. *Basis of our claim-precision metric.*
7. **FacTool** — Chern et al., 2023, arXiv:2307.13528. Tool-augmented framework that extracts claims and verifies them with search, code execution, etc.
8. **SAFE / LongFact** — Wei et al., 2024, arXiv:2403.18802. LLM agent rates each fact using Google Search; proposes F1@K for long-form factuality.
9. **MiniCheck** — Tang et al., 2024, arXiv:2404.10774 (EMNLP 2024). Small fact-checkers trained on synthetic data reach GPT-4-level grounding accuracy at far lower cost. *We use it as an independent evaluator (§5).*
10. **SelfCheckGPT** — Manakul et al., 2023, arXiv:2303.08896 (EMNLP 2023). Detects hallucinations from inconsistency across sampled responses, without retrieval. *Basis of our "no retrieval" ablation verifier.*
11. **Semantic entropy** — Farquhar et al., *Nature* 630, 2024, doi:10.1038/s41586-024-07421-0. Measures uncertainty over meaning clusters of sampled answers to detect confabulations.

**Recent (2025–2026) work**
12. **AFEV** — Zheng et al., 2025, arXiv:2506.07446. Iteratively decomposes complex claims into atomic facts with fine-grained retrieval and evidence reranking for fact verification.
13. **Atomic fact checking in medical RAG** — Vladika et al., 2025, arXiv:2505.24830. Decomposes long-form medical answers into atomic facts, verifies each against guideline documents, and corrects errors. *Closest in spirit to ours; domain-specific, long-form QA.*
14. **R2VC** — Dixit et al., 2026, arXiv:2609.11955. Modular retrieve–reason–verify–calibrate pipeline with an NLI cross-encoder and selective abstention for FEVER fact-checking; uses a fine-tuned, DPO-aligned generator.
15. **Evidence-aligned entity verification** — Jia et al., 2026, arXiv:2609.08267. Studies RAG-based hallucination detection using entity-level alignment with retrieved evidence.
16. **FaithJudge** — Tamber et al., 2025, arXiv:2505.04847 (EMNLP 2025 Industry). LLM-as-judge for RAG faithfulness using human-annotated examples; maintains a leaderboard.
17. **Hallucination-detection benchmark desiderata** — Chen et al., 2026, arXiv:2605.11330. Argues existing detection benchmarks lack long-context RAG grounding and realistic label noise.

**Benchmarks and surveys (context):** HaluEval (Li et al., 2023, arXiv:2305.11747); RAGTruth (Niu et al., arXiv:2401.00396); hallucination survey (Huang et al., arXiv:2311.05232); capability-oriented mitigation survey (Li et al., 2025, arXiv:2510.24476); RAG survey (Gao et al., arXiv:2312.10997); abstention survey (Wen et al., 2024, arXiv:2407.18418); "Lost in the Middle" (Liu et al., arXiv:2307.03172).

**Positioning (honest).** Decompose-then-verify is established (FActScore, FacTool, SAFE) and has been used for correction (RARR, Vladika et al. 2025) and with abstention (R2VC). Our contribution is therefore **not** the idea of claim verification itself. It is:
(i) a training-free, single-GPU (8 GB) pipeline that combines evidence-grounded claim verification with targeted revision *and* abstention, for short-answer open-domain QA;
(ii) a controlled comparison against CoVe, Self-RAG and vanilla RAG under identical retrieval;
(iii) a systematic component ablation plus a risk–coverage analysis and a per-query cost analysis.
Reviewers at CONECCT/ICRCICN will accept this as an empirical systems paper if the ablations are clean. It would be weak for a top NLP venue.

---

## 3. Research questions and hypotheses

| RQ | Question | Hypothesis (directional, to be tested) |
|---|---|---|
| RQ1 | Does claim-level verification with revision/abstention reduce hallucination compared with vanilla RAG, CoVe and Self-RAG? | H1: RAG-Verify has the lowest hallucination rate and highest claim precision, with a smaller change in EM/F1 than in hallucination rate. |
| RQ2 | Which component contributes most? | H2: removing the verifier raises hallucination the most; removing retrieval lowers EM/F1 the most. |
| RQ3 | How does the verification threshold trade coverage for precision? | H3: raising τ monotonically increases abstention and selective accuracy (a usable risk–coverage curve). |
| RQ4 | Do results hold across backbones and verifier types, and at what cost? | H4: gains hold for both Qwen2.5-7B and Llama-3.1-8B; a small NLI verifier matches an LLM judge at a fraction of the latency. |
| RQ5 | How good is the claim verifier as a hallucination detector on its own? | H5: NLI-based claim verification separates correct vs. hallucinated answers on HaluEval-QA well above chance (AUROC). |

If any hypothesis fails, we report it as a negative finding — no tuning on the test subsets.

---

## 4. Datasets (and why)

| Dataset | Role | Split / subset | Retrieval corpus |
|---|---|---|---|
| **HotpotQA** (distractor) — Yang et al., arXiv:1809.09600 | Main: multi-hop QA. Needs facts from 2 paragraphs, so decomposed claims map naturally to hops. Gold supporting facts allow an independent claim-precision check. | validation (7,405); 3 disjoint random subsets × 300 (seeds 0/1/2) | All paragraphs of the full validation set pooled and deduplicated (~66k), so every question competes against all distractors, not only its own 10 |
| **TriviaQA-Wikipedia** (rc.wikipedia) — Joshi et al., arXiv:1705.03551 | Main: single-hop, long-tail factoid QA where 7B models hallucinate closed-book. Ships with evidence Wikipedia pages. | validation (7,993); 3 × 300 | Evidence pages of the 900 sampled questions plus those of 1,100 extra validation questions as distractors, split into ~100-word passages |
| **HaluEval-QA** — Li et al., arXiv:2305.11747 | RQ5 only: does the verifier flag hallucinated answers given the knowledge? Cheap (NLI only). | 1,000 random items (seed 0), each with a correct and a hallucinated answer | Provided knowledge snippet |

**Not used, with reasons:** TruthfulQA (misconception-style questions, no evidence corpus; retrieval would need live web search) — can be added as a closed-book appendix if reviewers ask. FEVER is itself a verification task rather than generation. Natural Questions needs a full Wikipedia index (21M passages), which is out of scope on this laptop.
**Corpus caveat (will be stated in the paper):** our retrieval corpora are pooled from benchmark evidence, which is easier than full-Wikipedia retrieval.

---

## 5. Methods and metrics

**Shared setup:** retriever = bge-small-en-v1.5 dense, cosine, FAISS flat index (Xiao et al., arXiv:2309.07597); default k = 5; greedy decoding (temperature 0) unless a method needs sampling; identical prompts/answer format across methods where possible; backbone default = Qwen2.5-7B-Instruct Q4_K_M (arXiv:2412.15115).

**Systems**
- (a) **Closed-book** — LLM alone.
- (b) **Vanilla RAG** — Lewis et al., arXiv:2005.11401 style: top-k passages in the prompt.
- (c) **CoVe (factored)** — faithful prompt-based reimplementation (draft → plan questions → answer each independently → final). Closed-book as in the paper. We also report a CoVe+retrieval variant so it is not handicapped.
- (d) **Self-RAG** — the official `selfrag/selfrag_llama2_7b` checkpoint with the authors' inference settings, using our retriever and corpus. It has its own (Llama-2-7B) backbone; this is noted in the table. Loading plan: 4-bit via transformers + bitsandbytes using the official weights, with no third-party conversions. If that fails on Windows, I will report back before choosing an alternative.
- (e) **RAG-Verify (ours):**
  1. Retrieve top-k passages.
  2. Generate the answer plus a short rationale.
  3. Decompose the output into atomic claims (few-shot prompt); mark the claim carrying the final answer.
  4. Verify each claim against each passage with DeBERTa-v3-large NLI (Laurer et al., doi:10.1017/pan.2023.20; He et al., arXiv:2111.09543). Support score s(c) = max over passages of P(entail); label c *supported* if s(c) ≥ τ, *contradicted* if max P(contra) ≥ τ, otherwise *unsupported*.
  5. If any claim is not supported, revise: prompt the LLM with the flagged claims and the evidence; then re-verify once.
  6. Abstain if the answer-bearing claim is still not supported (or the support ratio is below ρ).

**Metrics** (all computed by code from saved outputs)
- **Task accuracy:** EM and token-F1 (SQuAD normalisation) over all questions (abstention counts as wrong) and over answered questions only (selective accuracy); coverage = 1 − abstention rate.
- **Hallucination rate:** fraction of all questions where the system *answered* and was wrong. This is the confident-error rate; abstaining is not counted as a hallucination.
- **Claim precision (FActScore-style):** the fraction of output claims supported by **gold** evidence (HotpotQA supporting facts / TriviaQA evidence), judged by **MiniCheck-Flan-T5-Large**. This is a different model from our pipeline's verifier, which avoids the system grading itself. Claims are extracted by one fixed extractor for all systems.
- **Risk–coverage curve / AURC** over τ.
- **Cost:** wall-clock seconds per query, LLM calls per query, generated tokens per query; peak VRAM.
- **RQ5:** AUROC, accuracy and F1 of verifier-based hallucination detection on HaluEval-QA.
- **Statistics:** mean ± std over the 3 seeds (= 3 disjoint subsets); paired bootstrap (10k resamples) for ours vs. the best baseline on pooled examples.

**Evaluator validity check:** 100 claims double-annotated by two humans (suggested: Yash and you) to report MiniCheck–human agreement (Cohen's κ). I will build the annotation sheet; I will not annotate these myself, since that would not count as human validation.

---

## 6. Experiment grid

**Main (Phase 3):** 5 systems (+ CoVe-RAG variant) × 2 datasets × 3 seeds × 300 questions, backbone Qwen2.5-7B. Self-RAG uses its own model. Plus RQ5 on HaluEval-QA.

**Ablations (Phase 4)** — HotpotQA + TriviaQA, seed 0, 300 questions, Qwen2.5-7B unless noted:

| Factor | Settings |
|---|---|
| Component removal | full; − retrieval (verifier switches to SelfCheckGPT-style sample consistency, 5 samples); − decomposition (whole response verified as one claim); − verifier (unconditional self-revision); − revision (verify → keep or abstain); − abstention (always answer) |
| Retrieval top-k | 1, 3, 5, 10 |
| Threshold τ | 0.3, 0.5, 0.7, 0.9 (+ dense sweep from cached NLI scores for the risk–coverage curve) |
| Backbone | Qwen2.5-7B vs Llama-3.1-8B (arXiv:2407.21783): vanilla RAG and RAG-Verify, both datasets, 3 seeds |
| Verifier | DeBERTa-v3 NLI vs LLM-as-judge (same backbone) |

**Error analysis:** 50 RAG-Verify failures (HotpotQA + TriviaQA, seed 0) sorted into categories: retrieval miss / decomposition error / verifier false-accept / verifier false-reject / revision introduced an error / over-abstention / gold-label issue. I will also include 2–3 qualitative examples.

---

## 7. Compute budget (estimates; to be replaced by Phase 2 measurements)

Measured so far: generation 45–48 tok/s, peak VRAM 5.8 GB with LLM + NLI + embedder.

| Block | Estimate |
|---|---|
| Corpus embedding (HotpotQA ~66k + TriviaQA passages) | < 30 min |
| Main grid (2 × 900 q × all systems) | ~14–16 h |
| Backbone swap | ~5 h |
| Component ablations + top-k + τ + verifier swap | ~8–10 h |
| RQ5 (HaluEval, NLI only) | < 15 min |
| **Total** | **~30 GPU-hours (3–4 overnight runs)** |

All runs use per-example caching (JSONL) so a crash resumes where it stopped.

---

## 8. Timeline (for the CONECCT 15 Dec deadline)

| Phase | Target finish |
|---|---|
| 2 — Implementation + 20-example smoke tests | 3 Nov |
| 3 — Main experiments | 10 Nov |
| 4 — Ablations, error analysis, figures | 18 Nov |
| 5 — Paper draft | 1 Dec |
| 6 — Final checks, PDF, zip | 5 Dec (10 days of buffer) |

---

## 9. Decisions needed from you
1. Conference: CONECCT 2027 (recommended) / ICRCICN 2027 / other.
2. Dataset choice (HotpotQA + TriviaQA-Wiki + HaluEval-QA) and subset size 3 × 300 per dataset.
3. Self-RAG via the official checkpoint (own Llama-2-7B backbone) as baseline (c2), alongside CoVe.
4. Whether Yash and you can annotate 100 claims (about 1 hour) for evaluator validation.
5. Author order.
