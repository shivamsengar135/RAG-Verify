# Dev-block selection report (100 dev questions per dataset; test subsets untouched)

Sources: `results/raw/*/*/dev_full.jsonl`, `logs/tune_dev.log`, `results/tuning/dev_design_tau_rho.csv`,
`results/tuning/selected.json`, `results/tuning/dev_compare.csv` (`code/dev_compare.py`).

## Selection (pre-declared rule, code/tune_dev.py)
Effective reliability Phi_1 = accuracy - hallucination rate (+1 correct, -1 wrong, 0 abstain), averaged over the two dev sets.
- Search space: design {NLI, hybrid, LLM judge} x tau {0.3, 0.5, 0.7, 0.9} x rho {0, 0.25, 0.5, 0.75, 1}.
- **Selected: hybrid verifier, tau = 0.3, rho = 0.0.** Mean Phi_1 = 0.200. The best value was 0.205 (rho = 0.75); the tie-break within 0.01 preferred higher coverage.
- tau and rho matter little for the hybrid design (Phi_1 range 0.175-0.205), because its decisions are dominated by the binary answer-claim judgement. rho = 0 means only the answer claim gates abstention.

## All systems on dev, Phi_1 (paired bootstrap 95% CI over 200 pooled questions)

| System | HotpotQA | TriviaQA | Mean | Abstention | Selected minus system [95% CI] |
|---|---|---|---|---|---|
| RAG | -0.280 | 0.400 | 0.060 | 0.020 | **+0.140 [0.050, 0.230]** |
| RAG + abstain prompt | -0.100 | 0.410 | 0.155 | 0.265 | +0.045 [-0.035, 0.125] |
| RAG-Verify, NLI only (tau 0.5, rho 0.5) | -0.080 | 0.400 | 0.160 | 0.250 | +0.040 [-0.035, 0.115] |
| RAG-Verify, LLM judge | -0.020 | 0.360 | 0.170 | 0.410 | +0.030 [-0.035, 0.095] |
| No decomposition | -0.280 | 0.400 | 0.060 | 0.040 | +0.140 [0.050, 0.230] |
| **Unconditional self-revision (no verifier)** | -0.010 | 0.460 | **0.225** | 0.305 | -0.025 [-0.090, 0.040] |
| **Selected: hybrid, tau 0.3, rho 0** | -0.070 | 0.470 | 0.200 | 0.420 | - |

## What this says (dev only, n = 200)
1. **The verifier helps relative to plain RAG.** The gain over RAG is the only difference whose CI excludes zero.
2. **The verifier does not beat the simple alternatives.** Against prompted abstention, NLI-only and the LLM judge, the differences are small and not significant.
3. **Unconditional self-revision is at least as good.** Re-reading the passages and correcting the answer, with "I don't know" allowed and no verifier, scores numerically higher than the selected design (not significant).
4. **The selected design abstains a lot (42%).** It buys low hallucination on TriviaQA (0.09) at a cost in accuracy.
5. **No system reaches positive Phi_1 on HotpotQA.** The 7B model answers multi-hop questions wrongly more often than correctly. Part of this is strict answer matching (F1 0.48 vs. match accuracy 0.35 for RAG), which the error analysis must quantify.

## Incidents
- HotpotQA tau = 0.9 first run: 76/100 examples hit **CUDA out-of-memory** because another process took VRAM. The bad file was kept as `logs/discarded_hotpotqa_rv_tau09_dev_full_cuda_oom.jsonl`. The rerun was clean.
- `code/run.py` now **stops** on a CUDA error instead of recording the example as an abstention.
