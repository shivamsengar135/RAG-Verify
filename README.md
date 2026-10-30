# RAG-Verify

Code, configurations, per-question outputs and logs for the paper on claim-level verification for
retrieval-augmented QA with 7–8B open models on a single 8 GB GPU (HotpotQA, TriviaQA, HaluEval-QA).

## Setup

Tested on Windows 11, Python 3.11, NVIDIA RTX 4060 (8 GB), Ollama 0.40.12.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
```

`torch` is the CUDA 12.8 build; install it from the PyTorch index if pip cannot find it.

## Reproduce

All settings, seeds and model names are in `configs/base.yaml`; the experiment grids are in `configs/grids.yaml`.

```
.venv\Scripts\python.exe code\env_check.py             # GPU, Ollama and models
.venv\Scripts\python.exe code\data_prep.py             # benchmarks, fixed splits, retrieval corpora -> data/
.venv\Scripts\python.exe code\run_grid.py main         # main results (generation + evaluation, resumable)
.venv\Scripts\python.exe code\run_grid.py ablations    # ablations (first test block)
.venv\Scripts\python.exe code\run_extras.py            # HaluEval, latency, claim precision, ablation table
.venv\Scripts\python.exe code\aggregate.py             # tables in results/tables and paper/tables
.venv\Scripts\python.exe code\check_paper.py           # checks every number in paper/main.tex against results/
```

Self-RAG needs `code\fetch_selfrag.py` before its runs.

## Layout

- `code/`: pipeline, baselines, evaluation and analysis scripts (each file's docstring gives its usage)
- `configs/`: settings and grids
- `results/raw/`: generated outputs per question; `results/metrics/`: scored outputs; `results/tables/`: summaries
- `results/error_analysis/`, `results/halueval/`: analyses reported in the paper
- `logs/`: run logs and `env_changes.log`
- `paper/`: LaTeX source (IEEE template) and tables generated from `results/`

`data/` (about 17 GB of benchmarks, indexes and model weights) is not included; `code/data_prep.py` rebuilds it.
