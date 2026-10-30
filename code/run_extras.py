"""Phase 3/4 extras, run unattended after the ablation grid: HaluEval detection (RQ5), controlled latency benchmark,
claim precision on test block seed0, then the ablation table. Waits for the ablation grid to finish first.

Latency benchmark: first 50 dev questions per dataset, one system at a time, nothing else on the GPU, AC power.
Claim precision: fixed Llama-3.1-8B extractor + MiniCheck on gold evidence (code/evaluate.py), seed0 only.

Usage: .venv\\Scripts\\python.exe code\\run_extras.py      (log: logs/extras_console.log)
"""
import subprocess
import sys
import time

from common import ROOT

LAT_METHODS = ["rag", "rag_abstain", "rv_no_verifier", "cove_rag", "ragverify"]
CP_METHODS = ["closed_book", "rag", "rag_abstain", "rv_no_verifier", "cove_rag", "ragverify"]
DATASETS = ["hotpotqa", "triviaqa"]


def sh(args):
    print(">>", " ".join(args), flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, *args], cwd=ROOT)
    print(f"   exit={rc} in {time.perf_counter() - t0:.0f}s", flush=True)
    return rc


def main():
    if sys.platform == "win32":  # block idle sleep while this runs (not lid-close or low-battery sleep)
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    grid_log = ROOT / "logs" / "grid_ablations_console.log"
    while "FAILURES:" not in grid_log.read_text(encoding="utf-8", errors="ignore"):
        time.sleep(60)
    print("ablation grid finished:", grid_log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1],
          flush=True)

    steps = [["code/halueval_detect.py"]]
    for ds in DATASETS:
        for m in LAT_METHODS:
            steps.append(["code/run.py", "--dataset", ds, "--split", "dev", "--method", m, "--n", "50", "--tag", "latency"])
            steps.append(["code/evaluate.py", "--dataset", ds, "--split", "dev", "--method", m, "--tag", "latency",
                          "--no-claims"])
    for ds in DATASETS:
        for m in CP_METHODS:
            steps.append(["code/evaluate.py", "--dataset", ds, "--split", "seed0", "--method", m])
    steps.append(["code/aggregate_ablations.py"])

    failures = [s for s in steps if sh(s) != 0]
    print("EXTRAS FAILURES:", failures or "none", flush=True)


if __name__ == "__main__":
    main()
