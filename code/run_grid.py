"""Run a named grid from configs/grids.yaml end to end (generation, then evaluation). Resumable.

Usage: .venv\\Scripts\\python.exe code\\run_grid.py smoke   [--no-eval] [--no-claims]
"""
import argparse
import subprocess
import sys
import time

import yaml

from common import ROOT


def sh(args, log):
    print(">>", " ".join(args), flush=True)
    log.write(">> " + " ".join(args) + "\n")
    log.flush()
    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, *args], cwd=ROOT)
    msg = f"   exit={rc} in {time.perf_counter() - t0:.0f}s"
    print(msg, flush=True)
    log.write(msg + "\n")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-claims", action="store_true")
    args = ap.parse_args()
    if sys.platform == "win32":
        # Block idle sleep while the grid runs (released when this process exits). A sleep/resume
        # bugcheck killed the grid on 28 Oct. Does not stop sleep on lid close.
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    g = yaml.safe_load((ROOT / "configs" / "grids.yaml").read_text())[args.grid]
    registry = yaml.safe_load((ROOT / "configs" / "methods.yaml").read_text())
    backbone = lambda m: (registry[m].get("cfg", {}).get("llm", {}).get("backbone", "qwen")  # noqa: E731
                          if registry[m]["fn"] != "selfrag" else "~selfrag")
    methods = sorted(g["methods"], key=lambda m: (backbone(m) != "qwen", backbone(m)))  # stable: qwen, llama, selfrag
    extra = (["--n", str(g["n"])] if g.get("n") else []) + (["--tag", g["tag"]] if g.get("tag") else [])

    failures = []
    with open(ROOT / "logs" / f"grid_{args.grid}.log", "a", encoding="utf-8") as log:
        for m in methods:
            for ds in g["datasets"]:
                for sp in g["splits"]:
                    script = "code/selfrag_run.py" if registry[m]["fn"] == "selfrag" else "code/run.py"
                    margs = [] if registry[m]["fn"] == "selfrag" else ["--method", m]
                    if sh([script, "--dataset", ds, "--split", sp, *margs, *extra], log) != 0:
                        failures.append((m, ds, sp))
        if not args.no_eval:
            for m in methods:
                for ds in g["datasets"]:
                    for sp in g["splits"]:
                        eargs = ["code/evaluate.py", "--dataset", ds, "--method", m, "--split", sp]
                        eargs += (["--tag", g["tag"]] if g.get("tag") else []) + (["--no-claims"] if args.no_claims else [])
                        if sh(eargs, log) != 0:
                            failures.append(("eval", m, ds, sp))
    print("FAILURES:", failures or "none")


if __name__ == "__main__":
    main()
