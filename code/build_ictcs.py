"""Build the ICTCS 2026 (Springer SVProc) version of the paper and run its checks.

Copies the generated table rows (paper/tables) and the verified bibliography (paper/refs.bib, without the arXiv URL
lines) into the paper folder, runs pdflatex + bibtex, then code/check_paper.py on it.
Usage: .venv\\Scripts\\python.exe code\\build_ictcs.py                          (submitted version, paper_ictcs/)
       .venv\\Scripts\\python.exe code\\build_ictcs.py --paper paper_ictcs_crc  (camera-ready copy)
"""
import argparse
import re
import shutil
import subprocess
import sys

from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--paper", default="paper_ictcs", choices=["paper_ictcs", "paper_ictcs_crc"])
SRC, DST = ROOT / "paper", ROOT / ap.parse_args().paper


def main():
    height = "1.5" if DST.name == "paper_ictcs_crc" else "1.7"  # the camera-ready copy needs a slightly flatter figure
    subprocess.run([sys.executable, str(ROOT / "code" / "make_figure.py"), DST.name, height], check=True)
    (DST / "tables").mkdir(exist_ok=True)
    for f in (SRC / "tables").glob("*.tex"):
        shutil.copy2(f, DST / "tables" / f.name)
    # 10-page budget: the printed ablation table keeps the component and verifier groups (the camera-ready copy only the
    # component group); the other groups are reported in the prose and stay in full in results/tables/ablations.csv
    rows = (DST / "tables" / "ablation_rows.tex").read_text(encoding="utf-8").split("\\midrule\n")
    keep = 1 if DST.name == "paper_ictcs_crc" else 2
    (DST / "tables" / "ablation_rows.tex").write_text("\\midrule\n".join(rows[:keep]), encoding="utf-8")
    if DST.name == "paper_ictcs_crc":  # comparisons the text does not discuss stay in results/tables/main_tests.csv
        tests = (DST / "tables" / "tests_rows.tex").read_text(encoding="utf-8").splitlines(keepends=True)
        (DST / "tables" / "tests_rows.tex").write_text(
            "".join(l for l in tests if "vs.\\ Closed-book (Q)" not in l and "vs.\\ CoVe+RAG (Q)" not in l), encoding="utf-8")
    # Springer style: arXiv preprints are cited by their arXiv number alone (the note field), so drop the URL line
    bib = (SRC / "refs.bib").read_text(encoding="utf-8")
    bib = re.sub(r",\n  url = \{https://arxiv\.org/abs/[^}]*\}", "", bib)
    # and lists of more than three authors are shortened to three plus "et al." to keep within the page budget
    bib = re.sub(r"(  author = \{)((?:[^{}]|\{[^{}]*\})*)\}",
                 lambda m: m.group(1) + (" and ".join(m.group(2).split(" and ")[:3] + ["others"])
                                         if len(m.group(2).split(" and ")) > 3 else m.group(2)) + "}", bib)
    (DST / "refs.bib").write_text(bib, encoding="utf-8")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=DST, stdout=subprocess.DEVNULL)
    sys.exit(subprocess.run([sys.executable, str(ROOT / "code" / "check_paper.py"), "--paper", DST.name]).returncode)


if __name__ == "__main__":
    main()
