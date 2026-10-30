"""Build the ICTCS 2026 (Springer SVProc) version of the paper and run its checks.

Copies the generated table rows (paper/tables) and the verified bibliography (paper/refs.bib, without the arXiv URL
lines) into paper_ictcs/, runs pdflatex + bibtex, then code/check_paper.py --paper paper_ictcs.
Usage: .venv\\Scripts\\python.exe code\\build_ictcs.py
"""
import re
import shutil
import subprocess
import sys

from common import ROOT

SRC, DST = ROOT / "paper", ROOT / "paper_ictcs"


def main():
    (DST / "tables").mkdir(exist_ok=True)
    for f in (SRC / "tables").glob("*.tex"):
        shutil.copy2(f, DST / "tables" / f.name)
    # Springer style: arXiv preprints are cited by their arXiv number alone (the note field), so drop the URL line
    bib = (SRC / "refs.bib").read_text(encoding="utf-8")
    (DST / "refs.bib").write_text(re.sub(r",\n  url = \{https://arxiv\.org/abs/[^}]*\}", "", bib), encoding="utf-8")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=DST, stdout=subprocess.DEVNULL)
    sys.exit(subprocess.run([sys.executable, str(ROOT / "code" / "check_paper.py"), "--paper", "paper_ictcs"]).returncode)


if __name__ == "__main__":
    main()
