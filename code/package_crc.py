"""Package the ICTCS 2026 camera-ready copy as the organisers ask: the final PDF and the LaTeX source renamed to the
paper ID ("1080-ICTCS 2026"), in a folder of that name, zipped. The renamed .bbl lets the source compile without
BibTeX. The zip is test-compiled in a temporary folder before it is reported as ready.

Usage: .venv\\Scripts\\python.exe code\\package_crc.py      (after code\\build_ictcs.py --paper paper_ictcs_crc passes)
Output: paper_ictcs_crc/submission/1080-ICTCS 2026/, paper_ictcs_crc/submission/1080-ICTCS 2026.zip
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from common import ROOT

NAME = "1080-ICTCS 2026"
SRC = ROOT / "paper_ictcs_crc"
SUPPORT = ["refs.bib", "svproc.cls", "spmpsci.bst", "figures/phi1.pdf", "tables/main_rows.tex", "tables/tests_rows.tex",
           "tables/ablation_rows.tex", "tables/verdict_rows.tex"]


def pages(log_text):
    m = re.search(r"Output written on .*?\((\d+) pages", log_text, re.S)
    return int(m.group(1)) if m else None


def main():
    if subprocess.run([sys.executable, str(ROOT / "code" / "check_paper.py"), "--paper", SRC.name]).returncode:
        raise SystemExit("checks fail: fix the paper before packaging")
    out = SRC / "submission" / NAME
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir()
    shutil.copy2(SRC / "main.pdf", out / f"{NAME}.pdf")
    shutil.copy2(SRC / "main.tex", out / f"{NAME}.tex")
    shutil.copy2(SRC / "main.bbl", out / f"{NAME}.bbl")
    for f in SUPPORT:
        shutil.copy2(SRC / f, out / f)
    zpath = SRC / "submission" / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p for p in out.rglob("*") if p.is_file()):
            z.write(f, f"{NAME}/{f.relative_to(out).as_posix()}")
    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(zpath).extractall(d)
        work = Path(d) / NAME
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{NAME}.tex"], cwd=work, capture_output=True,
                               text=True, errors="ignore")
        n = pages(r.stdout)
        if r.returncode or n != pages((SRC / "main.log").read_text(encoding="latin-1")):
            raise SystemExit(f"test compile of the zip failed or changed the page count ({n} pages):\n{r.stdout[-1500:]}")
    print(f"ready: {zpath} ({zpath.stat().st_size // 1024} KB), test compile {n} pages")
    for f in sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()):
        print("  ", f)


if __name__ == "__main__":
    main()
