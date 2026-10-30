"""Figure: effective reliability Phi_1 (mean +/- std over the three test blocks) of the Qwen2.5-7B systems, from
results/tables/main_summary.csv. One panel per dataset; vector PDF, readable in black and white.

Usage: .venv\\Scripts\\python.exe code\\make_figure.py
Output: paper_ictcs/figures/phi1.pdf (+ .png preview)
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT

SYSTEMS = [("closed_book", "Closed-book"), ("cove", "CoVe (closed-book)"), ("rag", "Standard RAG"),
           ("cove_rag", "CoVe + RAG"), ("rag_abstain", "RAG + abstain prompt"),
           ("rv_no_verifier", "RAG + self-revision"), ("ragverify", "RAG-Verify")]
COLOR = {"ragverify": "#2a78d6", "rv_no_verifier": "#eb6834"}  # palette slots 1-2; other systems neutral gray
NEUTRAL, INK, MUTED = "#a3a3a3", "#3d3d3d", "#7a7a7a"


def main():
    rows = {(r["dataset"], r["method"]): r for r in csv.DictReader((ROOT / "results/tables/main_summary.csv").open(encoding="utf-8"))
            if r["backbone"] == "Qwen2.5-7B"}
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.7), sharey=True)
    ys = range(len(SYSTEMS))[::-1]
    for ax, (ds, title) in zip(axes, [("hotpotqa", "HotpotQA"), ("triviaqa", "TriviaQA")]):
        for y, (m, _) in zip(ys, SYSTEMS):
            r = rows[(ds, m)]
            v, sd = 100 * float(r["phi1_mean"]), 100 * float(r["phi1_std"])
            ax.barh(y, v, height=0.62, color=COLOR.get(m, NEUTRAL), zorder=2)
            ax.errorbar(v, y, xerr=sd, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=3)
            if m in COLOR:  # direct value labels only on the two systems the text compares
                ax.text(v + sd + 2 if v >= 0 else v - sd - 2, y, f"{v:.1f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=6.5, color=INK)
        ax.axvline(0, color=INK, lw=0.6, zorder=1)
        ax.grid(axis="x", color="#e2e2e2", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel(r"Effective reliability $\Phi_1$ (%)")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.5, width=0.6)
    axes[0].set_xlim(-62, 32)
    axes[1].set_xlim(0, 75)
    axes[0].set_yticks(list(ys), [label for _, label in SYSTEMS])
    fig.tight_layout(pad=0.3, w_pad=1.2)
    out = ROOT / "paper_ictcs" / "figures" / "phi1.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
