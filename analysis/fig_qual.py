"""Figure of qualitative examples from analysis/qualitative_examples.npz (notebook 04).

Usage: python analysis/fig_qual.py qualitative_examples.npz qualitative_runs.csv out.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ("q4", "qp4", "tr4", "cl4", "id4", "full", "unet")
LAB = {"q4": r"Q$_4$", "qp4": r"Qp$_4$", "tr4": r"Tr$_4$", "cl4": r"Cl$_4$", "id4": r"Id$_4$", "full": "Full", "unet": "U-Net"}
ROWS = (("busi", "BUSI"), ("busbra", "BUS-BRA"), ("breast", "BrEaST"), ("normal", "BUSI normal"))
REF, PRED = "#ffd400", "#00b4ff"


def dice(r, p):
    s = r.sum() + p.sum()
    return 1.0 if s == 0 else 2 * (r & p).sum() / s


def main(npz, runs_csv, out):
    z = np.load(npz)
    runs = pd.read_csv(runs_csv).set_index("model")
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(len(ROWS), len(MODELS), figsize=(6.3, 3.45),
                           gridspec_kw=dict(wspace=0.03, hspace=0.06))
    for i, (ds, name) in enumerate(ROWS):
        img = z[f"img_{ds}"].astype(float)
        ref = z[f"ref_{ds}"].astype(bool)
        for j, m in enumerate(MODELS):
            a = ax[i, j]
            pr = z[f"pred_{m}_{ds}"].astype(bool)
            a.imshow(img, cmap="gray", interpolation="bilinear")
            if ref.any():
                a.contour(ref, levels=[0.5], colors=REF, linewidths=0.9)
            if pr.any():
                a.contour(pr, levels=[0.5], colors=PRED, linewidths=0.9)
            if ds == "normal":
                txt = "no mask" if not pr.any() else f"{int(pr.sum())} px"
            else:
                d = dice(ref, pr)
                stored = runs.loc[m, f"dice_{ds}_example"]
                assert abs(d - stored) < 1e-6, (m, ds, d, stored)
                txt = f"{d:.2f}"
            a.text(0.03, 0.04, txt, transform=a.transAxes, color="white", fontsize=7,
                   bbox=dict(facecolor="black", alpha=0.55, pad=1.0, edgecolor="none"))
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                a.set_title(LAB[m], fontsize=8, pad=2)
            if j == 0:
                a.set_ylabel(name, fontsize=8)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main(*sys.argv[1:4])
