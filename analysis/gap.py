"""Exploratory analysis (not in the protocol): does the circuit lose less Dice from BUSI to an external set than a
control does? For models a and b, gap contrast D = (a - b on BUSI) - (a - b on the external set).
Bootstrap: the same resampled seeds for both datasets; clusters resampled independently in each dataset
(BUSI duplicate groups, external patients). B = 10,000; generator from SeedSequence(1). Plan A, all training images.

Usage: python gap.py <data folder> <output csv>
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import analysis as A

PAIRS = [("q4", "qp4"), ("q4", "s4"), ("q4", "tr4"), ("q4", "cl4"), ("q4", "full"), ("tr4", "full")]


def gap(tabs, ext, a, b, rng, n=A.B, batch=1000):
    seeds = [s for s in A.SEEDS if A.complete(tabs["busi"], "A", 1.0, a, b, s) and A.complete(tabs[ext], "A", 1.0, a, b, s)]
    S1, N1, _ = A.paired(tabs["busi"], "A", 1.0, a, b, "group", seeds)
    S2, N2, _ = A.paired(tabs[ext], "A", 1.0, a, b, "patient", seeds)
    est = S1.sum() / (len(seeds) * N1.sum()) - S2.sum() / (len(seeds) * N2.sum())
    ns = len(seeds)
    out = []
    for i in range(0, n, batch):
        k = min(batch, n - i)
        V = rng.multinomial(ns, np.full(ns, 1 / ns), size=k).astype(float)
        W1 = rng.multinomial(len(N1), np.full(len(N1), 1 / len(N1)), size=k).astype(float)
        W2 = rng.multinomial(len(N2), np.full(len(N2), 1 / len(N2)), size=k).astype(float)
        d1 = np.einsum("bs,sc,bc->b", V, S1, W1) / (V.sum(1) * (W1 @ N1))
        d2 = np.einsum("bs,sc,bc->b", V, S2, W2) / (V.sum(1) * (W2 @ N2))
        out.append(d1 - d2)
    bt = np.concatenate(out)
    lo, hi = np.percentile(bt, [2.5, 97.5])
    p = min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean()))
    return dict(external=ext, contrast=f"{a} - {b}", gap_diff=est, ci95_lo=lo, ci95_hi=hi, p=p, seeds=ns)


def main(src, out):
    _, _, tabs = A.load(src)
    rng = np.random.default_rng(np.random.SeedSequence(1))
    rows = [gap(tabs, ext, a, b, rng) for ext in ("busbra", "breast") for a, b in PAIRS]
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for ext in ("busbra", "breast"):
        m = df.external == ext
        df.loc[m, "p_holm"] = A.holm(df.loc[m, "p"].values)
    df.to_csv(out, index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
