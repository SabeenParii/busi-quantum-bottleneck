"""Exploratory analysis (added after the protocol, which fixed only 1,024 shots without hardware noise):
Dice of the trained Q4 models of plan A (25 runs) under simulated hardware noise and finite shots, against the
same models evaluated exactly. Levels (src/qnoise.LEVELS): depolarizing p1 after single-qubit gates, two-qubit
depolarizing p2 after each CNOT, amplitude damping gamma, readout flip pm. Shots: 1,024, 4,096, 8,192, exact.
Contrast = condition - (none, exact) on the same images and seeds; crossed bootstrap over BUSI duplicate groups and
seeds (B = 10,000; generator SeedSequence(2)); equivalence if the 95% CI lies within +-0.02 Dice.

Usage: python analysis/noise.py analysis/noise_q4.csv.gz analysis/noise_contrasts.csv
"""
import sys
import numpy as np
import pandas as pd

B = 10_000


def main(src, out):
    d = pd.read_csv(src, dtype={"shots": str})
    ref = d[(d.level == "none") & (d.shots == "exact")]
    rng = np.random.default_rng(np.random.SeedSequence(2))
    seeds = sorted(ref.seed.unique())
    groups = sorted(ref.group.unique(), key=str)
    gi = {g: i for i, g in enumerate(groups)}
    rows = []
    for (lev, sh), t in d.groupby(["level", "shots"], sort=False):
        m = t.merge(ref, on=["seed", "fold", "image", "group"], suffixes=("", "_ref"))
        assert len(m) == len(ref)
        m["dd"] = m.dice - m.dice_ref
        S = np.zeros((len(seeds), len(groups))); N = np.zeros(len(groups))
        for c, n in m[m.seed == seeds[0]].groupby("group").size().items():
            N[gi[c]] = n
        for si, s in enumerate(seeds):
            for c, v in m[m.seed == s].groupby("group").dd.sum().items():
                S[si, gi[c]] = v
        est = S.sum() / (len(seeds) * N.sum())
        bt = []
        for i in range(0, B, 1000):
            W = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=1000).astype(float)
            V = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=1000).astype(float)
            bt.append(np.einsum("bs,sc,bc->b", V, S, W) / (V.sum(1) * (W @ N)))
        bt = np.concatenate(bt)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        rows.append(dict(level=lev, shots=sh, dice=t.dice.mean(), diff=est, ci95_lo=lo, ci95_hi=hi,
                         equivalent=bool(-0.02 < lo and hi < 0.02), images=int(N.sum()), seeds=len(seeds)))
    r = pd.DataFrame(rows)
    order = {"none": 0, "low": 1, "mid": 2, "high": 3}
    r = r.sort_values(["level", "shots"], key=lambda c: c.map(order) if c.name == "level" else c.map({"1024": 0, "4096": 1, "8192": 2, "exact": 3}))
    r.to_csv(out, index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
