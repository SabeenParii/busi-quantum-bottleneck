"""Exploratory sensitivity analysis (added after the protocol): Q4 vs Qp4 and Q8 vs Qp8 / Tr8 when the blocks that the
selection runs gave k = 1 (Qp4, Q8) are retrained with k = 100 (notebook 05_equal_k). Same contrasts, crossed bootstrap
and generator seeds as src/analysis.py; the retrained models are labelled qp4_k100 and q8_k100.

Usage: python analysis/equal_k.py results/data results/k100 results/stats/equal_k_contrasts.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import analysis as A  # noqa: E402

PAIRS = [("F1k", "busi", "A", [("q4", "qp4_k100"), ("q4", "qp4")]),
         ("F1k_busbra", "busbra", "A", [("q4", "qp4_k100")]),
         ("F1k_breast", "breast", "A", [("q4", "qp4_k100")]),
         ("F4k", "busi", "B", [("q8_k100", "qp8"), ("q8_k100", "tr8"), ("q8_k100", "q8")])]


def main(data, k100, out):
    summ, meta, tabs = A.load(data)
    ks, kmeta, ktabs = A.load(k100)
    ks["model"] = ks.model + "_k100"
    kmeta = kmeta.copy(); kmeta["model"] = kmeta.model + "_k100"
    for ds, t in ktabs.items():
        t = t.drop(columns=["plan", "model", "fold", "seed", "frac"]).merge(kmeta, on="run_id")
        t["run_id"] = t.run_id + "_k100"
        tabs[ds] = pd.concat([tabs[ds], t], ignore_index=True)
    ss = np.random.SeedSequence(3).spawn(len(PAIRS))
    rows = []
    for (fam, ds, plan, pairs), sq in zip(PAIRS, ss):
        rng = np.random.default_rng(sq)
        cl = "group" if ds == "busi" else "patient"
        res = [dict(family=fam, dataset=ds, plan=plan, **A.contrast(tabs[ds], plan, 1.0, a, b, cl, rng, False)) for a, b in pairs]
        ph = A.holm([r["p"] for r in res])
        for r, h in zip(res, ph):
            r["p_holm"] = h
            r["verdict"] = A.verdict(r, False)
        rows += res
    r = pd.DataFrame(rows)
    r.to_csv(out, index=False)
    print(r[["family", "dataset", "contrast", "diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "p_holm", "verdict"]].round(4).to_string(index=False))
    lv = ks.groupby("model").agg(runs=("run_id", "size"), k=("k", "first"), dice=("test_dice", "mean"),
                                 change_block=("change_block", "median"), drop_const=("drop_const", "mean"),
                                 drop_swap=("drop_swap", "mean"), mw=("entanglement_mw", "mean"),
                                 r2=("surrogate_r2", "min"))
    print(lv.round(4).to_string())
    lv.to_csv(out.replace("contrasts", "levels"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
