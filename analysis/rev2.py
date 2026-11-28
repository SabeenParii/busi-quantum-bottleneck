"""Review round 2: every block at the same learning-rate factor (k = 100) and a positive control (notebook 07).

Added after the protocol (addendum, part C). Same bootstrap, folds, seeds and cluster definitions as src/analysis.py.
Retrained models are labelled tr4_k100 and tr8_k100 (notebook 07) and qp4_k100 and q8_k100 (notebook 05); the
zero block is z4.

F1c  BUSI, BUS-BRA, BrEaST: q4 against qp4_k100, s4, tr4_k100 and cl4; every block has k = 100.
F4c  BUSI: q2 against qp2 and tr2 (all k = 100), q8_k100 against qp8 and tr8_k100.
PC   BUSI, BUS-BRA, BrEaST: z4 against id4, q4 and cl4. The slot is shown to be sensitive if z4 scores lower than
     id4 with a 95% interval that excludes zero.
Each family is Holm-adjusted; equivalence is shown unadjusted (90% interval) and adjusted (1 - 2 alpha / m).

Usage: python analysis/rev2.py results/data results/k100 results/rev2 results/stats/rev2
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import analysis as A  # noqa: E402
from review_extra import run  # noqa: E402

FAMS = [("F1c", "busi", "A", [("q4", "qp4_k100"), ("q4", "s4"), ("q4", "tr4_k100"), ("q4", "cl4")]),
        ("F1c_busbra", "busbra", "A", [("q4", "qp4_k100"), ("q4", "s4"), ("q4", "tr4_k100"), ("q4", "cl4")]),
        ("F1c_breast", "breast", "A", [("q4", "qp4_k100"), ("q4", "s4"), ("q4", "tr4_k100"), ("q4", "cl4")]),
        ("F4c", "busi", "B", [("q2", "qp2"), ("q2", "tr2"), ("q8_k100", "qp8"), ("q8_k100", "tr8_k100")]),
        ("PC", "busi", "A", [("z4", "id4"), ("z4", "q4"), ("z4", "cl4")]),
        ("PC_busbra", "busbra", "A", [("z4", "id4"), ("z4", "q4"), ("z4", "cl4")]),
        ("PC_breast", "breast", "A", [("z4", "id4"), ("z4", "q4"), ("z4", "cl4")])]


def add(tabs, src, rename):
    s, meta, t2 = A.load(src)
    meta = meta.copy()
    meta["model"] = meta.model.map(lambda m: rename.get(m, m))
    meta["run_id"] = meta.run_id + "_" + os.path.basename(os.path.normpath(src))
    for ds, t in t2.items():
        t = t.drop(columns=["plan", "model", "fold", "seed", "frac"])
        t["run_id"] = t.run_id + "_" + os.path.basename(os.path.normpath(src))
        tabs[ds] = pd.concat([tabs[ds], t.merge(meta, on="run_id")], ignore_index=True)
    s = s.copy()
    s["model"] = s.model.map(lambda m: rename.get(m, m))
    return s


def main(data, k100, rev2, out):
    os.makedirs(out, exist_ok=True)
    summ, meta, tabs = A.load(data)
    sk = add(tabs, k100, {"qp4": "qp4_k100", "q8": "q8_k100"})
    sr = add(tabs, rev2, {"tr4": "tr4_k100", "tr8": "tr8_k100"})
    ss = np.random.SeedSequence(2028).spawn(len(FAMS))
    res = []
    for (fam, ds, plan, pairs), sq in zip(FAMS, ss):
        r = run(tabs[ds], plan, 1.0, pairs, "group" if ds == "busi" else "patient", np.random.default_rng(sq), fam, ds)
        r["verdict"] = [A.verdict(x, False) for _, x in r.iterrows()]
        res.append(r)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(os.path.join(out, "rev2_contrasts.csv"), index=False)
    allsum = pd.concat([summ, sk, sr], ignore_index=True)
    lv = allsum[allsum.model.isin(["q4", "qp4_k100", "s4", "tr4", "tr4_k100", "cl4", "id4", "z4", "q8_k100", "qp8",
                                   "tr8", "tr8_k100"]) & (allsum.frac == 1.0)]
    lv = lv.groupby("model").agg(runs=("run_id", "size"), k=("k", "first"), dice_busi=("test_dice", "mean"),
                                 dice_busbra=("busbra_dice", "mean"), dice_breast=("breast_dice", "mean"),
                                 change_block=("change_block", "median"), epochs=("epochs", "median"))
    lv.to_csv(os.path.join(out, "rev2_levels.csv"))
    pd.set_option("display.width", 220)
    print(res[["family", "contrast", "diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "ci_adj_lo", "ci_adj_hi",
               "p_holm", "equiv_protocol", "equiv_adjusted", "verdict"]].round(4).to_string(index=False))
    print(lv.round(4).to_string())




NM = {"q4": "Q$_4$", "qp4_k100": "Qp$_4$", "s4": "S$_4$", "tr4_k100": "Tr$_4$", "cl4": "Cl$_4$", "id4": "Id$_4$",
      "z4": "Z$_4$", "q2": "Q$_2$", "qp2": "Qp$_2$", "tr2": "Tr$_2$", "q8_k100": "Q$_8$", "qp8": "Qp$_8$", "tr8_k100": "Tr$_8$"}
DS = {"busi": "BUSI", "busbra": "BUS-BRA", "breast": "BrEaST"}


def tex(out):
    r = pd.read_csv(os.path.join(out, "rev2_contrasts.csv"))
    f = lambda x: (f"{x:.4f}" if abs(x) < 0.0005 and x != 0 else f"{x:.3f}").replace("-", "$-$")
    pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    lines = [r"\begin{table}[!ht]", r"\centering",
             r"\caption{Exploratory runs added after the protocol. Top: every block of F1 and F4 with the same learning-rate "
             r"factor $k=100$ (Qp$_4$ and Q$_8$ from Table S11; Tr$_4$ and Tr$_8$ retrained). Bottom: positive control Z$_4$, "
             r"a slot whose output is zero, so the decoder receives no code. Holm-adjusted within each family and dataset; "
             r"equivalence with the 90\% interval inside $\pm0.02$, also adjusted with the $1-2\alpha/m$ interval. The contrasts with "
             r"Qp$_4$, S$_4$, Cl$_4$ and Q$_8$ repeat contrasts of Tables 4 and S11 in new families with new bootstrap draws, so "
             r"their limits can differ from those tables by 0.001 to 0.002 and their Holm $p$ values by more.}\label{stab:rev2}",
             r"\footnotesize", r"\begin{tabular}{@{}llrccrl@{}}", r"\toprule",
             r"Set & Contrast & Difference & 95\% interval & 90\% interval & Holm $p$ & Verdict \\", r"\midrule"]
    for fam in ["F1c", "F1c_busbra", "F1c_breast", "F4c", "PC", "PC_busbra", "PC_breast"]:
        if fam == "PC":
            lines.append(r"\midrule")
        for _, x in r[r.family == fam].iterrows():
            a, b = x.contrast.split(" - ")
            v = x.verdict.replace("difference (higher)", "Higher").replace("difference (lower)", "Lower").replace("inconclusive", "Inconclusive")
            if x.equiv_protocol:
                v = "Equivalent, " + ("also adjusted" if x.equiv_adjusted else "unadjusted only")
            lines.append(f"{DS[x.dataset]} & {NM[a]} vs {NM[b]} & {f(x['diff'])} & {f(x.ci95_lo)} to {f(x.ci95_hi)} & "
                         f"{f(x.ci90_lo)} to {f(x.ci90_hi)} & {pf(x.p_holm)} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(out, "supp_rev2.tex"), "w").write("\n".join(lines))



if __name__ == "__main__":
    if len(sys.argv) > 5 and sys.argv[5] == "tex":
        tex(sys.argv[4])
    else:
        main(*sys.argv[1:5])
