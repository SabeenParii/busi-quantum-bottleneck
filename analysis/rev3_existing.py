"""Second addendum, part G: analyses of existing runs (post hoc, reported as such).

G1  Z4 and Id4 against Full on lesion Dice, with the F2 noninferiority rule (BUSI, BUS-BRA, BrEaST).
G2  Dice of lesion and normal BUSI images together (empty prediction on a normal image = 1, otherwise 0; each normal
    image averaged over the five fold models of a seed), crossed bootstrap, for the F2 contrasts and Z4/Id4 - Full.
G3  Nested bootstrap on BUSI for F1 and F2: seeds, then fold models within a seed, then images within a fold.
G5  False alarms on normal images with any pixel and with more than 500 pixels (Table S15).
G4  Difference each F1 contrast detects with 80% power: 2.80 x the standard error of the crossed bootstrap.

Usage: python analysis/rev3_existing.py results/data results/rev2 results/stats/rev3
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import analysis as A  # noqa: E402
from rev2 import add  # noqa: E402

F1 = [("q4", "qp4"), ("q4", "s4"), ("q4", "tr4"), ("q4", "cl4")]
F2 = [("tr4", "full"), ("q4", "full"), ("tr4", "unet"), ("q4", "unet")]
G1 = [("z4", "full"), ("id4", "full")]
DS = ("busi", "busbra", "breast")


def summarize(bt, est, ni=False):
    lo95, hi95 = np.percentile(bt, [2.5, 97.5])
    lo90, hi90 = np.percentile(bt, [5, 95])
    r = dict(diff=est, ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90, se=bt.std(ddof=1),
             p=min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean())))
    if ni:
        r["p_noninf"] = (bt <= -A.MARGIN).mean()
    return r


def crossed(tab, pairs, cluster, rng, fam, ds, ni=False):
    rows = []
    for a, b in pairs:
        seeds = [s for s in A.SEEDS if A.complete(tab, "A", 1.0, a, b, s)]
        S, N, _ = A.paired(tab, "A", 1.0, a, b, cluster, seeds)
        est = S.sum() / (len(seeds) * N.sum())
        rows.append(dict(family=fam, dataset=ds, contrast=f"{a} - {b}", **summarize(A.boot(S, N, rng), est, ni)))
    r = pd.DataFrame(rows)
    r["p_holm"] = A.holm(r.p.values)
    if ni:
        r["p_noninf_holm"] = A.holm(r.p_noninf.values)
    return r


def nested(tab, a, b, rng, nb=A.B, batch=500):
    """BUSI only: every image has one test fold. Seeds -> fold models (folds) within seed -> images within fold."""
    t = tab[(tab.plan == "A") & (tab.frac == 1.0) & tab.model.isin([a, b])]
    w = t.pivot_table(index=["seed", "fold", "image"], columns="model", values="dice").dropna()
    w["d"] = w[a] - w[b]
    seeds, folds = sorted(w.index.unique(0)), sorted(w.index.unique(1))
    n = {f: w.xs((seeds[0], f), level=(0, 1)).shape[0] for f in folds}
    nmax = max(n.values())
    D = np.zeros((len(seeds), len(folds), nmax))
    M = np.zeros((len(folds), nmax))
    for si, s in enumerate(seeds):
        for fi, f in enumerate(folds):
            v = w.xs((s, f), level=(0, 1)).sort_index().d.values
            D[si, fi, :len(v)] = v
            M[fi, :len(v)] = 1
    nf = np.array([n[f] for f in folds])
    est = D.sum() / (len(seeds) * nf.sum())
    out = []
    for i in range(0, nb, batch):
        k = min(batch, nb - i)
        si = rng.integers(0, len(seeds), (k, len(seeds)))                       # seeds
        fi = rng.integers(0, len(folds), (k, len(seeds), len(folds)))           # fold models within seed
        u = rng.random((k, len(seeds), len(folds), nmax))
        ii = np.floor(u * nf[fi][..., None]).astype(int)                        # images within fold
        valid = np.arange(nmax)[None, None, None, :] < nf[fi][..., None]
        vals = D[si[..., None, None], fi[..., None], ii]
        out.append((vals * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)))
    return est, np.concatenate(out)


def main(data, rev2, out):
    os.makedirs(out, exist_ok=True)
    summ, meta, tabs = A.load(data)
    add(tabs, rev2, {"tr4": "tr4_k100", "tr8": "tr8_k100"})
    ss = np.random.SeedSequence(2031).spawn(10)
    res = []
    # G1
    for ds, sq in zip(DS, ss[:3]):
        res.append(crossed(tabs[ds], G1, "group" if ds == "busi" else "patient", np.random.default_rng(sq), "G1", ds, ni=True))
    # G2: lesion + normal images
    s1, m1, _ = A.load(data)
    s2 = pd.read_csv(os.path.join(rev2, "summaries.csv"))
    m2 = s2[["run_id", "plan", "model", "fold", "seed", "frac"]].assign(
        run_id=lambda x: x.run_id + "_" + os.path.basename(os.path.normpath(rev2)),
        model=lambda x: x.model.replace({"tr4": "tr4_k100", "tr8": "tr8_k100"}))
    meta2 = pd.concat([m1, m2], ignore_index=True)
    nrm = pd.concat([pd.read_csv(os.path.join(data, "normals_busi.csv.gz")),
                     pd.read_csv(os.path.join(rev2, "normals_busi.csv.gz")).assign(
                         run_id=lambda x: x.run_id + "_" + os.path.basename(os.path.normpath(rev2)))], ignore_index=True)
    nrm = nrm.merge(meta2, on="run_id")
    nrm = nrm[(nrm.plan == "A") & (nrm.frac == 1.0)]
    nrm = nrm.assign(dice=(nrm.pred_pixels == 0).astype(float), group="normal_" + nrm.image.astype(str))
    les = tabs["busi"][(tabs["busi"].plan == "A") & (tabs["busi"].frac == 1.0)]
    joint = pd.concat([les[["image", "group", "dice", "plan", "model", "fold", "seed", "frac"]],
                       nrm[["image", "group", "dice", "plan", "model", "fold", "seed", "frac"]]], ignore_index=True)
    res.append(crossed(joint, F2 + G1, "group", np.random.default_rng(ss[3]), "G2", "busi_with_normals", ni=True))
    lv = joint.groupby(["model", "seed", "image"]).dice.mean().groupby(["model", "seed"]).mean().groupby("model").mean()
    lv.rename("dice_with_normals").to_csv(os.path.join(out, "g2_levels.csv"))
    # G5: share of normal images with any predicted pixel and with more than 500 predicted pixels (threshold chosen
    # after the runs), including Z4, for Supplementary Table S15
    nrm.groupby("model").pred_pixels.agg(any_pixel=lambda x: (x > 0).mean(), over_500=lambda x: (x > 500).mean()).to_csv(
        os.path.join(out, "g5_normals_false_alarms.csv"))
    # G3: nested vs crossed on BUSI
    g3 = []
    rng = np.random.default_rng(ss[4])
    for fam, pairs in (("F1", F1), ("F2", F2)):
        for a, b in pairs:
            est, bt = nested(tabs["busi"], a, b, rng)
            g3.append(dict(family="G3_" + fam, dataset="busi", contrast=f"{a} - {b}", **summarize(bt, est, fam == "F2")))
    g3 = pd.DataFrame(g3)
    for fam in ("G3_F1", "G3_F2"):
        i = g3.family == fam
        g3.loc[i, "p_holm"] = A.holm(g3.loc[i, "p"].values)
        if fam == "G3_F2":
            g3.loc[i, "p_noninf_holm"] = A.holm(g3.loc[i, "p_noninf"].values)
    res.append(g3)
    # G4: crossed bootstrap SE of F1 with the main random streams
    ssm = np.random.SeedSequence(0).spawn(len(A.FAMILIES))
    g4 = []
    for (fam, ds, plan, frac, pairs), sq in zip(A.FAMILIES, ssm):
        if fam in ("F1", "F1_busbra", "F1_breast"):
            r = crossed(tabs[ds], pairs, "group" if ds == "busi" else "patient", np.random.default_rng(sq), "G4", ds)
            r["halfwidth95"] = (r.ci95_hi - r.ci95_lo) / 2
            r["mdd80"] = 2.80 * r.se
            g4.append(r)
    res.append(pd.concat(g4, ignore_index=True))
    res = pd.concat(res, ignore_index=True)
    res["equiv_90"] = (res.ci90_lo > -A.MARGIN) & (res.ci90_hi < A.MARGIN)
    res.to_csv(os.path.join(out, "rev3_existing.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["family", "dataset", "contrast", "diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "se", "p_holm",
            "p_noninf_holm", "halfwidth95", "mdd80", "equiv_90"]
    print(res[[c for c in cols if c in res]].round(4).to_string(index=False))
    print(lv.round(4))




NM = {"q4": "Q$_4$", "qp4": "Qp$_4$", "s4": "S$_4$", "tr4": "Tr$_4$", "cl4": "Cl$_4$", "id4": "Id$_4$", "z4": "Z$_4$",
      "full": "Full", "unet": "U-Net"}
DSN = {"busi": "BUSI", "busbra": "BUS-BRA", "breast": "BrEaST", "busi_with_normals": "BUSI"}


def tex(out):
    """Table S19 (half-widths and the difference detected with 80% power) and Table S22 (G1 to G3)."""
    r = pd.read_csv(os.path.join(out, "rev3_existing.csv"))
    f = lambda x: f"{x:.3f}".replace("-", "$-$")
    pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    name = lambda c: " vs ".join(NM[x.strip()] for x in c.split(" - "))
    g4 = r[r.family == "G4"]
    L = [r"\begin{table}[!ht]", r"\centering",
         r"\caption{Precision of the F1 contrasts. Half-width: half the width of the 95\% interval, which the protocol "
         r"(Section 7) called the smallest difference the design could detect; a true difference of that size is detected "
         r"only about half of the time. 80\% power: the difference detected with 80\% power at the 5\% level, 2.80 times the "
         r"standard error of the crossed bootstrap (post hoc).}\label{stab:mdd}", r"\footnotesize",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{BUSI} & \multicolumn{2}{c}{BUS-BRA} & \multicolumn{2}{c}{BrEaST} \\",
         r"Contrast & Half-width & 80\% power & Half-width & 80\% power & Half-width & 80\% power \\", r"\midrule"]
    for c in ["q4 - qp4", "q4 - s4", "q4 - tr4", "q4 - cl4"]:
        cells = []
        for ds in ("busi", "busbra", "breast"):
            x = g4[(g4.dataset == ds) & (g4.contrast == c)].iloc[0]
            cells += [f(x.halfwidth95), f(x.mdd80)]
        L.append(name(c) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    s19 = "\n".join(L)

    def verdict(x):
        v = []
        if x.p_holm < 0.05:
            v.append("Higher" if x["diff"] > 0 else "Lower")
        if x.equiv_90:
            v.append("equivalent")
        if "p_noninf_holm" in x and pd.notna(x.p_noninf_holm) and x.p_noninf_holm < 0.05:
            v.append("noninferior")
        s = ", ".join(v) if v else "Inconclusive"
        return s[0].upper() + s[1:]
    L = [r"\begin{longtable}{@{}lllrccrl@{}}",
         r"\caption{Post hoc analyses of existing runs (second addendum, part G). (a) The zero slot Z$_4$ and the identity "
         r"slot Id$_4$ against Full on lesion Dice, with the noninferiority rule of F2. (b) BUSI Dice over the 457 lesion and "
         r"72 normal images, where an empty prediction on a normal image scores 1 and any other prediction 0. (c) Nested "
         r"bootstrap on BUSI that resamples seeds, then fold models within a seed, then images within a fold. Holm correction "
         r"within each part and dataset; equivalence with the 90\% interval inside $\pm0.02$, unadjusted.}\label{stab:rev3old}\\",
         r"\toprule", r"Part & Set & Contrast & Difference & 95\% interval & 90\% interval & Holm $p$ & Verdict \\",
         r"\midrule", r"\endfirsthead", r"\toprule",
         r"Part & Set & Contrast & Difference & 95\% interval & 90\% interval & Holm $p$ & Verdict \\", r"\midrule", r"\endhead"]
    for fam, part in (("G1", "a"), ("G2", "b"), ("G3_F1", "c"), ("G3_F2", "c")):
        for _, x in r[r.family == fam].iterrows():
            L.append(f"({part}) & {DSN[x.dataset]} & {name(x.contrast)} & {f(x['diff'])} & {f(x.ci95_lo)} to {f(x.ci95_hi)} & "
                     f"{f(x.ci90_lo)} to {f(x.ci90_hi)} & {pf(x.p_holm)} & {verdict(x)} \\\\")
        if fam != "G3_F2":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{longtable}", ""]
    open(os.path.join(out, "supp_rev3_existing.tex"), "w").write(s19 + "\n" + "\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[4] == "tex":
        tex(sys.argv[3])
    else:
        main(*sys.argv[1:4])
