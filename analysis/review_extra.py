"""Additional analyses of the stored per-image results (not in the protocol; reported as post hoc).

Usage: python analysis/review_extra.py results/data results/stats/post_hoc

1. Multiplicity-adjusted equivalence for family F1 on each dataset: Bonferroni-type intervals at 1 - 2*alpha/m
   (97.5% for m = 4 contrasts) and Holm-adjusted TOST p values. The bootstrap draws are exactly those of the
   main analysis (same random streams as src/analysis.py), so the 95% and 90% intervals reproduce Table 4.
2. Every block against the identity slot (Q4, Qp4, S4, Tr4, Cl4 minus Id4) on BUSI, BUS-BRA and BrEaST,
   Holm-adjusted within each dataset, with the verdict rule of the protocol and the adjusted equivalence.
3. Image-level detection for plan A: sensitivity (lesion images with any predicted pixel), specificity on the
   72 normal BUSI images (no predicted pixel), and BUSI Dice including the normal images (empty on normal = 1).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import analysis as A  # noqa: E402

ALPHA, M = 0.05, 4


def stats(bt, est, m):
    lo95, hi95 = np.percentile(bt, [2.5, 97.5])
    lo90, hi90 = np.percentile(bt, [5, 95])
    q = 100 * ALPHA / m
    loa, hia = np.percentile(bt, [q, 100 - q])
    p = min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean()))
    p_tost = max((bt <= -A.MARGIN).mean(), (bt >= A.MARGIN).mean())
    return dict(diff=est, ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90, ci_adj_lo=loa, ci_adj_hi=hia,
                p=p, p_tost=p_tost)


def run(tab, plan, frac, pairs, cluster, rng, fam, ds):
    rows = []
    for a, b in pairs:
        seeds = [s for s in A.SEEDS if A.complete(tab, plan, frac, a, b, s)]
        S, N, _ = A.paired(tab, plan, frac, a, b, cluster, seeds)
        est = S.sum() / (len(seeds) * N.sum())
        bt = A.boot(S, N, rng)
        rows.append(dict(family=fam, dataset=ds, contrast=f"{a} - {b}", **stats(bt, est, len(pairs))))
    r = pd.DataFrame(rows)
    r["p_holm"] = A.holm(r.p.values)
    r["p_tost_holm"] = A.holm(r.p_tost.values)
    r["equiv_protocol"] = (r.ci90_lo > -A.MARGIN) & (r.ci90_hi < A.MARGIN)
    r["equiv_adjusted"] = (r.ci_adj_lo > -A.MARGIN) & (r.ci_adj_hi < A.MARGIN)
    r["equiv_tost_holm"] = r.p_tost_holm < ALPHA
    return r


def main(src, out):
    os.makedirs(out, exist_ok=True)
    summ, meta, tabs = A.load(src)
    # 1. F1 with the random streams of the main analysis
    ss = np.random.SeedSequence(0).spawn(len(A.FAMILIES))
    f1 = []
    for (fam, ds, plan, frac, pairs), sq in zip(A.FAMILIES, ss):
        if fam in ("F1", "F1_busbra", "F1_breast"):
            f1.append(run(tabs[ds], plan, frac, pairs, "group" if ds == "busi" else "patient", np.random.default_rng(sq), fam, ds))
    f1 = pd.concat(f1, ignore_index=True)
    f1.to_csv(os.path.join(out, "f1_adjusted_equivalence.csv"), index=False)
    # 2. blocks against the identity slot (new random streams)
    pairs = [(m, "id4") for m in ("q4", "qp4", "s4", "tr4", "cl4")]
    ss2 = np.random.SeedSequence(2027).spawn(3)
    fid = pd.concat([run(tabs[ds], "A", 1.0, pairs, "group" if ds == "busi" else "patient", np.random.default_rng(sq), "FId", ds)
                     for ds, sq in zip(("busi", "busbra", "breast"), ss2)], ignore_index=True)
    fid.to_csv(os.path.join(out, "identity_contrasts.csv"), index=False)
    # 3. detection
    nrm = pd.read_csv(os.path.join(src, "normals_busi.csv.gz")).merge(meta, on="run_id")
    nrm = nrm[nrm.plan == "A"]
    rows = []
    for model in sorted(nrm.model.unique()):
        r = dict(model=model)
        for ds in ("busi", "busbra", "breast"):
            t = tabs[ds]; t = t[(t.plan == "A") & (t.model == model)]
            r[f"sens_{ds}"] = 1 - t["empty"].astype(float).mean()
        n = nrm[nrm.model == model]
        r["spec_busi"] = (n.pred_pixels == 0).mean()
        t = tabs["busi"]; t = t[(t.plan == "A") & (t.model == model)]
        per_run = t.groupby("run_id").dice.agg(["sum", "count"])
        nr = n.assign(d=(n.pred_pixels == 0).astype(float)).groupby("run_id").d.agg(["sum", "count"])
        j = per_run.join(nr, rsuffix="_n")
        r["dice_busi_with_normals"] = ((j["sum"] + j["sum_n"]) / (j["count"] + j["count_n"])).mean()
        r["dice_busi"] = (j["sum"] / j["count"]).mean()
        rows.append(r)
    det = pd.DataFrame(rows)
    det.to_csv(os.path.join(out, "detection_planA.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["dataset", "contrast", "diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "ci_adj_lo", "ci_adj_hi", "p_holm",
            "p_tost_holm", "equiv_protocol", "equiv_adjusted", "equiv_tost_holm"]
    print(f1[cols].round(4).to_string(index=False)); print(fid[cols].round(4).to_string(index=False)); print(det.round(3).to_string(index=False))




# ------------------------------------------------------------------ LaTeX tables S13 to S15
NM = {'q4': 'Q$_4$', 'qp4': 'Qp$_4$', 's4': 'S$_4$', 'tr4': 'Tr$_4$', 'cl4': 'Cl$_4$', 'id4': 'Id$_4$', 'full': 'Full', 'unet': 'U-Net'}
DS = {'busi': 'BUSI', 'busbra': 'BUS-BRA', 'breast': 'BrEaST'}


def f3(x):
    d = 4 if (x != 0 and abs(x) < 0.0005) or abs(abs(x) - 0.02) < 0.0005 else 3
    return f"{x:.{d}f}".replace("-", "$-$")


def pf(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pair(c):
    a, b = [x.strip() for x in c.split("-")]
    return f"{NM[a]} vs {NM[b]}"


def longtab(label, cap, head, body, spec):
    return "\n".join([r"\begin{longtable}{" + spec + "}", r"\caption{" + cap + r"}\label{" + label + r"}\\", r"\toprule", head + r" \\",
                      r"\midrule", r"\endfirsthead", r"\toprule", head + r" \\", r"\midrule", r"\endhead"] + body +
                     [r"\bottomrule", r"\end{longtable}", ""])


def tex(out):
    f1 = pd.read_csv(os.path.join(out, "f1_adjusted_equivalence.csv"))
    yn = lambda b: "Yes" if b else "No"
    body = [f"{DS[r.dataset]} & {pair(r.contrast)} & {f3(r['diff'])} & {f3(r.ci90_lo)} to {f3(r.ci90_hi)} & {f3(r.ci_adj_lo)} to {f3(r.ci_adj_hi)} & "
            f"{pf(r.p_tost_holm)} & {yn(r.equiv_protocol)} & {yn(r.equiv_adjusted)} \\\\" for _, r in f1.iterrows()]
    s13 = longtab("stab:adjeq", "Equivalence in family F1 with and without adjustment for multiplicity (post hoc). Protocol: 90\\% interval inside $\\pm0.02$. "
                  "Adjusted: 97.5\\% interval ($1-2\\alpha/m$ with $m=4$) inside $\\pm0.02$. $p_{\\mathrm{TOST}}$: Holm-adjusted $p$ value of the two one-sided tests. "
                  "Same bootstrap draws as Table 4 of the main text.",
                  r"Set & Contrast & Difference & 90\% interval & Adjusted interval & $p_{\mathrm{TOST}}$ & Protocol & Adjusted",
                  body, "@{}llrccrll@{}")
    fid = pd.read_csv(os.path.join(out, "identity_contrasts.csv"))
    def v(r):
        out_ = []
        if r.p_holm < 0.05: out_.append("Higher" if r["diff"] > 0 else "Lower")
        if r.equiv_protocol: out_.append("Equivalent" + ("" if r.equiv_adjusted else " (unadjusted)"))
        return ", ".join(out_) if out_ else "Inconclusive"
    body = [f"{DS[r.dataset]} & {pair(r.contrast)} & {f3(r['diff'])} & {f3(r.ci95_lo)} to {f3(r.ci95_hi)} & {pf(r.p_holm)} & {v(r)} \\\\" for _, r in fid.iterrows()]
    s14 = longtab("stab:identity", "Each block against the identity slot Id$_4$ (plan A, post hoc). Holm correction within each dataset; verdicts as in Eq.~(2) of the main text. "
                  "Unadjusted: equivalent only without adjustment for multiplicity.",
                  r"Set & Contrast & Difference & 95\% interval & $p$ & Verdict", body, "@{}llrcrl@{}")
    det = pd.read_csv(os.path.join(out, "detection_planA.csv")).set_index("model").reindex(["q4", "qp4", "s4", "tr4", "cl4", "id4", "full", "unet"])
    body = [f"{NM[m]} & {100*r.sens_busi:.1f} & {100*r.sens_busbra:.1f} & {100*r.sens_breast:.1f} & {100*r.spec_busi:.1f} & {r.dice_busi:.3f} & {r.dice_busi_with_normals:.3f} \\\\"
            for m, r in det.iterrows()]
    s15 = longtab("stab:detect", "Image-level detection for plan A (post hoc, mean over 25 runs). Sensitivity: lesion images with any predicted lesion pixel. "
                  "Specificity: normal BUSI images without any predicted pixel. Dice with normals: mean over the 457 lesion and 72 normal BUSI images, "
                  "where an empty prediction on a normal image scores 1 and any prediction on it scores 0.",
                  r"Model & \multicolumn{3}{c}{Sensitivity (\%)} & Specificity (\%) & \multicolumn{2}{c}{BUSI Dice} \\ & BUSI & BUS-BRA & BrEaST & BUSI normals & Lesions & With normals",
                  body, "@{}lrrrrrr@{}")
    open(os.path.join(out, "supp_posthoc.tex"), "w").write("\n".join([s13, s14, s15]))


if __name__ == "__main__":
    if sys.argv[3:] != ["tex"]:
        main(sys.argv[1], sys.argv[2])
    tex(sys.argv[2])
