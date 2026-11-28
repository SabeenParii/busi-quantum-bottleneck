"""Prespecified outputs that were computed but not tabulated in the first manuscript version (addendum, part D):
pixel precision and recall on BUSI (protocol Section 5), weight change of the dense layers around the slot
(Section 6.2), the 95% interval half-width of each F1 contrast as the smallest difference the design could detect
(Section 7), and the BUSI exclusion flow and difference-hash audit (Section 2.1).

Usage: python analysis/protocol_outputs.py results/data results/stats out_dir [manifest_all.csv]
"""
import os
import sys

import numpy as np
import pandas as pd

NM = {"q4": "Q$_4$", "qp4": "Qp$_4$", "s4": "S$_4$", "tr4": "Tr$_4$", "cl4": "Cl$_4$", "id4": "Id$_4$",
      "full": "Full", "unet": "U-Net"}


def main(data, stats, out, manifest_all=None):
    os.makedirs(out, exist_ok=True)
    summ = pd.read_csv(os.path.join(data, "summaries.csv"))
    t = pd.read_csv(os.path.join(data, "test_busi.csv.gz")).merge(summ[["run_id", "plan", "model", "frac"]], on="run_id")
    t = t[(t.plan == "A") & (t.frac == 1.0)]
    # precision of an empty prediction is undefined; it is counted as 0 here, so lesion misses lower precision too
    t["precision0"] = t.precision.fillna(0.0)
    pr = t.groupby(["model", "run_id"])[["precision0", "precision", "recall"]].mean().groupby("model").mean()
    a = pd.read_csv(os.path.join(stats, "audit.csv"))
    a = a[a.plan == "A"].set_index("model")[["change_pre_dense", "change_post_dense"]]
    f = pd.read_csv(os.path.join(stats, "post_hoc", "f1_adjusted_equivalence.csv"))
    f["half95"] = (f.ci95_hi - f.ci95_lo) / 2
    f["half90"] = (f.ci90_hi - f.ci90_lo) / 2
    pr.join(a).to_csv(os.path.join(out, "precision_recall_dense_change.csv"))
    f[["dataset", "contrast", "half95", "half90"]].to_csv(os.path.join(out, "f1_halfwidths.csv"), index=False)
    lines = [r"\begin{table}[!ht]", r"\centering",
             r"\caption{Prespecified outputs not shown in the main tables (plan A, BUSI, mean over 25 runs). Precision and recall "
             r"are pixel values per image; precision is undefined for an empty prediction and is given without those images "
             r"and with them counted as 0. Dense($n$) and Dense(512): relative change of the dense layers before and after the slot "
             r"from their initial values.}\label{stab:protout}",
             r"\footnotesize", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Model & Precision & Precision, empty as 0 & Recall & Dense($n$) & Dense(512) \\", r"\midrule"]
    for m in ["q4", "qp4", "s4", "tr4", "cl4", "id4", "full", "unet"]:
        r = pr.loc[m]
        d1 = f"{a.loc[m, 'change_pre_dense']:.3f}" if m in a.index and not np.isnan(a.loc[m, "change_pre_dense"]) else "--"
        d2 = f"{a.loc[m, 'change_post_dense']:.3f}" if m in a.index and not np.isnan(a.loc[m, "change_post_dense"]) else "--"
        lines.append(f"{NM[m]} & {r.precision:.3f} & {r.precision0:.3f} & {r.recall:.3f} & {d1} & {d2} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines += [r"\begin{table}[!ht]", r"\centering",
              r"\caption{Half-width of the 95\% and 90\% intervals of each F1 contrast, the smallest difference the design could "
              r"detect (protocol Section 7).}\label{stab:mdd}", r"\footnotesize", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
              r" & \multicolumn{2}{c}{BUSI} & \multicolumn{2}{c}{BUS-BRA} & \multicolumn{2}{c}{BrEaST} \\",
              r"Contrast & 95\% & 90\% & 95\% & 90\% & 95\% & 90\% \\", r"\midrule"]
    for c in ["q4 - qp4", "q4 - s4", "q4 - tr4", "q4 - cl4"]:
        cells = []
        for ds in ["busi", "busbra", "breast"]:
            r = f[(f.dataset == ds) & (f.contrast == c)].iloc[0]
            cells += [f"{r.half95:.3f}", f"{r.half90:.3f}"]
        a_, b_ = c.split(" - ")
        lines.append(f"{NM[a_]} vs {NM[b_]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if manifest_all:
        m = pd.read_csv(manifest_all)
        m["reason"] = m.reason.fillna("kept")
        ct = pd.crosstab(m.reason, m.cls)
        order = ["kept", "duplicate (kept another copy)", "cross-class duplicate", "axilla", "needle"]
        lab = {"kept": "Kept", "duplicate (kept another copy)": "Duplicate, another copy kept",
               "cross-class duplicate": "Duplicate group spanning classes", "axilla": "Axilla", "needle": "Biopsy needle"}
        lines += [r"\begin{table}[!ht]", r"\centering",
                  r"\caption{BUSI cleaning (protocol Section 2.1). No image was removed for a doubtful class label. The difference-hash "
                  r"audit (at most 4 differing bits) found no near-duplicate pair among the kept images beyond the published duplicate "
                  r"groups.}\label{stab:busiflow}", r"\footnotesize", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                  r"Outcome & Benign & Malignant & Normal & Total \\", r"\midrule"]
        for k in order:
            r = ct.loc[k] if k in ct.index else pd.Series(0, index=ct.columns)
            lines.append(f"{lab[k]} & {r.get('benign', 0)} & {r.get('malignant', 0)} & {r.get('normal', 0)} & {r.sum()} \\\\")
        lines.append(r"\midrule")
        lines.append(f"All & {ct['benign'].sum()} & {ct['malignant'].sum()} & {ct['normal'].sum()} & {ct.values.sum()} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        ct.to_csv(os.path.join(out, "busi_flow.csv"))
    open(os.path.join(out, "supp_protocol_outputs.tex"), "w").write("\n".join(lines))
    print(pr.join(a).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
