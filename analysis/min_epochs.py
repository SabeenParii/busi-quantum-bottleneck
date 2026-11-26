"""Sensitivity analysis for the 23 plan C runs whose first training stopped before epoch 30 (notebook 06_min_epochs.ipynb).

The protocol requires at least 30 epochs. In 23 plan C runs of the second wave this minimum was not in effect, so these runs
were retrained with the protocol settings; the retrained runs are part of results/data and of every reported result.
The original runs are kept in results/original23/.

Usage: python min_epochs.py <results/data> <results/original23> <output folder>

Puts the original runs back in place of the retrained ones and recomputes every F3 family with the same code, bootstrap and
random numbers as the main analysis. Writes the reported and the original-run contrasts side by side
(f3_reported_vs_original.csv, levels_planC.csv) and Supplementary Table S12 (tab_min30.tex).
"""
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import analysis  # noqa: E402


def f3(x):
    d = 4 if (x != 0 and abs(x) < 0.0005) or abs(abs(x) - 0.02) < 0.0005 else 3
    return f"{x:.{d}f}".replace("-", "$-$")


def write_tex(m, cmp, path):
    names = {"q4": "Q$_4$", "tr4": "Tr$_4$", "cl4": "Cl$_4$", "full": "Full"}
    sets = {"": "BUSI", "_busbra": "BUS-BRA", "_breast": "BrEaST"}
    same = int((cmp.best_old == cmp.best_new).sum())
    cap = (r"\caption{The 23 plan C runs whose first training stopped after 21 to 28 epochs, before the protocol minimum of 30 "
           r"epochs. All results in this article use these runs retrained with the minimum in effect. In the retraining the first "
           f"epochs repeated exactly; {same} runs kept their best epoch and weights and {23 - same} reached a later best epoch. "
           r"Family F3 as reported and with the original runs put back, with 95\% intervals and the same bootstrap draws. "
           r"Higher or lower: significant difference in that direction.}\label{stab:min30}\\")
    head = r"Set & Share & Contrast & Reported & Verdict & Original runs & Verdict \\"
    L = [r"\begin{longtable}{@{}lrlcccc@{}}", cap, r"\toprule", head, r"\midrule", r"\endfirsthead",
         r"\toprule", head, r"\midrule", r"\endhead"]
    v = lambda x: x.split("(")[1].rstrip(")").capitalize() if "(" in x else x.capitalize()
    for _, r in m.iterrows():
        fam = r.family
        a, b = r.contrast.split(" - ")
        L.append(f"{sets[fam[7:]]} & {int(fam[4:7])}\\% & {names[a]} $-$ {names[b]} & "
                 f"{f3(r.diff_rep)} ({f3(r.ci95_lo_rep)} to {f3(r.ci95_hi_rep)}) & {v(r.verdict_rep)} & "
                 f"{f3(r.diff_orig)} ({f3(r.ci95_lo_orig)} to {f3(r.ci95_hi_orig)}) & {v(r.verdict_orig)} \\\\")
    L += [r"\bottomrule", r"\end{longtable}", ""]
    open(path, "w").write("\n".join(L))


def main(data, orig, out):
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    ids = set(pd.read_csv(os.path.join(orig, "summaries.csv")).run_id)
    assert len(ids) == 23, len(ids)
    shutil.copy(os.path.join(data, "summaries.csv"), tmp)
    for nm in ("test_busi", "test_busbra", "test_breast"):
        cur = pd.read_csv(os.path.join(data, nm + ".csv.gz"))
        old = pd.read_csv(os.path.join(orig, nm + ".csv.gz"))
        assert len(old) == cur.run_id.isin(ids).sum(), nm
        pd.concat([cur[~cur.run_id.isin(ids)], old], ignore_index=True).to_csv(os.path.join(tmp, nm + ".csv.gz"), index=False)
    _, _, tabs_rep = analysis.load(data)
    _, _, tabs_orig = analysis.load(tmp)
    keep = lambda f: f[f.family.str.startswith("F3")]
    a = keep(analysis.run_families(tabs_rep))
    b = keep(analysis.run_families(tabs_orig))
    cols = ["diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "p_holm", "verdict"]
    m = a[["family", "contrast"] + cols].merge(b[["family", "contrast"] + cols], on=["family", "contrast"],
                                               suffixes=("_rep", "_orig"))
    m.to_csv(os.path.join(out, "f3_reported_vs_original.csv"), index=False)
    lev = lambda t: t[(t.plan == "C")].groupby(["frac", "model"]).dice.mean()
    pd.DataFrame({"busi_reported": lev(tabs_rep["busi"]), "busi_original": lev(tabs_orig["busi"])}).round(4).to_csv(
        os.path.join(out, "levels_planC.csv"))
    shutil.rmtree(tmp)
    write_tex(m, pd.read_csv(os.path.join(orig, "comparison.csv")), os.path.join(out, "tab_min30.tex"))
    print(m[["family", "contrast", "diff_rep", "diff_orig", "verdict_rep", "verdict_orig"]].round(4).to_string(index=False))
    print("verdicts that differ:", int((m.verdict_rep != m.verdict_orig).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
