"""Statistics of the pre-specified protocol (Section 7) from the files written by notebooks/02_collect.ipynb.

Usage: python analysis.py <folder with summaries.csv, test_busi.csv.gz, ...> <output folder>

Unit: per-image test Dice. Crossed hierarchical bootstrap over clusters and seeds (B = 10,000), with the same
resampled clusters and seeds for both models of a contrast. BUSI clusters are duplicate groups (one kept image per
group after cleaning); external clusters are patients. On the external sets every image is scored by the five fold
models of a seed; the image score of a seed is the mean over these five models. Random numbers: one generator per
family, from numpy SeedSequence(0) spawned in the order of the families listed in Section 7.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

B = 10_000
MARGIN = 0.02
SEEDS = (0, 1, 2, 3, 4)
FAMILIES = [
    ("F1", "busi", "A", 1.0, [("q4", "qp4"), ("q4", "s4"), ("q4", "tr4"), ("q4", "cl4")]),
    ("F2", "busi", "A", 1.0, [("tr4", "full"), ("q4", "full"), ("tr4", "unet"), ("q4", "unet")]),
    ("F3_p010", "busi", "C", 0.1, [("q4", "cl4"), ("tr4", "cl4"), ("q4", "full"), ("tr4", "full")]),
    ("F3_p025", "busi", "C", 0.25, [("q4", "cl4"), ("tr4", "cl4"), ("q4", "full"), ("tr4", "full")]),
    ("F3_p050", "busi", "C", 0.5, [("q4", "cl4"), ("tr4", "cl4"), ("q4", "full"), ("tr4", "full")]),
    ("F4", "busi", "B", 1.0, [("q2", "qp2"), ("q2", "tr2"), ("q8", "qp8"), ("q8", "tr8")]),
]
for ext in ("busbra", "breast"):
    FAMILIES += [(f"F1_{ext}", ext, "A", 1.0, FAMILIES[0][4]), (f"F2_{ext}", ext, "A", 1.0, FAMILIES[1][4])]
    FAMILIES += [(f"F3_p{int(p * 100):03d}_{ext}", ext, "C", p, FAMILIES[2][4]) for p in (0.1, 0.25, 0.5)]
NONINF = lambda fam: fam.startswith("F2")


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def matrix(tab, plan, frac, model, cluster):
    """Seeds x clusters: sum of per-image Dice and number of images per cluster (images averaged over fold models)."""
    t = tab[(tab.plan == plan) & (tab.frac == frac) & (tab.model == model)]
    img = t.groupby(["seed", "image"]).agg(dice=("dice", "mean"), cl=(cluster, "first")).reset_index()
    return img


def paired(tab, plan, frac, a, b, cluster, seeds):
    ia, ib = matrix(tab, plan, frac, a, cluster), matrix(tab, plan, frac, b, cluster)
    m = ia.merge(ib, on=["seed", "image", "cl"], suffixes=("_a", "_b"))
    m = m[m.seed.isin(seeds)]
    m["d"] = m.dice_a - m.dice_b
    cls = np.array(sorted(m.cl.unique(), key=str))
    ci = {c: i for i, c in enumerate(cls)}
    S = np.zeros((len(seeds), len(cls)))
    N = np.zeros(len(cls))
    first = m[m.seed == seeds[0]]
    for c, n in first.groupby("cl").size().items():
        N[ci[c]] = n
    for si, s in enumerate(seeds):
        g = m[m.seed == s].groupby("cl").d.sum()
        for c, v in g.items():
            S[si, ci[c]] = v
    return S, N, m


def boot(S, N, rng, n=B, batch=1000):
    ns, nc = S.shape
    out = []
    for i in range(0, n, batch):
        k = min(batch, n - i)
        W = rng.multinomial(nc, np.full(nc, 1 / nc), size=k).astype(float)
        V = rng.multinomial(ns, np.full(ns, 1 / ns), size=k).astype(float)
        num = np.einsum("bs,sc,bc->b", V, S, W)
        den = V.sum(1) * (W @ N)
        out.append(num / den)
    return np.concatenate(out)


def contrast(tab, plan, frac, a, b, cluster, rng, noninf):
    seeds = [s for s in SEEDS if complete(tab, plan, frac, a, b, s)]
    S, N, m = paired(tab, plan, frac, a, b, cluster, seeds)
    est = S.sum() / (len(seeds) * N.sum())
    bt = boot(S, N, rng)
    p = min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean()))
    p_ni = (bt <= -MARGIN).mean()
    per_seed = {int(s): float(S[i].sum() / N.sum()) for i, s in enumerate(seeds)}
    loso = {int(s): float(np.delete(S, i, 0).sum() / ((len(seeds) - 1) * N.sum())) for i, s in enumerate(seeds)}
    lo95, hi95 = np.percentile(bt, [2.5, 97.5])
    lo90, hi90 = np.percentile(bt, [5, 95])
    return dict(contrast=f"{a} - {b}", diff=est, ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90, p=p,
                p_noninf=p_ni if noninf else np.nan, halfwidth95=(hi95 - lo95) / 2, seeds=len(seeds),
                clusters=len(N), images=int(N.sum()), per_seed=json.dumps(per_seed), loso=json.dumps(loso))


def complete(tab, plan, frac, a, b, seed):
    t = tab[(tab.plan == plan) & (tab.frac == frac) & (tab.seed == seed) & tab.model.isin([a, b])]
    return t.groupby("model").fold.nunique().reindex([a, b]).fillna(0).min() == 5


def verdict(r, noninf):
    v = []
    if r["p_holm"] < 0.05:
        v.append("difference (higher)" if r["diff"] > 0 else "difference (lower)")
    if -MARGIN < r["ci90_lo"] and r["ci90_hi"] < MARGIN:
        v.append("equivalent")
    if noninf and r["p_noninf_holm"] < 0.05:
        v.append("non-inferior")
    return ", ".join(v) if v else "inconclusive"


def load(src):
    summ = pd.read_csv(os.path.join(src, "summaries.csv"))
    meta = summ[["run_id", "plan", "model", "fold", "seed", "frac"]]
    tabs = {}
    for nm, f in (("busi", "test_busi"), ("busbra", "test_busbra"), ("breast", "test_breast")):
        p = os.path.join(src, f + ".csv.gz")
        if os.path.exists(p):
            tabs[nm] = pd.read_csv(p).merge(meta, on="run_id")
    return summ, meta, tabs


def run_families(tabs, excl=None):
    ss = np.random.SeedSequence(0).spawn(len(FAMILIES))
    rows = []
    for (fam, ds, plan, frac, pairs), sq in zip(FAMILIES, ss):
        if ds not in tabs:
            continue
        tab = tabs[ds]
        if excl is not None and ds == "busi":
            tab = tab[~tab.image.isin(excl)]
        rng = np.random.default_rng(sq)
        cluster = "group" if ds == "busi" else "patient"
        res = [dict(family=fam, dataset=ds, plan=plan, frac=frac, **contrast(tab, plan, frac, a, b, cluster, rng, NONINF(fam)))
               for a, b in pairs]
        ph = holm([r["p"] for r in res])
        pn = holm([r["p_noninf"] for r in res]) if NONINF(fam) else [np.nan] * len(res)
        for r, h, n in zip(res, ph, pn):
            r["p_holm"], r["p_noninf_holm"] = h, n
            r["verdict"] = verdict(r, NONINF(fam))
        rows += res
    return pd.DataFrame(rows)


def levels(tabs, rng_seed=1):
    """Mean Dice per model (descriptive), bootstrap CI over clusters and seeds; between-seed SD."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for ds, tab in tabs.items():
        cluster = "group" if ds == "busi" else "patient"
        for (plan, frac, model), t in tab.groupby(["plan", "frac", "model"]):
            img = t.groupby(["seed", "image"]).agg(dice=("dice", "mean"), cl=(cluster, "first"),
                                                     empty=("empty", "mean"), hd95=("hd95", "mean"),
                                                     iou=("iou", "mean")).reset_index()
            seeds = sorted(img.seed.unique())
            cls = sorted(img.cl.unique(), key=str)
            ci = {c: i for i, c in enumerate(cls)}
            S = np.zeros((len(seeds), len(cls)))
            N = np.zeros(len(cls))
            for c, n in img[img.seed == seeds[0]].groupby("cl").size().items():
                N[ci[c]] = n
            for si, s in enumerate(seeds):
                for c, v in img[img.seed == s].groupby("cl").dice.sum().items():
                    S[si, ci[c]] = v
            bt = boot(S, N, rng, n=2000)
            per_seed = img.groupby("seed").dice.mean()
            rows.append(dict(dataset=ds, plan=plan, frac=frac, model=model, dice=img.dice.mean(),
                             ci95_lo=np.percentile(bt, 2.5), ci95_hi=np.percentile(bt, 97.5),
                             seed_sd=per_seed.std(ddof=1), iou=img.iou.mean(), hd95=img.hd95.mean(),
                             empty_rate=img["empty"].mean(), seeds=len(seeds), images=int(N.sum())))
    return pd.DataFrame(rows)


def audit(summ):
    """Protocol Section 6: mean over runs (plans A and B) with a 95% bootstrap CI over runs (descriptive)."""
    cols = ["drop_const", "drop_swap", "change_block", "change_pre_dense", "change_post_dense", "grad_share_init",
            "grad_share_final", "entanglement_mw", "surrogate_r2", "finite_shot_dice", "test_dice"]
    t = summ[summ.plan.isin(["A", "B"])]
    rng = np.random.default_rng(2)
    rows = []
    for (plan, model), g in t.groupby(["plan", "model"]):
        r = dict(plan=plan, model=model, runs=len(g))
        for c in cols:
            if c in g and g[c].notna().any():
                v = g[c].dropna().values
                bt = rng.choice(v, (2000, len(v))).mean(1)
                r[c] = v.mean()
                r[c + "_lo"], r[c + "_hi"] = np.percentile(bt, [2.5, 97.5])
        if "drop_const" in r:
            r["uses_code_const"] = bool(r["drop_const"] >= 0.02 and r["drop_const_lo"] > 0)
            r["uses_code_swap"] = bool(r["drop_swap"] >= 0.02 and r["drop_swap_lo"] > 0)
        rows.append(r)
    return pd.DataFrame(rows)


def costs(summ):
    keys = [k for k in ("trainable", "block", "bottleneck_params", "minutes", "epochs", "infer_ms_per_image", "k") if k in summ]
    t = summ[summ.plan.isin(["A", "B", "C"])]
    return t.groupby(["plan", "frac", "model"])[keys].mean().reset_index()


def normals(src, meta):
    p = os.path.join(src, "normals_busi.csv.gz")
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p).merge(meta, on="run_id")
    t["fp"] = (t.pred_pixels > 0).astype(int)
    return t.groupby(["plan", "frac", "model"]).agg(fp_rate=("fp", "mean"), images=("image", "nunique"),
                                                    runs=("run_id", "nunique")).reset_index()


def by_class(tabs):
    t = tabs["busi"]
    t = t[(t.plan == "A")]
    return t.groupby(["model", "cls"]).dice.mean().unstack().reset_index()


def main(src, out):
    os.makedirs(out, exist_ok=True)
    summ, meta, tabs = load(src)
    fam = run_families(tabs)
    fam.to_csv(os.path.join(out, "contrasts.csv"), index=False)
    man = pd.read_csv(os.path.join(src, "busi_manifest.csv"))
    excl = set(man.image[man.annotated == 1]) if "annotated" in man else set()
    sens = run_families({"busi": tabs["busi"]}, excl=excl)
    sens = sens[sens.family.isin(["F1", "F2"])]
    sens.to_csv(os.path.join(out, "contrasts_no_annotated.csv"), index=False)
    levels(tabs).to_csv(os.path.join(out, "model_levels.csv"), index=False)
    audit(summ).to_csv(os.path.join(out, "audit.csv"), index=False)
    costs(summ).to_csv(os.path.join(out, "costs.csv"), index=False)
    nrm = normals(src, meta)
    if nrm is not None:
        nrm.to_csv(os.path.join(out, "normals_fp.csv"), index=False)
    by_class(tabs).to_csv(os.path.join(out, "by_class_planA.csv"), index=False)
    print(fam[["family", "contrast", "diff", "ci95_lo", "ci95_hi", "p_holm", "verdict"]].round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
