"""Qualitative examples for the manuscript figure (inference only, no new analysis).

The checkpoints of plan A were not stored. The seven models of fold 0 and seed 0 are therefore retrained with
the same fold, seed, initial weights and training code as runner.one_run, and the BUSI test Dice of each retrained
model is compared with the stored run. A q4 checkpoint saved by the noise test (runs_noise) is reused when present.

Example images are chosen by a fixed rule from the STORED per-image results, before any retrained prediction is seen:
  lesion images (BUSI test fold 0, BUS-BRA, BrEaST): the image whose mean Dice over the seven models is closest to
  the median of that mean over the data set (ties: first image identifier);
  normal BUSI image: the image whose number of models with a non-empty mask is closest to the mean of that number
  over the normal images (ties: first image identifier).
"""
import json
import os
import time

import numpy as np
import pandas as pd

import qbus
import runner

MODELS = ("q4", "qp4", "tr4", "cl4", "id4", "full", "unet")
FOLD, SEED = 0, 0


def rid(m):
    return runner.run_id(dict(plan="A", model=m, fold=FOLD, seed=SEED, frac=1.0))


def select_examples(runs_root):
    """Image identifiers per data set, from the stored CSVs of the original runs only."""
    sel = {}
    for ds, fn in (("busi", "test_busi.csv"), ("busbra", "test_busbra.csv"), ("breast", "test_breast.csv")):
        tabs = [pd.read_csv(os.path.join(runs_root, "A", rid(m), fn))[["image", "dice"]].rename(columns={"dice": m})
                for m in MODELS]
        t = tabs[0]
        for x in tabs[1:]:
            t = t.merge(x, on="image")
        t["mean"] = t[list(MODELS)].mean(1)
        t["gap"] = (t["mean"] - t["mean"].median()).abs()
        t = t.sort_values(["gap", "image"], kind="mergesort")
        sel[ds] = dict(image=str(t.image.iloc[0]), stored=t.iloc[0][list(MODELS)].astype(float).to_dict(),
                       median_mean=float(t["mean"].median()))
    tabs = [pd.read_csv(os.path.join(runs_root, "A", rid(m), "normals_busi.csv"))[["image", "pred_pixels"]]
            .rename(columns={"pred_pixels": m}) for m in MODELS]
    t = tabs[0]
    for x in tabs[1:]:
        t = t.merge(x, on="image")
    t["count"] = (t[list(MODELS)] > 0).sum(1)
    t["gap"] = (t["count"] - t["count"].mean()).abs()
    t = t.sort_values(["gap", "image"], kind="mergesort")
    sel["normal"] = dict(image=str(t.image.iloc[0]), stored_pixels=t.iloc[0][list(MODELS)].astype(int).to_dict(),
                         mean_count=float(t["count"].mean()))
    return sel


def retrain(m, prep, k):
    """Rebuild and train model m of fold 0, seed 0 exactly as runner.one_run does."""
    runner.free_memory()
    data = np.load(os.path.join(prep, "busi_lesion.npz"))
    X, Y = data["x"], data["y"]
    tr, va, te, man = runner.load_split(prep, FOLD, 1.0)
    kind, n = qbus.parse(m)
    ref = qbus.reference_init(n if n else 4, SEED)
    qbus.set_seeds(SEED)
    model = qbus.build_model(m, k=k)
    if kind != "unet":
        qbus.apply_init(model, ref)
    qbus.gradient_share(model, X[tr], Y[tr])                     # same calls as the original run
    qbus.fit(model, X[tr], Y[tr], X[va], Y[va], seed=SEED)
    return model


def load_or_retrain(m, prep, k, noise_root):
    kind, n = qbus.parse(m)
    ck = os.path.join(noise_root, rid(m), "model.weights.h5")
    if m == "q4" and os.path.exists(ck):
        runner.free_memory()
        qbus.set_seeds(SEED)
        model = qbus.build_model(m, k=k)
        model.load_weights(ck)
        return model, "noise checkpoint"
    return retrain(m, prep, k), "retrained"


def one_model(m, prep, runs_root, noise_root, out_root, sel, k):
    d = os.path.join(out_root, rid(m))
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    model, src = load_or_retrain(m, prep, k, noise_root)
    data = np.load(os.path.join(prep, "busi_lesion.npz"))
    X, Y = data["x"], data["y"]
    _, _, te, man = runner.load_split(prep, FOLD, 1.0)
    prob = qbus.predict(model, X[te])
    dice = np.array([r["dice"] for r in qbus.per_image(Y[te], prob)])
    orig = pd.read_csv(os.path.join(runs_root, "A", rid(m), "test_busi.csv"))
    out = dict(model=m, source=src, k=k, busi_dice=float(dice.mean()), stored_dice=float(orig.dice.mean()),
               max_abs_image_diff=float(np.abs(orig.dice.values - dice).max()))
    masks = {}
    ids = list(man.image.values[te])
    i = ids.index(sel["busi"]["image"])
    masks["busi"] = prob[i, ..., 0] >= qbus.CFG["threshold"]
    out["dice_busi_example"] = float(dice[i])
    for ds in ("busbra", "breast"):
        e = np.load(os.path.join(prep, f"{ds}.npz"), allow_pickle=True)
        j = [str(a) for a in e["ids"]].index(sel[ds]["image"])
        p = qbus.predict(model, e["x"][j:j + 1])
        masks[ds] = p[0, ..., 0] >= qbus.CFG["threshold"]
        out[f"dice_{ds}_example"] = float(qbus.per_image(e["y"][j:j + 1], p)[0]["dice"])
    e = np.load(os.path.join(prep, "busi_normal.npz"), allow_pickle=True)
    j = [str(a) for a in e["ids"]].index(sel["normal"]["image"])
    masks["normal"] = qbus.predict(model, e["x"][j:j + 1])[0, ..., 0] >= qbus.CFG["threshold"]
    out["pixels_normal_example"] = int(masks["normal"].sum())
    np.savez_compressed(os.path.join(d, "masks.npz"), **masks)
    if src == "retrained":
        model.save_weights(os.path.join(d, "model.weights.h5"))
    out["minutes"] = (time.time() - t0) / 60
    out["env"] = qbus.env_info()
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1, default=str)
    lock = d + ".lock"
    if os.path.exists(lock):
        os.remove(lock)
    runner.free_memory()
    return out


def collect(prep, out_root, sel, dst):
    """One file with the four example images, reference masks and the masks of the seven models."""
    data = np.load(os.path.join(prep, "busi_lesion.npz"))
    _, _, te, man = runner.load_split(prep, FOLD, 1.0)
    ids = list(man.image.values[te])
    i = te[ids.index(sel["busi"]["image"])]
    img, ref = {"busi": data["x"][i]}, {"busi": data["y"][i]}
    for ds in ("busbra", "breast"):
        e = np.load(os.path.join(prep, f"{ds}.npz"), allow_pickle=True)
        j = [str(a) for a in e["ids"]].index(sel[ds]["image"])
        img[ds], ref[ds] = e["x"][j], e["y"][j]
    e = np.load(os.path.join(prep, "busi_normal.npz"), allow_pickle=True)
    j = [str(a) for a in e["ids"]].index(sel["normal"]["image"])
    img["normal"], ref["normal"] = e["x"][j], np.zeros_like(e["x"][j])
    arrs, rows = {}, []
    for ds in ("busi", "busbra", "breast", "normal"):
        arrs[f"img_{ds}"] = np.asarray(img[ds]).squeeze()
        arrs[f"ref_{ds}"] = (np.asarray(ref[ds]).squeeze() > 0)
    for m in MODELS:
        mk = np.load(os.path.join(out_root, rid(m), "masks.npz"))
        s = json.load(open(os.path.join(out_root, rid(m), "summary.json")))
        s.pop("env", None)
        rows.append(s)
        for ds in ("busi", "busbra", "breast", "normal"):
            arrs[f"pred_{m}_{ds}"] = mk[ds]
    np.savez_compressed(dst, ids=json.dumps(sel), **arrs)
    return pd.DataFrame(rows)
