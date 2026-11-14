"""Run list, lock files and one complete training + evaluation + audit run (protocol Sections 4-6)."""
import json
import os
import socket
import time

import numpy as np
import pandas as pd

import qbus

FOLDS = range(5)
FACTORS = (1, 10, 100)


SEEDS = (0, 1, 2, 3, 4)            # protocol version 1.1
WAVES = ((0, 1, 2), (3, 4))


def run_list(plan, seeds=SEEDS):
    """Main runs of plans A, B, C (protocol Section 4), as dicts. Order: seed, fold, model."""
    runs = []
    if plan == "A":
        for s in seeds:
            for f in FOLDS:
                for m in ("q4", "qp4", "s4", "tr4", "cl4", "id4", "full", "unet"):
                    runs.append(dict(plan="A", model=m, fold=f, seed=s, frac=1.0))
    elif plan == "B":
        for s in seeds:
            for f in FOLDS:
                for n in (2, 8):
                    for b in ("q", "qp", "s", "tr", "cl"):
                        runs.append(dict(plan="B", model=f"{b}{n}", fold=f, seed=s, frac=1.0))
    elif plan == "C":
        for s in seeds:
            for f in FOLDS:
                for p in (0.1, 0.25, 0.5):
                    for m in ("q4", "tr4", "cl4", "full"):
                        runs.append(dict(plan="C", model=m, fold=f, seed=s, frac=p))
    elif plan == "SEL":
        for n in (2, 4, 8):
            for b in ("q", "qp", "s", "tr", "cl"):
                for k in FACTORS:
                    runs.append(dict(plan="SEL", model=f"{b}{n}", fold=0, seed=100, frac=1.0, k=k))
    else:
        raise ValueError(plan)
    return runs


def run_id(r):
    rid = f"{r['plan']}_{r['model']}_f{r['fold']}_s{r['seed']}"
    if r["frac"] < 1:
        rid += f"_p{int(round(r['frac'] * 100)):03d}"
    if r["plan"] == "SEL":
        rid += f"_k{r['k']}"
    return rid


def claim(out_root, rid, session, settle=8):
    """Lock file on Drive: returns True if this session may run `rid` (not done, not locked by a live session).
    Writes a lock with a unique token, waits `settle` seconds and reads it back, so that two sessions that
    claim the same run at the same moment do not both run it (the later writer wins; the other backs off)."""
    import uuid
    d = os.path.join(out_root, rid)
    if os.path.exists(os.path.join(d, "summary.json")):
        return False
    os.makedirs(out_root, exist_ok=True)
    lock = os.path.join(out_root, rid + ".lock")
    if os.path.exists(lock):
        age = time.time() - os.path.getmtime(lock)
        if age < 6 * 3600:      # a lock older than 6 h is considered stale (session died)
            return False
    token = f"{session} {socket.gethostname()} {uuid.uuid4().hex} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    with open(lock, "w") as f:
        f.write(token)
    time.sleep(settle)
    try:
        with open(lock) as f:
            return f.read() == token
    except OSError:
        return False


def ordered(runs, session):
    """Session A works through a list from the start, session B from the end, so that two sessions meet only once."""
    return list(reversed(runs)) if str(session).upper() == "B" else list(runs)


def load_split(prep, fold, frac, seed_subset=2027):
    """Indices of training, validation and test images for a fold and training fraction."""
    m = pd.read_csv(os.path.join(prep, "busi_manifest.csv"))
    te, va = fold, (fold + 1) % 5
    test = np.where(m.fold == te)[0]
    val = np.where(m.fold == va)[0]
    train = np.where(~m.fold.isin([te, va]))[0]
    if frac < 1:
        col = f"sub{int(round(frac * 100)):03d}_f{fold}"
        train = np.where(m[col].astype(bool).values)[0]
    return train, val, test, m


def one_run(r, prep, out_root, selection=None, session="A"):
    rid = run_id(r)
    d = os.path.join(out_root, rid)
    os.makedirs(d, exist_ok=True)
    data = np.load(os.path.join(prep, "busi_lesion.npz"))
    X, Y = data["x"], data["y"]
    tr, va, te, man = load_split(prep, r["fold"], r["frac"])
    kind, n = qbus.parse(r["model"])
    k = r.get("k") or (selection or {}).get(r["model"], 1)
    k = 1 if kind in ("id", "full", "unet") else k
    # ---- build with matched initialization
    ref = qbus.reference_init(n if n else 4, r["seed"])
    qbus.set_seeds(r["seed"])
    model = qbus.build_model(r["model"], k=k)
    init_diff = qbus.apply_init(model, ref) if kind != "unet" else 0.0
    blk = qbus.block_layer(model)
    blk_init = blk.effective() if blk is not None else None
    dense_init = {l.name: [np.array(w) for w in l.get_weights()] for l in model.layers if l.name.startswith(("pre_dense", "post_dense"))}
    grad0 = qbus.gradient_share(model, X[tr], Y[tr])
    # ---- train
    info = qbus.fit(model, X[tr], Y[tr], X[va], Y[va], seed=r["seed"],
                    max_epochs=80 if r["plan"] == "SEL" else None, log=os.path.join(d, "history.csv"))
    summary = dict(**{a: b for a, b in r.items() if a != "k"}, run_id=rid, k=k, init_max_abs_diff=init_diff, session=session, **info, **qbus.count_params(model),
                   env=qbus.env_info())
    if r["plan"] == "SEL":
        _finish(d, summary)
        return summary
    # ---- test (BUSI fold)
    prob = qbus.predict(model, X[te])
    rows = pd.DataFrame(qbus.per_image(Y[te], prob))
    rows.insert(0, "image", man.image.values[te])
    rows.insert(1, "group", man.group.values[te])
    rows.insert(2, "cls", man.cls.values[te])
    rows.to_csv(os.path.join(d, "test_busi.csv"), index=False)
    summary["test_dice"] = float(rows.dice.mean())
    one = X[te][:1, ..., None].astype(np.float32) / 255.0
    _ = model(one, training=False)                                  # warm-up
    t0 = time.time()
    for _ in range(20):
        _ = model(one, training=False)
    summary["infer_ms_per_image"] = (time.time() - t0) / 20 * 1000
    # ---- external sets and normals (plan A only)
    if r["plan"] == "A":
        for nm in ("busbra", "breast"):
            fp = os.path.join(prep, f"{nm}.npz")
            if os.path.exists(fp):
                e = np.load(fp, allow_pickle=True)
                er = pd.DataFrame(qbus.per_image(e["y"], qbus.predict(model, e["x"])))
                er.insert(0, "image", e["ids"])
                er.insert(1, "patient", e["patients"])
                er.to_csv(os.path.join(d, f"test_{nm}.csv"), index=False)
                summary[f"{nm}_dice"] = float(er.dice.mean())
        fp = os.path.join(prep, "busi_normal.npz")
        if os.path.exists(fp):
            e = np.load(fp, allow_pickle=True)
            pn = qbus.predict(model, e["x"])[..., 0] >= qbus.CFG["threshold"]
            pd.DataFrame(dict(image=e["ids"], pred_pixels=pn.reshape(len(pn), -1).sum(1))).to_csv(
                os.path.join(d, "normals_busi.csv"), index=False)
    # ---- audit (plans A and B; models with a code)
    if r["plan"] in ("A", "B") and kind not in ("full", "unet"):
        cr = qbus.code_replacement(model, X[tr], X[te], Y[te])
        pd.DataFrame(dict(image=man.image.values[te], normal=cr["normal"], const=cr["const"], swap=cr["swap"])).to_csv(
            os.path.join(d, "code_replacement.csv"), index=False)
        summary["drop_const"] = float(np.mean(cr["normal"] - cr["const"]))
        summary["drop_swap"] = float(np.mean(cr["normal"] - cr["swap"]))
        summary.update({f"change_{k2}": v for k2, v in qbus.weight_change(model, dense_init).items()})
        summary["change_block"] = qbus.block_change(blk, blk_init) if blk is not None else np.nan
        summary["grad_share_init"] = grad0
        summary["grad_share_final"] = qbus.gradient_share(model, X[tr], Y[tr])
        summary["entanglement_mw"] = qbus.entanglement(model, X[te])
        summary["surrogate_r2"] = qbus.surrogate_fit(model, X[tr], X[te])
        if r["model"] == "q4" and r["plan"] == "A":
            fs = qbus.finite_shot_dice(model, X[te], Y[te])
            summary["finite_shot_dice"] = float(fs.mean())
            pd.DataFrame(dict(image=man.image.values[te], dice_1024shots=fs)).to_csv(os.path.join(d, "finite_shot.csv"), index=False)
    if blk is not None:
        np.savez(os.path.join(d, "block_weights.npz"), **{f"init_{a}": b for a, b in blk_init.items()},
                 **{f"final_{a}": b for a, b in blk.effective().items()})
    _finish(d, summary)
    return summary


def _finish(d, summary):
    def clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({a: clean(b) for a, b in summary.items()}, f, indent=1, default=str)
    lock = d + ".lock"
    if os.path.exists(lock):
        os.remove(lock)


def select_factors(out_root):
    """Lowest best validation loss per model over k in {1, 10, 100} (selection runs)."""
    rows = []
    for r in run_list("SEL"):
        p = os.path.join(out_root, run_id(r), "summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            rows.append(dict(model=s["model"], k=s["k"], best_val_loss=s["best_val_loss"]))
    t = pd.DataFrame(rows)
    sel = t.loc[t.groupby("model").best_val_loss.idxmin()].set_index("model").k.astype(int).to_dict()
    return t, sel
