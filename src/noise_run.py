"""Synthetic noise test of the trained circuit Q4 (protocol addendum, inference only).

The model checkpoints of plan A were not stored, so each q4 run of plan A is retrained with the same fold, seed,
initial weights and training code as runner.one_run, and the test Dice on BUSI is compared with the stored value.
Noise is then added only at prediction time: the trained angles are kept and no weight is refitted.
"""
import os, json, time
import numpy as np
import pandas as pd
import qbus, qnoise, runner

SHOTS = (1024, 4096, 8192, None)          # None = exact expectation values
REPEATS = 5


def retrain(r, prep):
    """Rebuild and train run r exactly as runner.one_run does. Returns model, data and split."""
    runner.free_memory()
    data = np.load(os.path.join(prep, "busi_lesion.npz"))
    X, Y = data["x"], data["y"]
    tr, va, te, man = runner.load_split(prep, r["fold"], r["frac"])
    kind, n = qbus.parse(r["model"])
    ref = qbus.reference_init(n, r["seed"])
    qbus.set_seeds(r["seed"])
    model = qbus.build_model(r["model"], k=r["k"])
    qbus.apply_init(model, ref)
    qbus.gradient_share(model, X[tr], Y[tr])                     # same calls as the original run
    info = qbus.fit(model, X[tr], Y[tr], X[va], Y[va], seed=r["seed"])
    return model, X, Y, tr, te, man, info


def noise_eval(model, xte, yte, seed=0):
    blk = qbus.block_layer(model)
    angles = blk.effective()["angles"]
    c = qbus.block_inputs(model, xte)
    rng = np.random.default_rng(seed)
    out = {}
    levels = dict(none=dict(p1=0, p2=0, gamma=0, pm=0), **qnoise.LEVELS)
    for lv, p in levels.items():
        ez = qnoise.noisy_expvals(c, angles, blk.n, blk.cnot, **p)
        for shots in SHOTS:
            reps = 1 if shots is None else REPEATS
            res = []
            for _ in range(reps):
                est = qnoise.sample_expvals(ez, shots, rng).astype(np.float32)
                qbus.set_replace(model, 2, bank=est, perm=np.arange(4096) % len(est), ptr=0)
                res.append([d["dice"] for d in qbus.per_image(yte, qbus.predict_in_order(model, xte))])
            out[(lv, "exact" if shots is None else str(shots))] = np.mean(np.array(res), axis=0)
    qbus.set_replace(model, 0)
    return out


def one_noise_run(r, prep, runs_root, out_root):
    rid = runner.run_id(r)
    d = os.path.join(out_root, rid)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    model, X, Y, tr, te, man, info = retrain(r, prep)
    plain = np.array([q["dice"] for q in qbus.per_image(Y[te], qbus.predict(model, X[te]))])
    orig = pd.read_csv(os.path.join(runs_root, "A", rid, "test_busi.csv"))
    match = float(np.abs(orig.dice.values - plain).max())
    res = noise_eval(model, X[te], Y[te], seed=r["seed"])
    rows = []
    for (lv, sh), v in res.items():
        rows.append(pd.DataFrame(dict(image=man.image.values[te], group=man.group.values[te], level=lv, shots=sh, dice=v)))
    pd.concat(rows).to_csv(os.path.join(d, "noise.csv"), index=False)
    model.save_weights(os.path.join(d, "model.weights.h5"))
    s = dict(run_id=rid, fold=r["fold"], seed=r["seed"], k=r["k"], retrain_dice=float(plain.mean()),
             original_dice=float(orig.dice.mean()), max_abs_image_diff=match,
             exact_via_code_path=float(res[("none", "exact")].mean()), minutes=(time.time() - t0) / 60,
             epochs=info["epochs"], env=qbus.env_info())
    json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=1, default=str)
    lock = d + ".lock"
    if os.path.exists(lock):
        os.remove(lock)
    runner.free_memory()
    return s
