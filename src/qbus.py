"""Models, data, training, evaluation and bottleneck audit for the BUSI quantum-bottleneck study.
Protocol: protocols/PROTOCOL_busi_prespecified.txt. Keras 3 / TensorFlow 2.x.

Model names: q{n}, qp{n}, s{n}, tr{n}, cl{n}, id{n} (n = 2, 4, 8), full, unet.
Block weights are stored as v = w / k and used as w = k v (effective learning rate x k).
"""
import json
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import qsim

BLOCKS = ("q", "qp", "s", "tr", "cl", "id")
CFG = dict(image_size=256, batch_size=16, lr=1e-4, max_epochs=200, min_epochs=30, es_patience=20,
           lr_patience=8, lr_factor=0.5, min_lr=1e-6, flip=0.5, rot_deg=10.0, bright=0.2, contrast=0.2,
           n_layers=2, threshold=0.5)


def parse(name):
    """'qp4' -> ('qp', 4); 'full' -> ('full', None)."""
    if name in ("full", "unet"):
        return name, None
    for b in sorted(BLOCKS, key=len, reverse=True):
        if name.startswith(b) and name[len(b):].isdigit():
            return b, int(name[len(b):])
    raise ValueError(name)


# ============================================================================ block layers
class _Scaled(layers.Layer):
    """Base: trainable variables v, effective weights k * v."""

    def __init__(self, n, k=1.0, **kw):
        super().__init__(**kw)
        self.n, self.k = int(n), float(k)

    def get_config(self):
        return {**super().get_config(), "n": self.n, "k": self.k}

    def effective(self):
        return {v.name.split("/")[-1].split(":")[0]: np.asarray(v.numpy()) * self.k for v in self.trainable_weights}

    def set_effective(self, d):
        for v in self.trainable_weights:
            key = v.name.split("/")[-1].split(":")[0]
            v.assign(np.asarray(d[key], np.float32) / self.k)


class Circuit(_Scaled):
    def __init__(self, n, k=1.0, n_layers=2, cnot=True, **kw):
        super().__init__(n, k, **kw)
        self.n_layers, self.cnot = int(n_layers), bool(cnot)

    def get_config(self):
        return {**super().get_config(), "n_layers": self.n_layers, "cnot": self.cnot}

    def build(self, input_shape):
        k = self.k
        self.angles = self.add_weight(name="angles", shape=(self.n_layers, self.n, 3),
                                      initializer=keras.initializers.RandomUniform(0.0, 2 * np.pi / k))
        super().build(input_shape)

    def call(self, x):
        return qsim.circuit_expvals(tf.cast(x, tf.float32), self.k * tf.cast(self.angles, tf.float32), qsim.tf_ops(),
                                    self.n, self.cnot)

    def compute_output_shape(self, s):
        return s


class Surrogate(_Scaled):
    """tanh(W f(c) + b), f = the 3^n Fourier features of the circuit encoding."""

    def build(self, input_shape):
        m = 3 ** self.n
        lim = np.sqrt(6.0 / (m + self.n)) / self.k
        self.W = self.add_weight(name="W", shape=(m, self.n), initializer=keras.initializers.RandomUniform(-lim, lim))
        self.b = self.add_weight(name="b", shape=(self.n,), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        f = qsim.fourier_features(tf.cast(x, tf.float32), qsim.tf_ops(), self.n)
        return tf.tanh(tf.matmul(f, self.k * self.W) + self.k * self.b)

    def compute_output_shape(self, s):
        return s


class Trig(_Scaled):
    """tanh(A cos(pi c / 2 + phi))."""

    def build(self, input_shape):
        lim = np.sqrt(6.0 / (2 * self.n)) / self.k
        self.A = self.add_weight(name="A", shape=(self.n, self.n), initializer=keras.initializers.RandomUniform(-lim, lim))
        self.phi = self.add_weight(name="phi", shape=(self.n,), initializer=keras.initializers.RandomUniform(0.0, 2 * np.pi / self.k))
        super().build(input_shape)

    def call(self, x):
        c = tf.cos(np.pi / 2 * tf.cast(x, tf.float32) + self.k * self.phi)
        return tf.tanh(tf.matmul(c, self.k * self.A, transpose_b=True))

    def compute_output_shape(self, s):
        return s


class DenseK(_Scaled):
    """tanh(c W + b), Dense(n) with scaled learning rate."""

    def build(self, input_shape):
        lim = np.sqrt(6.0 / (2 * self.n)) / self.k
        self.W = self.add_weight(name="W", shape=(self.n, self.n), initializer=keras.initializers.RandomUniform(-lim, lim))
        self.b = self.add_weight(name="b", shape=(self.n,), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        return tf.tanh(tf.matmul(tf.cast(x, tf.float32), self.k * self.W) + self.k * self.b)

    def compute_output_shape(self, s):
        return s


class ReplaceCode(layers.Layer):
    """Identity in training. mode 1: mean training code; mode 2: code of a training image (index from `perm`)."""

    def __init__(self, bank_size=256, **kw):
        super().__init__(**kw)
        self.bank_size = bank_size

    def build(self, input_shape):
        n = input_shape[-1]
        self.mode = self.add_weight(name="mode", shape=(), initializer="zeros", trainable=False)
        self.bank = self.add_weight(name="bank", shape=(self.bank_size, n), initializer="zeros", trainable=False)
        self.idx = self.add_weight(name="idx", shape=(4096,), initializer="zeros", trainable=False, dtype="int32")
        self.ptr = self.add_weight(name="ptr", shape=(), initializer="zeros", trainable=False, dtype="int32")
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        b = tf.shape(x)[0]
        const = tf.broadcast_to(tf.reduce_mean(self.bank, axis=0, keepdims=True), tf.shape(x))
        sel = tf.gather(self.idx, tf.math.floormod(tf.range(b) + self.ptr, 4096))
        swap = tf.gather(self.bank, sel)
        return tf.where(self.mode > 1.5, swap, tf.where(self.mode > 0.5, const, x))

    def get_config(self):
        return {**super().get_config(), "bank_size": self.bank_size}


def make_block(kind, n, k):
    if kind == "q":
        return Circuit(n, k, CFG["n_layers"], cnot=True, name="block")
    if kind == "qp":
        return Circuit(n, k, CFG["n_layers"], cnot=False, name="block")
    if kind == "s":
        return Surrogate(n, k, name="block")
    if kind == "tr":
        return Trig(n, k, name="block")
    if kind == "cl":
        return DenseK(n, k, name="block")
    if kind == "id":
        return None
    raise ValueError(kind)


# ============================================================================ networks
def conv_block(x, f, d, name):
    for i in (1, 2):
        x = layers.Conv2D(f, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    if d:
        x = layers.Dropout(d, name=f"{name}_drop")(x)
    return x


def build_model(name, k=1.0, size=256):
    kind, n = parse(name)
    inp = layers.Input((size, size, 1), name="image")
    if kind == "unet":
        filters, skips, x = (64, 128, 256, 512, 1024), [], inp
        for i, f in enumerate(filters[:-1]):
            x = conv_block(x, f, 0, f"u_enc{i + 1}")
            skips.append(x)
            x = layers.MaxPooling2D(2, name=f"u_pool{i + 1}")(x)
        x = conv_block(x, filters[-1], 0, "u_bott")
        for i, (f, s) in enumerate(zip(reversed(filters[:-1]), reversed(skips))):
            x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", kernel_initializer="he_normal", name=f"u_up{i + 1}")(x)
            x = conv_block(layers.Concatenate(name=f"u_cat{i + 1}")([x, s]), f, 0, f"u_dec{i + 1}")
        out = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="out")(x)
        return keras.Model(inp, out, name="unet")
    filters, drops = (32, 64, 128, 256, 512), (0.1, 0.1, 0.2, 0.2, 0.3)
    skips, x = [], inp
    for i, (f, d) in enumerate(zip(filters[:-1], drops[:-1])):
        x = conv_block(x, f, d, f"enc{i + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = conv_block(x, filters[-1], drops[-1], "bott")
    if kind != "full":
        z = layers.GlobalAveragePooling2D(name="gap")(x)
        z = layers.Dense(n, activation="tanh", name=f"pre_dense{n}")(z)
        blk = make_block(kind, n, k)
        if blk is not None:
            z = blk(z)
        z = ReplaceCode(name="replace_code")(z)
        z = layers.Dense(filters[-1], activation="relu", name=f"post_dense{n}")(z)
        z = layers.Reshape((1, 1, filters[-1]), name="to_map")(z)
        x = layers.UpSampling2D(size // 16, interpolation="nearest", name="tile")(z)
    for i, (f, s, d) in enumerate(zip(reversed(filters[:-1]), reversed(skips), reversed(drops[:-1]))):
        x = layers.UpSampling2D(2, name=f"up{i + 1}")(x)
        x = conv_block(layers.Concatenate(name=f"cat{i + 1}")([x, s]), f, d, f"dec{i + 1}")
    out = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="out")(x)
    return keras.Model(inp, out, name=name)


def block_layer(model):
    try:
        return model.get_layer("block")
    except ValueError:
        return None


def count_params(model):
    tot = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    blk = block_layer(model)
    return dict(trainable=tot, block=int(sum(np.prod(v.shape) for v in blk.trainable_weights)) if blk else 0)


# ============================================================================ initialization matching
def reference_init(n, seed):
    """Initial effective weights of model q{n} for this seed: {layer: [arrays]} plus the circuit angles."""
    set_seeds(seed)
    m = build_model(f"q{n}", k=1.0)
    ref = {l.name: [np.array(w) for w in l.get_weights()] for l in m.layers if l.get_weights() and l.name not in ("block", "replace_code")}
    ref["__angles__"] = [m.get_layer("block").effective()["angles"]]
    return ref


def apply_init(model, ref):
    """Copy every layer with the same name and shapes; circuit models (q, qp) also copy the angles.
    Returns the maximum absolute difference after copying (0 if the copy is exact)."""
    diff = 0.0
    for l in model.layers:
        if l.name in ref and l.get_weights():
            src = ref[l.name]
            if [w.shape for w in src] == [w.shape for w in l.get_weights()]:
                l.set_weights(src)
                diff = max(diff, max(float(np.abs(a - b).max()) for a, b in zip(src, l.get_weights())))
    blk = block_layer(model)
    if isinstance(blk, Circuit):
        blk.set_effective({"angles": ref["__angles__"][0]})
        diff = max(diff, float(np.abs(blk.effective()["angles"] - ref["__angles__"][0]).max()))
    return diff


# ============================================================================ data
def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def _rotate(img, angle, method):
    # rotation about the centre via a projective transform
    c, s = tf.cos(angle), tf.sin(angle)
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    t = tf.stack([c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0])
    out = tf.raw_ops.ImageProjectiveTransformV3(images=img[None], transforms=t[None], output_shape=tf.shape(img)[:2],
                                                fill_value=0.0, interpolation=method, fill_mode="CONSTANT")
    return out[0]


def augment(img, msk, seed):
    s = tf.random.experimental.stateless_split(seed, num=4)
    flip = tf.random.stateless_uniform([], s[0]) < CFG["flip"]
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
    msk = tf.cond(flip, lambda: tf.image.flip_left_right(msk), lambda: msk)
    a = tf.random.stateless_uniform([], s[1], -1.0, 1.0) * CFG["rot_deg"] * np.pi / 180
    img, msk = _rotate(img, a, "BILINEAR"), _rotate(msk, a, "NEAREST")
    img = img * tf.random.stateless_uniform([], s[2], 1 - CFG["bright"], 1 + CFG["bright"])
    img = tf.image.stateless_random_contrast(img, 1 - CFG["contrast"], 1 + CFG["contrast"], s[3])
    return tf.clip_by_value(img, 0.0, 1.0), msk


def to_float(x, y):
    return tf.cast(x, tf.float32)[..., None] / 255.0, tf.cast(y > 0, tf.float32)[..., None]


def train_ds(x, y, seed):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    seeds = tf.data.Dataset.random(seed=seed).batch(2)
    ds = tf.data.Dataset.zip((ds, seeds)).map(lambda xy, s: augment(xy[0], xy[1], s), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)


def eval_ds(x, y):
    return tf.data.Dataset.from_tensor_slices((x, y)).map(to_float).batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)


# ============================================================================ loss and metrics
def soft_dice(t, p, smooth=1.0):
    t, p = tf.cast(t, tf.float32), tf.cast(p, tf.float32)
    inter = tf.reduce_sum(t * p, axis=(1, 2, 3))
    return tf.reduce_mean((2 * inter + smooth) / (tf.reduce_sum(t, axis=(1, 2, 3)) + tf.reduce_sum(p, axis=(1, 2, 3)) + smooth))


def bce_dice(t, p):
    return tf.reduce_mean(keras.losses.binary_crossentropy(t, p)) + 1.0 - soft_dice(t, p)


def hd95(t, p):
    from scipy import ndimage
    if p.sum() == 0 or t.sum() == 0:
        return np.nan
    bt = t ^ ndimage.binary_erosion(t)
    bp = p ^ ndimage.binary_erosion(p)
    dt = ndimage.distance_transform_edt(~bt)
    dp = ndimage.distance_transform_edt(~bp)
    return float(np.percentile(np.concatenate([dt[bp], dp[bt]]), 95))


def per_image(y_true, prob, thr=0.5):
    rows = []
    for t, p in zip(y_true > 0, prob[..., 0] >= thr):
        tp, fp, fn = np.sum(t & p), np.sum(~t & p), np.sum(t & ~p)
        rows.append(dict(dice=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
                         iou=tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
                         precision=tp / (tp + fp) if (tp + fp) else np.nan,
                         recall=tp / (tp + fn) if (tp + fn) else np.nan,
                         hd95=hd95(t, p), empty=int(p.sum() == 0), pred_pixels=int(p.sum())))
    return rows


# ============================================================================ training
class Stop(keras.callbacks.Callback):
    """Early stopping on val_loss with a minimum number of epochs; keeps the best weights in memory."""

    def __init__(self, patience, min_epochs):
        super().__init__()
        self.patience, self.min_epochs = patience, min_epochs
        self.best, self.wait, self.best_epoch, self.best_weights = np.inf, 0, -1, None

    def on_epoch_end(self, epoch, logs=None):
        v = logs["val_loss"]
        if v < self.best:
            self.best, self.wait, self.best_epoch = v, 0, epoch
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch + 1 >= self.min_epochs:
                self.model.stop_training = True


def fit(model, xtr, ytr, xva, yva, seed, max_epochs=None, log=None):
    model.compile(optimizer=keras.optimizers.Adam(CFG["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-7), loss=bce_dice,
                  metrics=[soft_dice], jit_compile=False)
    stop = Stop(CFG["es_patience"], CFG["min_epochs"])
    cbs = [stop, keras.callbacks.ReduceLROnPlateau("val_loss", factor=CFG["lr_factor"], patience=CFG["lr_patience"],
                                                    min_lr=CFG["min_lr"], verbose=0)]
    if log:
        cbs.append(keras.callbacks.CSVLogger(log))
    t0 = time.time()
    h = model.fit(train_ds(xtr, ytr, seed), validation_data=eval_ds(xva, yva), epochs=max_epochs or CFG["max_epochs"],
                  callbacks=cbs, verbose=0)
    model.set_weights(stop.best_weights)
    return dict(epochs=len(h.history["loss"]), best_epoch=stop.best_epoch + 1, best_val_loss=float(stop.best),
                minutes=(time.time() - t0) / 60, final_lr=float(keras.ops.convert_to_numpy(model.optimizer.learning_rate)))


def predict(model, x):
    return model.predict(eval_ds(x, np.zeros_like(x)), verbose=0)


# ============================================================================ audit
def codes(model, x):
    """Block output (input of the replace-code layer) for images x."""
    sub = keras.Model(model.input, model.get_layer("replace_code").input)
    return sub.predict(eval_ds(x, np.zeros_like(x)), verbose=0)


def block_inputs(model, x):
    n = parse(model.name)[1]
    sub = keras.Model(model.input, model.get_layer(f"pre_dense{n}").output)
    return sub.predict(eval_ds(x, np.zeros_like(x)), verbose=0)


def set_replace(model, mode, bank=None, perm=None, ptr=0):
    """mode 0 normal, 1 const (mean of bank), 2 swap (bank[perm[(i + ptr) mod 4096]] for the i-th image of a batch)."""
    rc = model.get_layer("replace_code")
    if bank is not None:
        bank = np.asarray(bank, np.float32)[:rc.bank_size]
        rc.bank.assign(np.resize(bank, (rc.bank_size, bank.shape[1])))   # repeats the codes if fewer than bank_size
    if perm is not None:
        rc.idx.assign(np.asarray(perm, np.int32))
    rc.ptr.assign(ptr)
    rc.mode.assign(float(mode))


def code_replacement(model, xtr, xte, yte, n_draws=10, seed=0):
    """Mean test Dice with normal, const and swap codes (swap averaged over n_draws fixed permutations)."""
    bank = codes(model, xtr)
    rng = np.random.default_rng(seed)
    out = {}
    set_replace(model, 0, bank)
    out["normal"] = np.array([r["dice"] for r in per_image(yte, predict(model, xte))])
    set_replace(model, 1, bank)
    out["const"] = np.array([r["dice"] for r in per_image(yte, predict(model, xte))])
    sw = []
    for d in range(n_draws):
        set_replace(model, 2, bank, perm=rng.integers(0, min(len(bank), 256), 4096), ptr=0)
        sw.append([r["dice"] for r in per_image(yte, predict(model, xte))])
    out["swap"] = np.mean(np.array(sw), axis=0)
    set_replace(model, 0)
    return out


def weight_change(model, init):
    """Relative change of the block, pre_dense and post_dense weights (effective units)."""
    n = parse(model.name)[1]
    res = {}
    for nm in (f"pre_dense{n}", f"post_dense{n}"):
        a = np.concatenate([w.ravel() for w in init[nm]])
        b = np.concatenate([w.ravel() for w in model.get_layer(nm).get_weights()])
        res[nm.rstrip("0123456789")] = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))
    return res


def block_change(blk, init_eff):
    a = np.concatenate([init_eff[k].ravel() for k in sorted(init_eff)])
    b = blk.effective()
    b = np.concatenate([b[k].ravel() for k in sorted(b)])
    if isinstance(blk, Circuit):
        d = (b - a + np.pi) % (2 * np.pi) - np.pi
        return float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-12))
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))


def gradient_share(model, x, y, n_batches=4):
    blk = block_layer(model)
    if blk is None:
        return np.nan
    ids = {id(v) for v in blk.trainable_weights}
    num = den = 0.0
    for i in range(n_batches):
        xb, yb = to_float(x[i * 16:(i + 1) * 16], y[i * 16:(i + 1) * 16])
        with tf.GradientTape() as tape:
            loss = bce_dice(yb, model(xb, training=False))
        g = tape.gradient(loss, model.trainable_weights)
        for v, gi in zip(model.trainable_weights, g):
            if gi is None:
                continue
            s = float(tf.reduce_sum(tf.square(gi)))
            den += s
            num += s if id(v) in ids else 0.0
    return float(np.sqrt(num / den)) if den else np.nan


def entanglement(model, x):
    blk = block_layer(model)
    if not isinstance(blk, Circuit):
        return np.nan
    c = block_inputs(model, x)
    psi = qsim.circuit_state(c, blk.effective()["angles"], qsim.NumpyOps, blk.n, blk.cnot)
    return float(qsim.meyer_wallach(psi, blk.n).mean())


def surrogate_fit(model, xtr, xte):
    """R^2 on the test codes of a least-squares Fourier model fitted to the trained circuit on the training codes."""
    blk = block_layer(model)
    if not isinstance(blk, Circuit):
        return np.nan
    w = blk.effective()["angles"]
    ctr, cte = block_inputs(model, xtr), block_inputs(model, xte)
    ytr = qsim.circuit_expvals(ctr, w, qsim.NumpyOps, blk.n, blk.cnot)
    yte = qsim.circuit_expvals(cte, w, qsim.NumpyOps, blk.n, blk.cnot)
    F = qsim.fourier_features(ctr, qsim.NumpyOps, blk.n)
    coef, *_ = np.linalg.lstsq(F, ytr, rcond=None)
    pred = qsim.fourier_features(cte, qsim.NumpyOps, blk.n) @ coef
    ss = np.sum((yte - yte.mean(0)) ** 2)
    return float(1 - np.sum((yte - pred) ** 2) / ss) if ss > 0 else np.nan


def finite_shot_dice(model, xte, yte, shots=1024, repeats=5, seed=0):
    """Test Dice when each <Z_j> of the circuit is estimated from `shots` binomial draws (qubits drawn independently).
    The estimated codes are fed through the replace-code layer in image order (bank = codes of the test images)."""
    blk = block_layer(model)
    assert len(xte) <= model.get_layer("replace_code").bank_size, "test set larger than the code bank"
    c = block_inputs(model, xte)
    ez = qsim.circuit_expvals(c, blk.effective()["angles"], qsim.NumpyOps, blk.n, blk.cnot)
    rng = np.random.default_rng(seed)
    res = []
    for r in range(repeats):
        est = 1 - 2 * rng.binomial(shots, np.clip((1 - ez) / 2, 0, 1)) / shots
        set_replace(model, 2, bank=est.astype(np.float32), perm=np.arange(4096) % len(est), ptr=0)
        res.append([d["dice"] for d in per_image(yte, predict_in_order(model, xte))])
    set_replace(model, 0)
    return np.mean(np.array(res), axis=0)


def predict_in_order(model, x):
    """Predict batch by batch with the swap pointer at the batch start, so image i receives bank code i."""
    rc = model.get_layer("replace_code")
    out, bs = [], CFG["batch_size"]
    for i in range(0, len(x), bs):
        rc.ptr.assign(i)
        xb = tf.cast(x[i:i + bs], tf.float32)[..., None] / 255.0
        out.append(model(xb, training=False).numpy())
    rc.ptr.assign(0)
    return np.concatenate(out)


def env_info():
    import platform
    info = dict(python=platform.python_version(), tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__)
    try:
        info["gpu"] = tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name")
    except Exception:
        info["gpu"] = None
    return info
