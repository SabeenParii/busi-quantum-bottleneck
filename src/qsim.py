"""Exact state-vector simulation of the bottleneck circuit for n qubits (n <= 8), NumPy or TensorFlow.

Circuit: RY(pi/2 * c_j) on qubit j; L x [Rot(phi, theta, omega) on every qubit, then (optionally) a ring of CNOT
gates j -> j+1, n-1 -> 0]; outputs <Z_j>. Wire 0 is the most significant bit (PennyLane convention). With
cnot=False the circuit prepares a product state (the product-state control).
"""
import numpy as np

_LETTERS = "acdefghi"


def _cnot_perm(n, control, target):
    perm = np.arange(2 ** n)
    for k in range(2 ** n):
        if (k >> (n - 1 - control)) & 1:
            perm[k] = k ^ (1 << (n - 1 - target))
    return perm


def ring_perms(n):
    if n == 1:
        return []
    pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 1 else [])
    return [_cnot_perm(n, c, t) for c, t in pairs]


def circuit_state(x, w, ops, n, cnot=True):
    """x: (B, n) in [-1, 1]; w: (L, n, 3). Returns the state, shape (B, 2, ..., 2), complex."""
    theta = x * (np.pi / 2.0)
    qubits = [ops.stack([ops.cos(theta[:, i] / 2), ops.sin(theta[:, i] / 2)], axis=1) for i in range(n)]
    idx = _LETTERS[:n]
    psi = ops.to_complex(ops.einsum(",".join("b" + c for c in idx) + "->b" + idx, *qubits))
    perms = ring_perms(n) if cnot else []
    for l in range(ops.shape(w)[0]):
        for i in range(n):
            phi, th, om = w[l, i, 0], w[l, i, 1], w[l, i, 2]
            c, s = ops.cos(th / 2), ops.sin(th / 2)
            a, b = (phi + om) / 2, (phi - om) / 2
            u = ops.stack([
                ops.stack([ops.complex(ops.cos(a) * c, -ops.sin(a) * c), ops.complex(-ops.cos(b) * s, -ops.sin(b) * s)]),
                ops.stack([ops.complex(ops.cos(b) * s, -ops.sin(b) * s), ops.complex(ops.cos(a) * c, ops.sin(a) * c)]),
            ])
            psi = ops.einsum("z" + idx[i] + ",b" + idx + "->b" + idx[:i] + "z" + idx[i + 1:], u, psi)
            psi = ops.reshape(psi, [-1] + [2] * n)
        if perms:
            flat = ops.reshape(psi, [-1, 2 ** n])
            for p in perms:
                flat = ops.gather(flat, p)
            psi = ops.reshape(flat, [-1] + [2] * n)
    return psi


def circuit_expvals(x, w, ops, n, cnot=True):
    psi = circuit_state(x, w, ops, n, cnot)
    prob = ops.abs2(psi)
    idx = _LETTERS[:n]
    z = ops.const([1.0, -1.0])
    return ops.stack([ops.einsum("b" + idx + "," + idx[i] + "->b", prob, z) for i in range(n)], axis=1)


def fourier_features(x, ops, n):
    """All 3^n products of {1, cos(pi x_j / 2), sin(pi x_j / 2)}; x: (B, n) -> (B, 3^n)."""
    th = x * (np.pi / 2.0)
    feats = None
    for j in range(n):
        f = ops.stack([ops.ones_like(th[:, j]), ops.cos(th[:, j]), ops.sin(th[:, j])], axis=1)   # (B, 3)
        feats = f if feats is None else ops.reshape(ops.einsum("bi,bj->bij", feats, f), [-1, 3 ** (j + 1)])
    return feats


def meyer_wallach(psi_np, n):
    """Meyer-Wallach entanglement of each state in psi_np (B, 2, ..., 2) complex numpy array."""
    B = psi_np.shape[0]
    out = np.zeros(B)
    for j in range(n):
        a = np.moveaxis(psi_np, j + 1, 1).reshape(B, 2, -1)
        rho = np.einsum("bik,bjk->bij", a, a.conj())
        out += np.real(np.einsum("bij,bji->b", rho, rho))
    return 2.0 * (1.0 - out / n)


class NumpyOps:
    stack = staticmethod(lambda xs, axis=0: np.stack(xs, axis=axis))
    cos, sin = staticmethod(np.cos), staticmethod(np.sin)
    einsum = staticmethod(np.einsum)
    to_complex = staticmethod(lambda a: a.astype(complex))
    complex = staticmethod(lambda re, im: re + 1j * im)
    reshape = staticmethod(lambda a, s: np.reshape(a, s))
    gather = staticmethod(lambda a, p: a[:, p])
    abs2 = staticmethod(lambda a: np.abs(a) ** 2)
    const = staticmethod(lambda v: np.array(v))
    shape = staticmethod(lambda a: np.shape(a))
    ones_like = staticmethod(np.ones_like)


def tf_ops():
    import tensorflow as tf

    class TFOps:
        stack = staticmethod(lambda xs, axis=0: tf.stack(xs, axis=axis))
        cos, sin = staticmethod(tf.cos), staticmethod(tf.sin)
        einsum = staticmethod(tf.einsum)
        to_complex = staticmethod(lambda a: tf.cast(a, tf.complex64))
        complex = staticmethod(lambda re, im: tf.complex(re, im))
        reshape = staticmethod(lambda a, s: tf.reshape(a, s))
        gather = staticmethod(lambda a, p: tf.gather(a, p, axis=1))
        abs2 = staticmethod(lambda a: tf.math.real(a * tf.math.conj(a)))
        const = staticmethod(lambda v: tf.constant(v, tf.float32))
        shape = staticmethod(lambda a: a.shape)
        ones_like = staticmethod(tf.ones_like)

    return TFOps
