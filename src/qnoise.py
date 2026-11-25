"""Density-matrix simulation of the bottleneck circuit with synthetic noise (inference only).

Same circuit as qsim.circuit_expvals: RY(pi/2 * c_j) encoding, L x [Rot on every qubit, then a CNOT ring], <Z_j>.
Noise model (applied after every gate, as in common NISQ benchmarks):
  single-qubit gates (encoding RY and Rot): depolarising p1 and amplitude damping gamma on that qubit;
  CNOT: two-qubit depolarising p2 on control and target, and amplitude damping gamma on both;
  measurement: each Z outcome flipped with probability pm (readout error).
Channel strengths are synthetic levels, not calibration data of a device.
"""
import numpy as np

LEVELS = {
    "low":  dict(p1=1e-4, p2=2e-3, gamma=1e-4, pm=5e-3),
    "mid":  dict(p1=5e-4, p2=8e-3, gamma=5e-4, pm=1.5e-2),
    "high": dict(p1=2e-3, p2=2e-2, gamma=2e-3, pm=3e-2),
}

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = [_I, _X, _Y, _Z]


def _embed(ops_by_wire, n):
    """Kronecker product over wires 0..n-1 (wire 0 most significant); missing wires get the identity."""
    m = np.array([[1.0 + 0j]])
    for j in range(n):
        m = np.kron(m, ops_by_wire.get(j, _I))
    return m


def _rot(phi, th, om):
    c, s = np.cos(th / 2), np.sin(th / 2)
    a, b = (phi + om) / 2, (phi - om) / 2
    return np.array([[np.exp(-1j * a) * c, -np.exp(1j * b) * s],
                     [np.exp(-1j * b) * s, np.exp(1j * a) * c]])


def _cnot(n, c, t):
    dim = 2 ** n
    P = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        k2 = k ^ (1 << (n - 1 - t)) if (k >> (n - 1 - c)) & 1 else k
        P[k2, k] = 1
    return P


def _unitary(rho, U):
    return U @ rho @ U.conj().T


def _depol1(rho, j, p, n):
    if p <= 0:
        return rho
    out = (1 - p) * rho
    for P in _PAULI[1:]:
        E = _embed({j: P}, n)
        out = out + (p / 3) * (E @ rho @ E.conj().T)
    return out


def _depol2(rho, a, b, p, n):
    if p <= 0:
        return rho
    out = (1 - p) * rho
    for i, Pa in enumerate(_PAULI):
        for k, Pb in enumerate(_PAULI):
            if i == 0 and k == 0:
                continue
            E = _embed({a: Pa, b: Pb}, n)
            out = out + (p / 15) * (E @ rho @ E.conj().T)
    return out


def _damp(rho, j, g, n):
    if g <= 0:
        return rho
    K0 = _embed({j: np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)}, n)
    K1 = _embed({j: np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)}, n)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def noisy_expvals(x, w, n, cnot=True, p1=0.0, p2=0.0, gamma=0.0, pm=0.0):
    """x: (B, n) codes in [-1, 1]; w: (L, n, 3) effective angles. Returns (B, n) <Z_j> after readout error."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    pairs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)] if (cnot and n > 1) else []
    cn = [_cnot(n, c, t) for c, t in pairs]
    zs = [np.real(np.diag(_embed({j: _Z}, n))) for j in range(n)]
    out = np.zeros((len(x), n))
    for b in range(len(x)):
        th = x[b] * (np.pi / 2.0)
        psi = np.array([1.0 + 0j])
        for j in range(n):
            psi = np.kron(psi, np.array([np.cos(th[j] / 2), np.sin(th[j] / 2)]))
        rho = np.outer(psi, psi.conj())
        for j in range(n):                                       # noise after each encoding gate
            rho = _damp(_depol1(rho, j, p1, n), j, gamma, n)
        for l in range(w.shape[0]):
            for j in range(n):
                rho = _unitary(rho, _embed({j: _rot(*w[l, j])}, n))
                rho = _damp(_depol1(rho, j, p1, n), j, gamma, n)
            for (c, t), P in zip(pairs, cn):
                rho = _unitary(rho, P)
                rho = _depol2(rho, c, t, p2, n)
                rho = _damp(_damp(rho, c, gamma, n), t, gamma, n)
        prob = np.clip(np.real(np.diag(rho)), 0, None)
        prob = prob / prob.sum()
        out[b] = [(1 - 2 * pm) * float(prob @ z) for z in zs]
    return out


def sample_expvals(ez, shots, rng):
    """Estimate each <Z_j> from `shots` measurements; ez already includes the readout error."""
    if shots is None:
        return ez
    return 1 - 2 * rng.binomial(shots, np.clip((1 - ez) / 2, 0, 1)) / shots
