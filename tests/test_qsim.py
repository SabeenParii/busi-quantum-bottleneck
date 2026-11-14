import sys, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
# optional: path to src/ of the companion repository gp-quantum-bottleneck-unet for the n=4 identity check
P2 = os.environ.get('P2_SRC')
if P2:
    sys.path.insert(0, P2)
import qsim
try:
    from qseg import quantum_sim as ref
except ImportError:
    ref = None
rng = np.random.default_rng(1); O = qsim.NumpyOps
# 1) n=4 identical to the circuit of P2
x = rng.uniform(-1, 1, (64, 4)); w = rng.uniform(0, 2*np.pi, (2, 4, 3))
if ref is not None:
    a = qsim.circuit_expvals(x, w, O, 4); b = ref.circuit_expvals(x, w, ref.NumpyOps, 4)
    print('match P2 n=4:', np.abs(a-b).max())
for n in (2, 4, 8):
    x = rng.uniform(-1, 1, (200, n)); w = rng.uniform(0, 2*np.pi, (2, n, 3))
    # 2) product state: output j depends only on x_j
    y = qsim.circuit_expvals(x, w, O, n, cnot=False); x2 = x.copy(); x2[:, 1:] = rng.uniform(-1, 1, (200, n-1))
    y2 = qsim.circuit_expvals(x2, w, O, n, cnot=False)
    print(n, 'product-state output 0 independent of others:', np.abs(y[:, 0]-y2[:, 0]).max())
    # 3) Fourier features span the circuit exactly (least squares on many points)
    xs = rng.uniform(-1, 1, (max(4*3**n, 400), n)); F = qsim.fourier_features(xs, O, n); Y = qsim.circuit_expvals(xs, w, O, n)
    coef, *_ = np.linalg.lstsq(F, Y, rcond=None); res = Y - F @ coef
    print(n, 'Fourier fit max residual:', np.abs(res).max(), 'features', F.shape[1])
    # 4) Meyer-Wallach: 0 for product states, > 0 with CNOT
    psi_p = qsim.circuit_state(x, w, O, n, cnot=False); psi_e = qsim.circuit_state(x, w, O, n, cnot=True)
    print(n, 'MW product max', qsim.meyer_wallach(psi_p, n).max().round(10), 'MW entangled mean', qsim.meyer_wallach(psi_e, n).mean().round(3))
    # 5) state normalised
    print(n, 'norm', np.abs((np.abs(psi_e.reshape(len(x), -1))**2).sum(1)-1).max())
