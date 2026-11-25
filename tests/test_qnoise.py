import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import qsim, qnoise

rng = np.random.default_rng(1)
for n in (2, 4):
    for cnot in (True, False):
        x = rng.uniform(-1, 1, (8, n)); w = rng.uniform(0, 2 * np.pi, (2, n, 3))
        d = np.abs(qsim.circuit_expvals(x, w, qsim.NumpyOps, n, cnot) - qnoise.noisy_expvals(x, w, n, cnot)).max()
        assert d < 1e-10, (n, cnot, d)
x = rng.uniform(-1, 1, (4, 4)); w = rng.uniform(0, 2 * np.pi, (2, 4, 3))
assert np.abs(qnoise.noisy_expvals(x, w, 4, p1=0.75)).max() < 1e-10           # fully depolarised -> <Z> = 0
assert np.allclose(qnoise.noisy_expvals(x, w, 4, gamma=1.0), 1.0)              # full damping -> |0...0>
assert np.abs(qnoise.noisy_expvals(x, w, 4, pm=0.5)).max() < 1e-12             # random readout -> <Z> = 0
prev = 0.0
for lv in ('low', 'mid', 'high'):                                               # deviation grows with the level
    dev = np.abs(qnoise.noisy_expvals(x, w, 4, **qnoise.LEVELS[lv]) - qsim.circuit_expvals(x, w, qsim.NumpyOps, 4)).mean()
    assert dev > prev; prev = dev
print('qnoise tests OK')
