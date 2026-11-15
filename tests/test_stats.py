"""Sanity checks for stats.py on synthetic data with known answers.  python tests/test_stats.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from stats import bootstrap_ci, paired_bootstrap, mcnemar_exact, cohen_kappa

rng = np.random.default_rng(1)

# 1. CI should cover the true rate ~95% of the time
true_p, hits = 0.7, 0
for i in range(200):
    x = rng.random(200) < true_p
    _, lo, hi = bootstrap_ci(x, n_boot=2000, seed=i)
    hits += lo <= true_p <= hi
print(f"CI coverage: {hits/200:.0%} (expect ~95%)"); assert 0.88 <= hits/200 <= 0.99

# 2. paired test detects a real 10-point gap that unpaired CIs would overlap on
base = rng.random(300) < 0.6
b = base.copy(); flip = rng.choice(np.where(~b)[0], 30, replace=False); b[flip] = True
d, lo, hi, p = paired_bootstrap(b, base)
print(f"paired diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.4f}"); assert lo > 0 and p < 0.01
pm, *_ = mcnemar_exact(b, base); print(f"McNemar p={pm:.2e}"); assert pm < 0.01

# 3. no difference -> p should be large
d, lo, hi, p = paired_bootstrap(base, base); assert d == 0 and p == 1.0
assert mcnemar_exact(base, base)[0] == 1.0

# 4. kappa: perfect, chance, and a hand-checked case
y = rng.choice(["YES", "NO"], 500)
assert cohen_kappa(y, y) == 1.0
k_rand = cohen_kappa(y, rng.choice(["YES", "NO"], 500)); print(f"random kappa {k_rand:.3f}"); assert abs(k_rand) < 0.1
# 20 items: agree on 8 YES + 8 NO, disagree on 4 -> po=0.8, pe=0.5 -> kappa 0.6
h = ["YES"]*8 + ["NO"]*8 + ["YES"]*2 + ["NO"]*2
j = ["YES"]*8 + ["NO"]*8 + ["NO"]*2 + ["YES"]*2
assert abs(cohen_kappa(h, j) - 0.6) < 1e-9
print("all stats checks passed")
