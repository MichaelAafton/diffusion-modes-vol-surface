"""Moving-block bootstrap of the study spectrum (block 25 days, 1000 replicates).

Prints point estimates with bootstrap mean, s.d. and 95% intervals for the shares,
the tail slope p_tail (modes 2-n), the local slopes and the tail-normalised shares
(modes 2-n divided by their sum), and writes
data/processed/real_spectrum_boot.npy (replicates x modes), which the test
scripts read for real-data intervals.

    python -m scripts.bootstrap_real_spectrum
"""
import numpy as np

from src.data_pipeline import PROCESSED_DIR, build_study_panel
from src.pca import run_pca, slope

BLOCK, N_BOOT, SEED = 25, 1000, 0

changes = np.diff(build_study_panel()["panel"].values, axis=0)

T, n = changes.shape
rng = np.random.default_rng(SEED)
n_blocks = int(np.ceil(T / BLOCK))

spectra = np.empty((N_BOOT, n))
for b in range(N_BOOT):
    starts = rng.integers(0, T - BLOCK + 1, size=n_blocks)
    sample = np.concatenate([changes[s:s + BLOCK] for s in starts])[:T]
    spectra[b] = run_pca(sample, n_components=n).explained_variance_ratio

point = run_pca(changes, n_components=n).explained_variance_ratio
mean, std = spectra.mean(axis=0), spectra.std(axis=0, ddof=1)
lo, hi = np.percentile(spectra, [2.5, 97.5], axis=0)

print("mode |  point   | boot mean |   std    | 95% CI")
for i in range(n):
    print(f"  {i+1}  | {point[i]:.4f}  | {mean[i]:.4f}   | {std[i]:.4f}  | [{lo[i]:.4f}, {hi[i]:.4f}]")



def tail_slope(s):
    return slope(s[1:], k_start=2)


def local_slopes(s):
    k = np.arange(2, len(s) + 1)
    return -np.diff(np.log(s[1:])) / np.diff(np.log(k))


def tail_shares(s):
    return s[1:] / s[1:].sum()


np.set_printoptions(precision=3, suppress=True)
for name, f in (("p_tail", tail_slope), ("local slopes", local_slopes),
                ("tail shares", tail_shares)):
    b = np.array([f(x) for x in spectra])
    lo_, hi_ = np.percentile(b, [2.5, 97.5], axis=0)
    print(f"\n{name}: point {np.round(f(point), 3)}")
    print(f"  95% lo {np.round(lo_, 3)}\n  95% hi {np.round(hi_, 3)}")

np.save(PROCESSED_DIR / "real_spectrum_boot.npy", spectra)
