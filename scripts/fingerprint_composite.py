"""Composite-model fingerprint: level factor + bending membrane vs real SPX.

Runs the winning composite cell (D=0.2, kappa=0.02, mean_mode_noise_std=3.35)
over several seeds and reports per-mode variance shares as mean +/- std,
next to the real Phase-3 spectrum. Seed spread = the denominator for any
claim that a residual is significant.
"""
import numpy as np
from src.simulate import StochasticHeatEquation, SPDEConfig
from src.pca import run_pca, slope

N_DAYS, N_Z = 2500, 8
Z_MIN, Z_MAX = -1.15, 1.10
SEEDS = [0, 1, 2, 3, 4]

REAL = np.array([0.915, 0.0493, 0.0164, 0.0067,
                 0.0064, 0.0029, 0.0019, 0.0014])

spectra, tails = [], []
for seed in SEEDS:
    cfg = SPDEConfig(D=0.2, kappa=0.02, n_z=N_Z, n_modes=N_Z,
                     z_min=Z_MIN, z_max=Z_MAX, seed=seed,
                     include_mean_mode=True, mean_mode_noise_std=3.35)
    surface = StochasticHeatEquation(cfg).simulate(n_days=N_DAYS)
    pca = run_pca(np.diff(np.asarray(surface, float), axis=0),
                  n_components=N_Z)
    spectra.append(pca.explained_variance_ratio)
    tails.append(slope(pca.explained_variance_ratio[1:], k_start=2))

spectra = np.array(spectra)              # (n_seeds, 8)
mean, std = spectra.mean(axis=0), spectra.std(axis=0, ddof=1)

print("mode |  real   | synth mean ± std   | gap/std")
for i in range(N_Z):
    gap = (mean[i] - REAL[i]) / std[i]
    print(f"  {i+1}  | {REAL[i]:.4f}  | {mean[i]:.4f} ± {std[i]:.4f} | {gap:+.1f}")

print(f"\nmode-1 share: {mean[0]:.4f} ± {std[0]:.4f}")
print(f"synthetic p_tail: {np.mean(tails):.3f} ± {np.std(tails, ddof=1):.3f}")