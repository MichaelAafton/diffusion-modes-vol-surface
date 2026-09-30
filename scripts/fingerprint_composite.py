"""Composite-model fingerprint: level factor + bending membrane vs real SPX.

Runs a composite cell over several seeds and reports per-mode variance
shares as mean +/- std, next to the study reference spectrum. Seed spread = the denominator for any
claim that a residual is significant.
"""
import numpy as np
from src.data_pipeline import load_study_spectrum
from src.simulate import StochasticHeatEquation, SPDEConfig
from src.pca import run_pca, slope

N_DAYS, N_MODES = 2500, 40
Z_MIN, Z_MAX = -1.15, 1.10
SEEDS = [0, 1, 2, 3, 4]

ref = load_study_spectrum()
REAL = ref["shares"]
Z_POINTS = tuple(ref["z_centers"])
N_Z = len(Z_POINTS)

# Composite cell in wavenumber units (D: z^2/day, kappa: z^4/day). These are
# the maturity-averaged hand-fit values (index units D=0.2, kappa=0.02)
# converted by (L/pi)^2 and (L/pi)^4 -- not a fit to the study panel.
D_CELL, KAPPA_CELL, LEVEL_STD = 0.1026, 0.005262, 3.35

spectra, tails = [], []
for seed in SEEDS:
    cfg = SPDEConfig(D=D_CELL, kappa=KAPPA_CELL, n_modes=N_MODES,
                     z_min=Z_MIN, z_max=Z_MAX, z_points=Z_POINTS, seed=seed,
                     include_mean_mode=True, mean_mode_noise_std=LEVEL_STD)
    surface = StochasticHeatEquation(cfg).simulate(n_days=N_DAYS)
    pca = run_pca(np.diff(np.asarray(surface, float), axis=0),
                  n_components=N_Z)
    spectra.append(pca.explained_variance_ratio)
    tails.append(slope(pca.explained_variance_ratio[1:], k_start=2))

spectra = np.array(spectra)              # (n_seeds, n_bins)
mean, std = spectra.mean(axis=0), spectra.std(axis=0, ddof=1)

print("mode |  real   | synth mean ± std   | gap/std")
for i in range(N_Z):
    gap = (mean[i] - REAL[i]) / std[i]
    print(f"  {i+1}  | {REAL[i]:.4f}  | {mean[i]:.4f} ± {std[i]:.4f} | {gap:+.1f}")

print(f"\nmode-1 share: {mean[0]:.4f} ± {std[0]:.4f}")
print(f"synthetic p_tail: {np.mean(tails):.3f} ± {np.std(tails, ddof=1):.3f}")