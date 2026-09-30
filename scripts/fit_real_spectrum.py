"""Calibrate the composite model to the real SPX spectrum, with
bootstrap-refit confidence intervals."""
import numpy as np
from pathlib import Path
from src.calibration import fit_composite, model_spectrum, crossover_mode_index
from src.data_pipeline import load_study_spectrum

ref = load_study_spectrum()
real = ref["shares"]
z = ref["z_centers"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sigma = np.load(PROJECT_ROOT / "data" / "processed" / "real_spectrum_boot_std.npy")
boot  = np.load(PROJECT_ROOT / "data" / "processed" / "real_spectrum_boot.npy")  # (1000, 8)

# Best fit to the real spectrum
best = fit_composite(real, sigma, z)
print("best fit:",
      {k: round(v, 4) if isinstance(v, (float, np.floating)) else v
       for k, v in best.items()},
      f" dof = {best['dof']}")

# Residual table: best-fit model vs real, per-mode z-scores
model_best = model_spectrum(best["D"], best["kappa"], best["mean_inc_var"], z)
print("\nmode |  real   |  model  |    z")
for i in range(len(real)):
    z = (model_best[i] - real[i]) / sigma[i]
    print(f"  {i+1}  | {real[i]:.4f} | {model_best[i]:.4f} | {z:+.2f}")

# Bootstrap refits
fits = np.array([[f["D"], f["kappa"], f["mean_inc_var"]]
                 for b in boot
                 for f in [fit_composite(b, sigma, z)]])
lo, hi = np.percentile(fits, [2.5, 97.5], axis=0)
med = np.median(fits, axis=0)

# 1. Compare optimiser minimum with previous hand-fit point
# (maturity-averaged hand-fit, index units 0.2 / 0.02 converted to wavenumber units)
hand_D, hand_kappa, hand_m = 0.1026, 0.005262, 11.2
hand_model = model_spectrum(hand_D, hand_kappa, hand_m, z)
n_fit = len(real) - 1
hand_chi2 = np.sum(((hand_model[:n_fit] - real[:n_fit]) / sigma[:n_fit]) ** 2)
print(f"\nHand-fit point: D={hand_D}, kappa={hand_kappa}, m={hand_m}")
print(f"  chi2 = {hand_chi2:.4f}")
print(f"Optimised chi2 = {best['chi2']:.4f}")
print(f"Improvement = {hand_chi2 - best['chi2']:.4f}")
print("(flat valley if < ~3.5; real improvement if > ~8)")

# 2. Bootstrap-propagated k* (as a mode index)
k_star_boot = crossover_mode_index(fits[:, 0], fits[:, 1])
k_lo, k_med, k_hi = np.percentile(k_star_boot, [2.5, 50, 97.5])
print(f"\nk*: {k_med:.2f}  95% CI [{k_lo:.2f}, {k_hi:.2f}]")

# 3. Bootstrap-propagated sqrt(m) = mean_mode_noise_std
s_lo, s_med, s_hi = np.percentile(np.sqrt(fits[:, 2]), [2.5, 50, 97.5])
print(f"mean_mode_noise_std: {s_med:.4f}  95% CI [{s_lo:.4f}, {s_hi:.4f}]")

# 4. D-kappa parameter degeneracy
corr = np.corrcoef(np.log(fits[:, 0]), np.log(fits[:, 1]))[0, 1]
print(f"corr(log D, log kappa) = {corr:.3f}")

# Parameter confidence intervals
for name, i in [("D", 0), ("kappa", 1), ("mean_inc_var", 2)]:
    print(f"{name}: {med[i]:.4f}  95% CI [{lo[i]:.4f}, {hi[i]:.4f}]")
