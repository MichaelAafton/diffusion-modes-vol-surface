"""Figures for the report. Fig1: spectrum; Fig2: mode shapes; Fig3: OOS ratios."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 1. Anchor project root and register it in sys.path so 'src' imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project module imports
from src.data_pipeline import load_optionmetrics, preprocess_options_data, \
     ReparamConfig, build_surface_panel
from src.pca import run_pca
from src.calibration import model_spectrum

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# 2. Setup project directories relative to project root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGDIR = PROJECT_ROOT / "report" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)  # Create figures folder if missing

# shared data
cfg = ReparamConfig(z_min=-1.15, z_max=1.10, n_z_bins=8)
out = build_surface_panel(preprocess_options_data(load_optionmetrics(), cfg), cfg)
panel = out["panel"].to_numpy(dtype=float)
zc = np.asarray(out["z_centers"])
changes = np.diff(panel, axis=0)
pca8 = run_pca(changes, n_components=8)
pca7 = run_pca(changes[:, 1:], n_components=7)

k = np.arange(1, 9)
real = pca8.explained_variance_ratio
boot = np.load(PROCESSED_DIR / "real_spectrum_boot.npy")
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

# Fig 1: spectrum
fit = model_spectrum(0.688, 0.0109, 6.99)
diff_best = model_spectrum(5.0, 0.0, 0.0)          # diffusion's best attempt (no level)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.errorbar(k, real, yerr=[real - lo, hi - real], fmt="o", ms=4,
            capsize=2, label="SPX (95% CI)", zorder=3)
ax.plot(k, fit, "s--", ms=3, label=r"composite fit ($D{=}0.69,\ \kappa{=}0.011$)")
ax.plot(k[:7], diff_best[:7], "^:", ms=3, label=r"pure diffusion (saturated, $\kappa{=}0$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(k); ax.set_xticklabels(k)
ax.set_xlabel("mode $k$"); ax.set_ylabel("variance share $\\lambda_k$")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGDIR / "fig1_spectrum.pdf")

# Fig 2: mode shapes
def signfix(v):                       # eigenvectors defined up to sign
    return v if v.sum() >= 0 else -v

fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
a.axhline(0, lw=0.5, color="grey")
a.plot(zc, signfix(pca8.eigenvectors[0]), "o-", ms=3, label="mode 1 (level)")
a.plot(zc, signfix(pca8.eigenvectors[1]), "s-", ms=3, label="mode 2 (skew)")
a.set_xlabel("$z$"); a.set_ylabel("loading"); a.set_title("(a) membrane-like modes")
a.legend(frameon=False, fontsize=7)

b.axhline(0, lw=0.5, color="grey")
b.plot(zc, signfix(pca8.eigenvectors[2]), "o-", ms=3, label="mode 3, 8 bins")
b.plot(zc[1:], signfix(pca7.eigenvectors[2]), "s--", ms=3, label="mode 3, edge bin removed")
b.set_xlabel("$z$"); b.set_title("(b) mode 3 follows the boundary")
b.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGDIR / "fig2_modes.pdf")

# Fig 3: OOS ratios (numbers from scripts/oos_risk_test.py)
ports = ["straddle", "risk rev.", "butterfly", "equal wt."]
emp3 = [0.4618, 0.6491, 1.1010, 0.4380]
field = [0.4605, 0.6423, 0.4947, 0.4412]
diag = [0.9045, 0.1760, 0.0722, 3.1489]

x = np.arange(len(ports)); w = 0.26
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.bar(x - w, emp3, w, label="empirical 3-factor")
ax.bar(x,     field, w, label="field theory (3 params)")
ax.bar(x + w, diag, w, label="diagonal")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(ports)
ax.set_ylabel("realised / predicted variance")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGDIR / "fig3_oos.pdf")
print("wrote 3 figures to", FIGDIR)