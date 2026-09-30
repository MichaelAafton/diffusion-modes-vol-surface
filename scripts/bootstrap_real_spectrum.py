"""Block-bootstrap error bars on the real SPX change-spectrum."""

import numpy as np
from pathlib import Path
from src.data_pipeline import build_study_panel
from src.pca import run_pca

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

mean, std = spectra.mean(axis=0), spectra.std(axis=0, ddof=1)
lo, hi = np.percentile(spectra, [2.5, 97.5], axis=0)

print("mode |  point   | boot mean |   std    | 95% CI")
for i in range(n):
    print(f"  {i+1}  | {run_pca(changes, n_components=n).explained_variance_ratio[i]:.4f}"
          f"  | {mean[i]:.4f}   | {std[i]:.4f}  | [{lo[i]:.4f}, {hi[i]:.4f}]")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define final file path
output_path = PROJECT_ROOT / "data" / "processed" / "real_spectrum_boot.npy"

# Save the file
np.save(output_path, spectra)




