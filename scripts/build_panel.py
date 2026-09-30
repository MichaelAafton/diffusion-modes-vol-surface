"""Build the study panel from raw parquet and record the reference spectrum.

    python scripts/build_panel.py

Writes data/processed/study_spectrum.npz (shares, z-centres, panel shape),
which downstream scripts load instead of hardcoding the spectrum.
"""
from pathlib import Path

import numpy as np

from src.data_pipeline import build_study_panel
from src.pca import run_pca, slope

out = build_study_panel()
panel = out["panel"]
changes = np.diff(panel.to_numpy(dtype=float), axis=0)
evr = run_pca(changes).explained_variance_ratio
p_tail = slope(evr[1:], k_start=2)
k = np.arange(2, len(evr) + 1)
local = -np.diff(np.log(evr[1:])) / np.diff(np.log(k))

np.set_printoptions(precision=6, suppress=True)
print("maturities:     ", out["maturities"])
print("panel shape:    ", panel.shape)
print("coverage:       ", out["coverage"].round(4).to_dict())
print("bins dropped:   ", out["n_bins_dropped"], "| kept bins:", out["kept_bins"].tolist())
print("z centres:      ", np.round(out["z_centers"], 4))
print("dates:          ", panel.index.min().date(), "->", panel.index.max().date())
print("shares:         ", evr)
print(f"p_tail:          {p_tail:.3f}")
print("local slopes:   ", np.round(local, 2))

root = Path(__file__).resolve().parent.parent
path = root / "data" / "processed" / "study_spectrum.npz"
np.savez(path, shares=evr, z_centers=out["z_centers"],
         kept_bins=out["kept_bins"], panel_shape=np.array(panel.shape),
         p_tail=p_tail)
print("saved:          ", path)
