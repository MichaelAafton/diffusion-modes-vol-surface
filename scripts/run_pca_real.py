import numpy as np
from src.data_pipeline import load_optionmetrics, preprocess_options_data, \
     ReparamConfig, build_surface_panel
from src.pca import run_pca

def slope(evr):
    k = np.arange(1, len(evr) + 1)
    return -np.polyfit(np.log(k), np.log(evr), 1)[0]

def sign_changes(v):
    return int(np.sum(np.diff(np.sign(v)) != 0))

cfg = ReparamConfig(z_min=-1.15, z_max=1.10, n_z_bins=8)
df  = preprocess_options_data(load_optionmetrics(), cfg)
out = build_surface_panel(df, cfg)

print("panel:", out["panel"].shape, "| bins dropped:", out["n_bins_dropped"])
print("coverage:", out["coverage"].round(4).tolist())

# headline: all 8 bins
pca8 = run_pca(np.diff(out["panel"].values, axis=0), n_components=8)
print("\n8-bin ratios:", np.round(pca8.explained_variance_ratio, 4),
      " p =", round(slope(pca8.explained_variance_ratio), 3))
for m in range(3):
    v = pca8.eigenvectors[m]
    print(f"mode {m+1}: {np.round(v, 3)}  nodes={sign_changes(v)}")

# robustness: drop marginal deep-put bin
pca7 = run_pca(np.diff(out["panel"].values[:, 1:], axis=0), n_components=7)
print("\n7-bin ratios:", np.round(pca7.explained_variance_ratio, 4),
      " p =", round(slope(pca7.explained_variance_ratio), 3))
for m in range(3):
    v = pca7.eigenvectors[m]
    print(f"mode {m+1}: {np.round(v, 3)}  nodes={sign_changes(v)}")