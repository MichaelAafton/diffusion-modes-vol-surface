import numpy as np
from src.data_pipeline import (
    ReparamConfig,
    load_optionmetrics,
    preprocess_options_data,
    build_surface_panel,
)

config = ReparamConfig(z_min=-1.15, z_max=1.10, n_z_bins=8)

df = load_optionmetrics()
out = preprocess_options_data(df, config)
s30 = out[out["days"] == 30]

result = build_surface_panel(s30, config)
panel = result["panel"]

print("panel shape:", panel.shape)
print("bins dropped:", result["n_bins_dropped"])
print("days dropped:", s30["date"].nunique() - panel.shape[0])
print("z centers:", np.round(result["z_centers"], 3))
print("any NaN:", panel.isna().any().any())