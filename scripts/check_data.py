'''from src.data_pipeline import load_optionmetrics, preprocess_options_data

print("Loading OptionMetrics data...")
df = load_optionmetrics()

# -----------------------------
# Check 1: Date range
# -----------------------------
print("\n=== Check 1: Date range ===")
print("Min date:", df["date"].min())
print("Max date:", df["date"].max())

# -----------------------------
# Check 2: Rows per trading day
# -----------------------------
print("\n=== Check 2: Rows per day ===")
print(df.groupby("date").size().describe())

# -----------------------------
# Check 3: Preprocessing
# -----------------------------
print("\n=== Check 3: Preprocessing ===")

print("Rows before preprocessing:", len(df))

df = preprocess_options_data(df)

print("Rows after preprocessing:", len(df))
print("z range:", df["z"].min(), "to", df["z"].max())'''
from src.data_pipeline import ReparamConfig, preprocess_options_data, load_optionmetrics

df = load_optionmetrics()
wide = ReparamConfig(z_min=-5.0, z_max=5.0)
out = preprocess_options_data(df, wide)

s30 = out[out["days"] == 30]
print(len(s30), len(s30) / s30["date"].nunique())
print(s30["z"].quantile([0.005, 0.05, 0.5, 0.95, 0.995]))
print(s30.groupby("delta")["z"].median().round(3))
print(s30.groupby("delta")["z"].std().round(3))