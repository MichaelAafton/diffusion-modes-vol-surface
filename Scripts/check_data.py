from src.data_pipeline import load_optionmetrics, preprocess_options_data

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

df = preprocess_options_data(df)   # or whatever arguments your function needs

print("Rows after preprocessing:", len(df))
print("z range:", df["z"].min(), "to", df["z"].max())