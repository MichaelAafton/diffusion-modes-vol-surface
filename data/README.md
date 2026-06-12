# Data Directory

Data files are **not** tracked by git (see `.gitignore`); only the directory
structure is kept via `.gitkeep` files.

| Subdir | Contents |
|--------|----------|
| `raw/` | Raw options data from a real source (see below). |
| `processed/` | Cleaned data reparameterized into `(z, τ)` coordinates. |
| `synthetic/` | Output of the stochastic-heat-equation simulator — the **validation harness**, not the study. |

## Real data sources

In order of preference (see the project README for detail):

1. **WRDS / OptionMetrics (IvyDB)** — standard academic source for historical
   equity implied vols and Greeks; free if your institution subscribes.
2. **Deribit BTC/ETH options** — free public API, several years of liquid
   crypto-option history.
3. **yfinance** — current chains only; a snapshot sanity check, not a basis for
   the time-series PCA.

Pick one real source early and commit to it — every later phase depends on it.
