"""One-time pull of SPX (secid 108105) surfaces and spot from WRDS, 2015-2024.

Writes data/raw/spx_vsurface.parquet (IvyDB vsurfd) and data/raw/spx_spot.parquet
(secprd close). Requires a WRDS account with OptionMetrics access.

    python -m scripts.pull_wrds
"""
import pandas as pd
import wrds

from src.data_pipeline import RAW_DIR

print("Connecting...")
db = wrds.Connection()
print("Connected!")

surf = []
spot = []

for y in range(2015, 2025):
    print(f"Downloading {y} surface...")
    surf.append(
        db.raw_sql(f"""
        SELECT date, days, delta, cp_flag,
               impl_volatility,
               impl_strike,
               impl_premium
        FROM optionm.vsurfd{y}
        WHERE secid = 108105
        """)
    )

    print(f"Downloading {y} spot...")
    spot.append(
        db.raw_sql(f"""
        SELECT date,
               close AS spot
        FROM optionm.secprd{y}
        WHERE secid = 108105
        """)
    )

print("Saving...")
RAW_DIR.mkdir(parents=True, exist_ok=True)
pd.concat(surf).to_parquet(RAW_DIR / "spx_vsurface.parquet")
pd.concat(spot).to_parquet(RAW_DIR / "spx_spot.parquet")
print("Done!")
