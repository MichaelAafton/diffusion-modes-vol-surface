import wrds
import pandas as pd

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
pd.concat(surf).to_parquet("data/raw/spx_vsurface.parquet")
pd.concat(spot).to_parquet("data/raw/spx_spot.parquet")
print("Done!")