import pandas as pd

surf = pd.read_parquet("E:/Projects/vol-surface-dynamics/data/raw/spx_vsurface.parquet")
#spot = pd.read_parquet("E:/Projects/vol-surface-dynamics/data/raw/spx_spot.parquet")

print(surf["delta"].describe())
print(surf["delta"].unique()[:10])
