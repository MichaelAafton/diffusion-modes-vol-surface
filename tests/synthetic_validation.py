import numpy as np
import matplotlib.pyplot as plt
from src.simulate import StochasticHeatEquation

# 1. Generate the synthetic surface. We KNOW its ground truth (built from
#    cosine modes with eigenvalues ~ 1/k^2), so it's the perfect test case.
equation = StochasticHeatEquation()
z, p, gt = equation.simulate(n_days=500)

# p has shape (n_days, n_z): each ROW is one day's surface across moneyness z.
print("p shape (days, z):", p.shape)
print("z grid: from", z[0], "to", z[-1], "with", len(z), "points")

# 2. LOOK before you compute. Plot a few individual days across z.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for day in [0, 500, 1000, 1500]:
    axes[0].plot(z, p[day], label=f"day {day}")
axes[0].set_xlabel("z (moneyness)")
axes[0].set_ylabel("surface perturbation p(z)")
axes[0].set_title("A few individual days")
axes[0].legend()

# 3. Heatmap of the whole thing: days on y, z on x.
im = axes[1].imshow(p, aspect="auto", origin="lower",
                    extent=[z[0], z[-1], 0, p.shape[0]], cmap="RdBu_r")
axes[1].set_xlabel("z (moneyness)")
axes[1].set_ylabel("day")
axes[1].set_title("Full surface over time")
fig.colorbar(im, ax=axes[1], label="p(z,t)")

plt.tight_layout()
plt.show()   # in a notebook use plt.show(); I used savefig only because I can't display interactively