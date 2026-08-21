import numpy as np
import pandas as pd

from src.simulate import StochasticHeatEquation, SPDEConfig
from src.pca import run_pca, slope


# Match the real-data instrument
N_DAYS = 2500
N_Z = 8

Z_MIN = -1.15
Z_MAX = 1.10
L = Z_MAX - Z_MIN

N_COMPONENTS = N_Z

# Repeat each parameter cell over several random seeds
SEEDS = [0, 1, 2, 3, 4]


# Parameter sweeps

# Sweep 1:
# kappa = 0, vary D
D_VALUES = [0.01, 0.05, 0.2, 1.0, 5.0]

# Sweep 2:
# D fixed, choose kappa so that
# k* = sqrt(D / kappa)
# moves through the observable wavenumber band.
KAPPA_VALUES = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]

D_FIXED = 0.2


'''def run_one(D: float, kappa: float, seed: int) -> tuple[float, float]:
    """
    Simulate one synthetic surface, take daily differences,
    run PCA, and return the fitted spectral slope p.
    """

    config = SPDEConfig(
        D=D,
        kappa=kappa,
        n_z=N_Z,
        z_min=Z_MIN,
        z_max=Z_MAX,
        seed=seed,
        n_modes=N_Z
    )

    model = StochasticHeatEquation(config)

    # exp underflow (a -> 0) is benign for the exact update: mode redraws from stationary dist.:
    # assert model.decay_rates.max() * config.dt < 700, "exp underflow territory"

    assert np.all(np.isfinite(model.decay_rates)), "non-finite decay rates"

    # Simulate daily surface snapshots
    surface = model.simulate(n_days=N_DAYS)

    # Daily increments
    changes = np.diff(
        np.asarray(surface, dtype=float),
        axis=0,
    )

    # PCA
    pca = run_pca(
        changes,
        n_components=N_COMPONENTS,
    )

    evr = pca.explained_variance_ratio

    p_full = slope(evr)
    p_tail = slope(evr[1:], k_start=2)

    return p_full, p_tail


def summarize_cell(D: float, kappa: float, p_values: list[tuple[float, float]], ) -> dict:
    """Summarize repeated p estimates for one (D, kappa) cell."""

    ps = np.array(p_values).T

    full_slopes = ps[0]
    tail_slopes = ps[1]

    result = {
        "D": D,
        "kappa": kappa,
        "p_mean": np.mean(full_slopes),
        "p_std": np.std(full_slopes, ddof=1),
        "p_tail_mean": np.mean(tail_slopes),
        "p_tail_std": np.std(tail_slopes, ddof=1),
    }

    # k* is undefined when kappa = 0
    if kappa > 0:
        result["k_star"] = np.sqrt(D / kappa)
    else:
        result["k_star"] = np.inf

    return result


def run_sweep() -> pd.DataFrame:
    rows = []

    # Sweep 1: kappa = 0, vary D
    for D in D_VALUES:
        kappa = 0.0

        ps = [
            run_one(D, kappa, seed)
            for seed in SEEDS
        ]

        rows.append({
            "sweep": "D",
            **summarize_cell(D, kappa, ps),
        })

    # Sweep 2: D fixed, vary kappa
    for kappa in KAPPA_VALUES:
        D = D_FIXED

        ps = [
            run_one(D, kappa, seed)
            for seed in SEEDS
        ]

        rows.append({
            "sweep": "kappa",
            **summarize_cell(D, kappa, ps),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_sweep()

    print("\nSynthetic SPDE spectral-slope sweep")
    print("=" * 100)

    print(
        results.to_string(
            index=False,
            formatters={
                "D": "{:.3g}".format,
                "kappa": "{:.3g}".format,
                "k_star": lambda x: (
                    "inf" if np.isinf(x) else f"{x:.3f}"
                ),
                "p_mean": "{:.3f}".format,
                "p_std": "{:.3f}".format,
                "p_tail_mean": "{:.3f}".format,
                "p_tail_std": "{:.3f}".format,
            },
        )
    )'''
# fingerprint test: full spectrum at the winning cell
config = SPDEConfig(D=0.2, kappa=0.02, n_z=N_Z, n_modes=N_Z,
                    z_min=Z_MIN, z_max=Z_MAX, seed=0, include_mean_mode=True, mean_mode_noise_std=3.35)
model = StochasticHeatEquation(config)
surface = model.simulate(n_days=N_DAYS)
changes = np.diff(np.asarray(surface, float), axis=0)
pca = run_pca(changes, n_components=N_Z)

real = np.array([0.915, 0.0493, 0.0164, 0.0067,
                 0.0064, 0.0029, 0.0019, 0.0014])
synth = pca.explained_variance_ratio

print("mode | real    | synthetic")
for i in range(N_Z):
    print(f"  {i+1}  | {real[i]:.4f}  | {synth[i]:.4f}")
print("mode-1 share:", round(synth[0], 4))
print("synthetic p_tail:", round(slope(synth[1:], k_start=2), 3))
'''import numpy as np
import pandas as pd

from src.simulate import StochasticHeatEquation, SPDEConfig
from src.pca import run_pca, slope

N_DAYS = 2500
N_Z = 8

Z_MIN = -1.15
Z_MAX = 1.10
L = Z_MAX - Z_MIN

N_COMPONENTS = N_Z

config = SPDEConfig(D=0.2, kappa=0.0, n_z=N_Z, n_modes=N_Z,
                    z_min=Z_MIN, z_max=Z_MAX, seed=0)
model = StochasticHeatEquation(config)
surface = model.simulate(n_days=N_DAYS)
changes = np.diff(np.asarray(surface, float), axis=0)

print("column stds:", np.round(changes.std(axis=0), 6))
pca = run_pca(changes, n_components=N_COMPONENTS if False else N_Z)
print("evr:", pca.explained_variance_ratio)'''