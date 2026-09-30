"""Out-of-sample covariance test for the 8-bin dSigma panel.

Three competing models are trained on the first 70% of the daily
Delta-sigma observations and evaluated on the remaining 30%.

Models:
    1. Empirical 3-factor PCA covariance
    2. Field-theory correlation + train-data standard deviations
    3. Diagonal covariance baseline

The reported ratio is:

    realised test variance / predicted variance

A ratio of 1 means perfect calibration.
"""

import numpy as np
import pandas as pd

from src.data_pipeline import build_study_panel
from src.calibration import model_spectrum



# Configuration

N_Z = 8
TRAIN_FRAC = 0.70

# Maturity-averaged Phase-5 fit (index units 0.688 / 0.0109) converted to
# wavenumber units; not a fit to the study panel.
D_FIT = 0.3529
KAPPA_FIT = 0.002868
M_FIT = 6.99

# Train/test split

def train_test_split(changes, train_frac=0.70):
    """Chronological split: earlier observations are train."""
    n_train = int(len(changes) * train_frac)
    return changes[:n_train], changes[n_train:]


# Risk models

def empirical_factor_cov(train, n_factors=3):
    """PCA-truncated covariance + diagonal residual top-up."""
    c = np.cov(train, rowvar=False)

    lam, V = np.linalg.eigh(c)
    lam = lam[::-1]
    V = V[:, ::-1]

    factors = V[:, :n_factors]
    factor_cov = (factors * lam[:n_factors]) @ factors.T

    residual_diag = np.diag(c) - np.diag(factor_cov)
    residual = np.diag(residual_diag)

    return factor_cov + residual


def field_theory_cov(train, corr_model):
    """Physics correlation structure in train-data units."""
    s = train.std(axis=0, ddof=1)
    return np.outer(s, s) * corr_model


def diagonal_cov(train):
    """Naive independent-bin covariance."""
    return np.diag(train.var(axis=0, ddof=1))


# Portfolio test

def portfolio_report(covs, test, portfolios):
    """Compare realised OOS variance against model predictions."""

    rows = []

    for pname, w in portfolios.items():
        w = np.asarray(w, dtype=float)

        # Realised portfolio daily Delta-sigma
        pnl = test @ w
        realised = np.var(pnl, ddof=1)

        row = {
            "portfolio": pname,
            "realised": realised * 1e4,
        }

        for model_name, Sigma in covs.items():
            predicted = w @ Sigma @ w
            row[model_name] = realised / predicted

        rows.append(row)

    return pd.DataFrame(rows)


# Main experiment

def main():

    # The study panel (single source of truth)

    out = build_study_panel()

    panel = out["panel"].to_numpy(dtype=float)

    print("Panel shape:", panel.shape)
    print("z centers:", np.asarray(out["z_centers"]))

    # Daily Delta-sigma

    changes = np.diff(panel, axis=0)

    print("Changes shape:", changes.shape)

    # Chronological split

    train, test = train_test_split(
        changes,
        train_frac=TRAIN_FRAC,
    )

    dates = out["panel"].index
    n_train = int(len(changes) * TRAIN_FRAC)

    print(f"Train: {dates[1]} -> {dates[n_train]}")
    print(f"Test:  {dates[n_train + 1]} -> {dates[-1]}")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # Model 1: empirical 3-factor PCA covariance

    Sigma_empirical_1 = empirical_factor_cov(
        train,
        n_factors=1,
    )

    Sigma_empirical_3 = empirical_factor_cov(
        train,
        n_factors=3,
    )

    Sigma_empirical_5 = empirical_factor_cov(
        train,
        n_factors=5,
    )

    # Model 2: field-theory correlation structure
    #
    # IMPORTANT:
    # model_spectrum(..., return_corr=True) must use the exact
    # same correlation construction as Phase 5.

    corr_model = model_spectrum(
        D_FIT,
        KAPPA_FIT,
        M_FIT,
        out["z_centers"],
        return_corr=True,
    )

    assert np.allclose(
        np.diag(corr_model),
        1.0,
    ), "corr_model diagonal != 1"

    Sigma_field = field_theory_cov(
        train,
        corr_model,
    )

    # Model 3: diagonal baseline

    Sigma_diag = diagonal_cov(train)

    covs = {
        "empirical_1factor": Sigma_empirical_1,
        "empirical_3factor": Sigma_empirical_3,
        "empirical_5factor": Sigma_empirical_5,
        "field_theory": Sigma_field,
        "diagonal": Sigma_diag,
    }

    # Fixed 8-bin portfolio exposures
    #
    # Bin ordering is:
    #
    #   left wing -> ... -> ATM -> ... -> right wing
    #
    # We use deliberately simple, transparent weights.
    #
    # Each vector represents exposure to Delta-sigma.

    # Equal-weight = pure level exposure
    equal_weight = np.ones(N_Z) / N_Z

    # ATM straddle = concentrated in the two central bins
    straddle = np.array([
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
    ])

    # Risk reversal = long left wing / short right wing
    risk_reversal = np.array([
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.5,
    ])

    # Butterfly = wings minus body
    butterfly = np.array([
        0.5,
        0.0,
        0.0,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.5,
    ])

    portfolios = {
        "straddle": straddle,
        "risk_reversal": risk_reversal,
        "butterfly": butterfly,
        "equal_weight": equal_weight,
    }

    n_bins = panel.shape[1]
    for name, w in portfolios.items():
        if len(w) != n_bins:
            raise ValueError(
                f"portfolio '{name}' has {len(w)} weights but the panel has "
                f"{n_bins} bins; define portfolios on the kept bins"
            )

    # Report

    report = portfolio_report(
        covs,
        test,
        portfolios,
    )

    report = report.set_index("portfolio")

    print("\nOut-of-sample risk test")
    print("=" * 90)
    print(
        report.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()