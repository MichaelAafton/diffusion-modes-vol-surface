"""
Data Pipeline & Reparameterization
====================================

Handles loading and preprocessing of options data, and implements the
critical reparameterization from raw (K, ttoexp) coordinates into the
normalised (z, τ) space where the heat equation structure becomes visible.

The reparameterization (Slide 5):

    Moneyness:   z = log(K/S) / σ      where σ = √VSW(ttoexp)
    Psych. time: τ = ψ log(1 + T/ψ)   where ψ ~ 20-30 bdays

In z-space, an option at z = -3 pays off on a 3σ crash, and z = +2
pays off on a 2σ rally — giving moneyness a direct probabilistic
interpretation regardless of the current spot level or volatility regime.

The τ-transform compresses long horizons: the difference between 1M and 2M
options is comparable to the difference between 1Y and 2Y options in
psychological terms (Slide 7).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ReparamConfig:
    """Configuration for the (z, τ) reparameterization.

    Parameters
    ----------
    psi : float
        Psychological time scale parameter (in business days).
        Controls the log-compression of the maturity axis.
        From the presentation, ψ ~ 20-30 business days works well.
    z_min : float
        Minimum moneyness to include (e.g. -3 for deep OTM puts).
    z_max : float
        Maximum moneyness to include (e.g. +3 for deep OTM calls).
    n_z_bins : int
        Number of bins for the discretised z-grid.
    n_tau_bins : int
        Number of bins for the discretised τ-grid.
    """
    psi: float = 25.0
    z_min: float = -3.0
    z_max: float = 3.0
    n_z_bins: int = 30
    n_tau_bins: int = 10


def compute_moneyness(
    K: np.ndarray,
    S: float,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """Compute normalised moneyness z = log(K/S) / σ.

    Parameters
    ----------
    K : np.ndarray
        Strike prices.
    S : float
        Current spot price.
    sigma : np.ndarray or float
        Market-implied standard deviation. If using variance swap (VSW)
        as in the presentation, this is √VSW(ttoexp) for each option's
        maturity.

    Returns
    -------
    z : np.ndarray
        Normalised moneyness values.
    """
    return np.log(K / S) / sigma


def compute_psychological_time(
    ttoexp: np.ndarray,
    psi: float = 25.0,
) -> np.ndarray:
    """Compute psychological time τ = ψ log(1 + T/ψ).

    This log-transform makes the maturity axis perceptually uniform:
    short-dated options are spread out (where the surface is "flexible")
    and long-dated options are compressed (where it's "stiff").

    Parameters
    ----------
    ttoexp : np.ndarray
        Time to expiry in business days.
    psi : float
        Scale parameter (business days). Default 25 ≈ 1 month.

    Returns
    -------
    tau : np.ndarray
        Psychological time values.
    """
    return psi * np.log(1 + ttoexp / psi)


def inverse_psychological_time(
    tau: np.ndarray,
    psi: float = 25.0,
) -> np.ndarray:
    """Invert the τ-transform to recover ttoexp in business days.

    Parameters
    ----------
    tau : np.ndarray
        Psychological time values.
    psi : float
        Scale parameter.

    Returns
    -------
    ttoexp : np.ndarray
        Time to expiry in business days.
    """
    return psi * (np.exp(tau / psi) - 1)


def build_grid(config: ReparamConfig | None = None) -> dict:
    """Build the discretised (z, τ) grid for binning option data.

    Returns
    -------
    grid : dict with keys:
        'z_edges'   : np.ndarray — bin edges for moneyness
        'z_centers' : np.ndarray — bin centers for moneyness
        'tau_edges' : np.ndarray — bin edges for psychological time
        'tau_centers' : np.ndarray — bin centers for psychological time
    """
    if config is None:
        config = ReparamConfig()

    z_edges = np.linspace(config.z_min, config.z_max, config.n_z_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # τ range: from ~3 business days to ~250 business days
    tau_min = compute_psychological_time(np.array([3.0]), config.psi)[0]
    tau_max = compute_psychological_time(np.array([250.0]), config.psi)[0]
    tau_edges = np.linspace(tau_min, tau_max, config.n_tau_bins + 1)
    tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])

    return {
        "z_edges": z_edges,
        "z_centers": z_centers,
        "tau_edges": tau_edges,
        "tau_centers": tau_centers,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Data loading (placeholder for when you get real data)
# ---------------------------------------------------------------------------

def load_options_data(filepath: str) -> pd.DataFrame:
    """Load raw options data from a CSV file.

    Expected columns:
        date, spot, strike, ttoexp, mid_price, option_type, implied_vol

    This is a placeholder — adapt it to whatever data format you end
    up using (CBOE, OptionMetrics, Bloomberg, etc.).

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        Raw options data.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])

    required_cols = ["date", "spot", "strike", "ttoexp", "mid_price", "option_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def preprocess_options_data(
    df: pd.DataFrame,
    config: ReparamConfig | None = None,
) -> pd.DataFrame:
    """Add reparameterized coordinates to options data.

    Adds columns:
        z   — normalised moneyness
        tau — psychological time

    Parameters
    ----------
    df : pd.DataFrame
        Raw options data with columns: spot, strike, ttoexp, implied_vol.
    config : ReparamConfig or None
        Reparameterization settings.

    Returns
    -------
    df : pd.DataFrame
        Data with z and tau columns added.
    """
    if config is None:
        config = ReparamConfig()

    df = df.copy()

    # Use implied vol as the normalising σ for moneyness
    # (In production you'd use VSW, but implied vol is a reasonable proxy)
    df["z"] = compute_moneyness(
        df["strike"].values,
        df["spot"].values,
        df["implied_vol"].values,
    )

    df["tau"] = compute_psychological_time(
        df["ttoexp"].values,
        psi=config.psi,
    )

    # Filter to the relevant domain
    mask = (df["z"] >= config.z_min) & (df["z"] <= config.z_max)
    df = df[mask].reset_index(drop=True)

    return df
