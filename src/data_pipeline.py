"""
Data Pipeline & Reparameterization
====================================

Loads option data — **real and synthetic** — and implements the
reparameterization from raw ``(K, ttoexp)`` coordinates into the normalised
``(z, τ)`` space where the heat-equation structure (if any) becomes visible.

Design note (the whole point of the project)
---------------------------------------------
Synthetic data is the **validation harness**: it proves this pipeline recovers
known ground truth. Real options data is the **actual study**: the research
question is where real surfaces match the diffusive-membrane picture and where
they deviate. Both flow through the *same* code below — the comparison is the
contribution.

The reparameterization
----------------------
Moneyness (standard deviations to expiry):

    z = log(K / S) / (σ · √T)

The ``√T`` is **not optional**. Implied vol ``σ`` is annualized, so to express
moneyness in standard deviations *to expiry* you must scale by ``√T`` (equivalently,
use total implied variance over the option's life). Without it, ``z`` is not
comparable across maturities and the entire ``τ``-dimension analysis is distorted.
State this convention explicitly in any writeup.

Psychological time (log-compressed maturity):

    τ = ψ · log(1 + T/ψ)     with ψ ≈ 20–30 business days

This compresses long horizons relative to short ones, matching the intuition that
1m-vs-2m feels larger than 1y-vs-2y.

Hedging convention
------------------
Delta-hedged P&L requires a chosen **hedging frequency**. "Continuously hedged" is
a fiction; the frequency you pick injects both noise and bias into every number
downstream. It is a modelling decision to document, captured here in
``ReparamConfig.hedge_frequency`` rather than hidden.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Time-scaling constant. Implied vol is annualized; maturities here are handled in
# business days, so convert to years with this factor for the √T moneyness scaling.
BDAYS_PER_YEAR = 252.0


@dataclass
class ReparamConfig:
    """Configuration for the ``(z, τ)`` reparameterization.

    Parameters
    ----------
    psi : float
        Psychological-time scale (business days). Controls the log-compression of
        the maturity axis; ψ ≈ 20–30 works well.
    z_min, z_max : float
        Moneyness domain (e.g. −3 → ~3σ crash, +3 → ~3σ rally).
    n_z_bins, n_tau_bins : int
        Grid resolution along each axis.
    hedge_frequency : str
        Delta-hedging frequency assumed when forming delta-hedged P&L
        ('daily', 'intraday', ...). A documented modelling choice, not a detail.
    """
    psi: float = 25.0
    z_min: float = -3.0
    z_max: float = 3.0
    n_z_bins: int = 30
    n_tau_bins: int = 10
    hedge_frequency: str = "daily"


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------

def compute_moneyness(
    K: np.ndarray,
    S: float,
    sigma: np.ndarray | float,
    T: np.ndarray | float,
) -> np.ndarray:
    """Standard-deviation-to-expiry moneyness ``z = log(K/S) / (σ·√T)``.

    Parameters
    ----------
    K : np.ndarray
        Strike prices.
    S : float
        Current spot price.
    sigma : np.ndarray or float
        Annualized implied volatility (or √VSW) for each option.
    T : np.ndarray or float
        Time to expiry **in years** (annualized, to match ``sigma``).

    Returns
    -------
    z : np.ndarray
        Normalised moneyness. ``z = −3`` ≈ a 3σ crash strike, ``z = +2`` ≈ a 2σ
        rally strike — independent of spot level or volatility regime.

    Notes
    -----
    The ``√T`` factor is what makes ``z`` comparable across maturities. Omitting it
    (the common ``log(K/S)/σ`` shortcut) only holds for a fixed-maturity slice or an
    implicit total-variance convention.
    """
    return np.log(K / S) / (sigma * np.sqrt(T))


def compute_psychological_time(
    ttoexp: np.ndarray,
    psi: float = 25.0,
) -> np.ndarray:
    """Psychological time ``τ = ψ·log(1 + T/ψ)``.

    Spreads out short-dated options (where the surface is flexible) and compresses
    long-dated ones (where it is stiff).

    Parameters
    ----------
    ttoexp : np.ndarray
        Time to expiry in business days.
    psi : float
        Scale parameter (business days). Default 25 ≈ 1 month.
    """
    return psi * np.log(1 + ttoexp / psi)


def inverse_psychological_time(
    tau: np.ndarray,
    psi: float = 25.0,
) -> np.ndarray:
    """Invert the ``τ``-transform to recover time to expiry in business days."""
    return psi * (np.exp(tau / psi) - 1)


def build_grid(config: ReparamConfig | None = None) -> dict:
    """Build the discretised ``(z, τ)`` grid for binning option data.

    Returns
    -------
    grid : dict with keys ``z_edges``, ``z_centers``, ``tau_edges``,
        ``tau_centers``, ``config``.
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


def preprocess_options_data(
    df: pd.DataFrame,
    config: ReparamConfig | None = None,
) -> pd.DataFrame:
    """Add reparameterized ``(z, τ)`` coordinates to an options DataFrame.

    Expects columns: ``spot``, ``strike``, ``ttoexp`` (business days),
    ``implied_vol`` (annualized). Adds:

    - ``T_years`` — time to expiry in years (``ttoexp / 252``)
    - ``z``       — standard-deviation moneyness (with the √T scaling)
    - ``tau``     — psychological time

    Rows outside ``[z_min, z_max]`` are dropped.
    """
    if config is None:
        config = ReparamConfig()

    df = df.copy()
    df["T_years"] = df["ttoexp"].values / BDAYS_PER_YEAR

    # Annualized implied vol as the normalising σ (use √VSW if available).
    df["z"] = compute_moneyness(
        df["strike"].values,
        df["spot"].values,
        df["implied_vol"].values,
        df["T_years"].values,
    )
    df["tau"] = compute_psychological_time(df["ttoexp"].values, psi=config.psi)

    mask = (df["z"] >= config.z_min) & (df["z"] <= config.z_max)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data loading — synthetic (harness) and real (study)
# ---------------------------------------------------------------------------

def load_synthetic_data(filepath: str = "data/synthetic/spde_dataset.npz") -> dict:
    """Load a simulated dataset produced by the validation harness.

    Returns the same dict shape as ``simulate.generate_synthetic_dataset`` so the
    downstream pipeline is identical for synthetic and real inputs.
    """
    npz = np.load(filepath)
    return {
        "z_grid": npz["z_grid"],
        "surface": npz["surface"],
        "returns": npz["returns"],
    }


def load_options_data(filepath: str) -> pd.DataFrame:
    """Load raw real options data from CSV/Parquet.

    Expected columns:
        date, spot, strike, ttoexp, mid_price, option_type, implied_vol

    Adapt to whatever real source you commit to (OptionMetrics/IvyDB, Deribit, ...).
    """
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, parse_dates=["date"])

    required = ["date", "spot", "strike", "ttoexp", "mid_price", "option_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_optionmetrics(*args, **kwargs) -> pd.DataFrame:
    """Load equity-option surfaces from WRDS / OptionMetrics (IvyDB).

    The cleanest path to real equity surfaces if your institution subscribes.
    Map the IvyDB schema onto the columns expected by ``load_options_data``.

    TODO (Phase 1): implement once a data source is secured.
    """
    raise NotImplementedError(
        "OptionMetrics/IvyDB loader — implement in Phase 1 once WRDS access is set up."
    )


def load_deribit(*args, **kwargs) -> pd.DataFrame:
    """Load BTC/ETH option history from Deribit's public API.

    Free, several years of liquid crypto-option history; bridges to a later
    crypto-perps project. Surfaces are younger/weirder than equity index — which
    makes the 'where does the membrane picture break' question more interesting.

    TODO (Phase 1): implement the API pull + schema mapping.
    """
    raise NotImplementedError(
        "Deribit loader — implement in Phase 1 (public API → standard schema)."
    )


def load_yfinance_snapshot(*args, **kwargs) -> pd.DataFrame:
    """Pull a current option chain via yfinance for a sanity check ONLY.

    yfinance gives current chains, not deep history, so it cannot drive the
    time-series PCA at the core of this project. Use it to validate plumbing or to
    accumulate data going forward — do not build the study on it.

    TODO (Phase 1): implement the snapshot pull.
    """
    raise NotImplementedError(
        "yfinance snapshot — implement as a sanity check; not a basis for the study."
    )


def compute_delta_hedged_pnl_series(
    df: pd.DataFrame,
    config: ReparamConfig | None = None,
) -> pd.DataFrame:
    """Form the delta-hedged P&L time series on the ``(z, τ)`` grid.

    This is the quantity whose cross-grid correlations PCA decomposes. The result
    depends on ``config.hedge_frequency`` — a documented modelling choice (see the
    module docstring), since "continuous" hedging is a fiction.

    TODO (Phase 1/2): bin options onto the grid, difference prices, subtract the
    delta hedge at the chosen frequency, and align across dates.
    """
    raise NotImplementedError(
        "Delta-hedged P&L series — implement in Phase 1/2; document the hedging "
        "frequency assumption explicitly."
    )
