"""
Data pipeline: OptionMetrics vol surfaces -> the study panel
============================================================

Loads SPX implied-volatility surfaces (IvyDB ``vsurfd``), maps each vendor pillar
to standardised moneyness

    z = log(K / S) / (sigma_ATM * sqrt(T)),

and builds the study panel: the 30-calendar-day slice, daily implied vol averaged
within 8 uniform z-bins on [-1.15, 1.10]. ``build_study_panel`` is the single
place this panel is built; every real-data script calls it.

The vendor surface is kernel-smoothed (calls and puts separately), so the panel
is a smoothed measurement of the smile, not raw quotes. See README.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Implied vol is annualised; maturities are handled in business days.
BDAYS_PER_YEAR = 252.0

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class ReparamConfig:
    """Moneyness window and binning of the study panel.

    Parameters
    ----------
    z_min, z_max : float
        Moneyness window in units of sigma_ATM * sqrt(T).
    n_z_bins : int
        Number of uniform bins on the window.
    """
    z_min: float = -1.15
    z_max: float = 1.10
    n_z_bins: int = 8


# ---------------------------------------------------------------------------
# Moneyness
# ---------------------------------------------------------------------------

def compute_moneyness(
    K: np.ndarray,
    S: float,
    sigma: np.ndarray | float,
    T: np.ndarray | float,
) -> np.ndarray:
    """Moneyness in standard deviations to expiry, ``z = log(K/S) / (sigma sqrt(T))``.

    Parameters
    ----------
    K : strike(s)
    S : spot
    sigma : annualised volatility used for scaling (the study uses sigma_ATM)
    T : time to expiry in years
    """
    return np.log(K / S) / (sigma * np.sqrt(T))


def preprocess_options_data(
    df: pd.DataFrame,
    config: ReparamConfig | None = None,
) -> pd.DataFrame:
    """Add standardised moneyness ``z`` to an options DataFrame.

    Expects columns: ``spot``, ``strike``, ``ttoexp`` (business days),
    ``implied_vol`` and ``sigma_atm`` (annualised). Adds ``T_years``
    (``ttoexp / 252``) and ``z``. Rows outside ``[z_min, z_max]`` are dropped.
    """
    if config is None:
        config = ReparamConfig()

    df = df.copy()
    df["T_years"] = df["ttoexp"].values / BDAYS_PER_YEAR

    if "sigma_atm" not in df.columns:
        raise KeyError("sigma_atm missing: load data with load_optionmetrics")

    df["z"] = compute_moneyness(
        df["strike"].values,
        df["spot"].values,
        df["sigma_atm"].values,
        df["T_years"].values,
    )

    mask = (df["z"] >= config.z_min) & (df["z"] <= config.z_max)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_optionmetrics(
    surface_path: str | Path = RAW_DIR / "spx_vsurface.parquet",
    spot_path: str | Path = RAW_DIR / "spx_spot.parquet",
    otm_only: bool = True,
) -> pd.DataFrame:
    """Load SPX smoothed vol surfaces (IvyDB vsurfd) into pipeline schema.

    Files are written by ``scripts/pull_wrds.py``. Choices (report, Sec. 3):
    the vendor's impl_strike is the strike; only out-of-the-money pillars are
    kept (|delta| <= 50), so puts cover z < 0 and calls z > 0; maturities are
    calendar days, so T = days / 365. Numeric columns are coerced on load.
    """
    surf = pd.read_parquet(surface_path)
    spot = pd.read_parquet(spot_path)

    for d in (surf, spot):
        d["date"] = pd.to_datetime(d["date"])

    df = surf.merge(spot, on="date", how="inner")

    df = df.rename(columns={
        "impl_strike": "strike",
        "impl_volatility": "implied_vol",
        "impl_premium": "mid_price",
        "cp_flag": "option_type",
    })

    num_cols = ["spot", "strike", "implied_vol", "mid_price", "delta", "days"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df["ttoexp"] = df["days"] * (252 / 365)
    atm = (
        df[np.isclose(df["delta"].abs(), 50)]
        .groupby(["date", "days"])["implied_vol"]
        .mean()
        .rename("sigma_atm")
        .reset_index()
    )
    df = df.merge(atm, on=["date", "days"], how="inner")

    if otm_only:
        df = df[df["delta"].abs() <= 50]

    df = df.dropna(subset=["strike", "implied_vol", "spot"])

    return df[
        ["days", "date", "spot", "strike", "ttoexp",
         "mid_price", "option_type",
         "implied_vol", "sigma_atm", "delta"]
    ].reset_index(drop=True)


def build_surface_panel(
    df: pd.DataFrame,
    config: ReparamConfig,
    min_coverage: float = 0.95,
) -> dict:
    """Day x z-bin panel of implied vol at a fixed maturity slice.

    Bins each option's z into config's grid, averages implied_vol
    per (date, bin), pivots to a (n_days, n_bins) matrix. Bins with
    < min_coverage fill across days are dropped (no interpolation:
    invented data would leak smoothness into the spectrum), then
    remaining incomplete days are dropped.

    The input must contain a single maturity; averaging across maturities
    within a z-bin is refused. Real-data callers should use
    ``build_study_panel``.
    """
    if "days" in df.columns and df["days"].nunique() > 1:
        raise ValueError(
            f"build_surface_panel got {df['days'].nunique()} maturities "
            f"{sorted(df['days'].unique().tolist())}; filter to one maturity "
            "first (use build_study_panel for the study slice)."
        )

    edges = np.linspace(config.z_min, config.z_max, config.n_z_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    df = df.copy()
    df["z_bin"] = pd.cut(df["z"], edges, labels=False)

    panel = (df.groupby(["date", "z_bin"])["implied_vol"]
               .mean().unstack("z_bin").astype(float))

    coverage = panel.notna().mean(axis=0)            # per-bin fill rate
    keep = coverage >= min_coverage
    kept_bins = keep.index[keep].astype(int).to_numpy()
    panel = panel.loc[:, keep].dropna(axis=0)

    return {"panel": panel,                    # DataFrame: days x bins
            "z_centers": centers[kept_bins],
            "kept_bins": kept_bins,
            "n_bins_dropped": config.n_z_bins - len(kept_bins),
            "coverage": coverage}


# ---------------------------------------------------------------------------
# The study panel: the single source of truth for every real-data script
# ---------------------------------------------------------------------------

STUDY_MATURITY_DAYS = 30


def build_study_panel(
    df_raw: pd.DataFrame | None = None,
    config: ReparamConfig | None = None,
    min_coverage: float = 0.95,
) -> dict:
    """Build the study panel: SPX 30-calendar-day slice, daily implied vol per z-bin.

    Specification (fixed; every real-data script uses this function):
      - maturity: days == 30 (single slice, asserted)
      - moneyness: z = log(K/S) / (sigma_ATM * sqrt(T)), T = days / 365
      - grid: 8 uniform bins on [-1.15, 1.10]  (ReparamConfig defaults)
      - coverage: bins with < 95% daily fill are dropped, then incomplete days

    Parameters
    ----------
    df_raw : DataFrame or None
        Output of ``load_optionmetrics``; loaded from the default paths if None.
    config : ReparamConfig or None
        Defaults to ``ReparamConfig()`` (the study grid).

    Returns
    -------
    dict with keys ``panel`` (DataFrame, days x kept bins), ``z_centers``,
    ``kept_bins``, ``n_bins_dropped``, ``coverage``, ``maturities``, ``config``.
    """
    if config is None:
        config = ReparamConfig()
    if df_raw is None:
        df_raw = load_optionmetrics()

    df = df_raw[df_raw["days"] == STUDY_MATURITY_DAYS]
    df = preprocess_options_data(df, config)

    maturities = sorted(df["days"].unique().tolist())
    assert set(maturities) == {STUDY_MATURITY_DAYS}, (
        f"study panel must be the {STUDY_MATURITY_DAYS}-day slice, got {maturities}"
    )

    out = build_surface_panel(df, config, min_coverage=min_coverage)
    out["maturities"] = maturities
    out["config"] = config
    return out


STUDY_SPECTRUM_PATH = PROCESSED_DIR / "study_spectrum.npz"


def load_study_spectrum(path: str | Path | None = None) -> dict:
    """Load the reference spectrum written by ``scripts/build_panel.py``.

    Keys: ``shares``, ``z_centers``, ``kept_bins``, ``panel_shape``, ``p_tail``.
    """
    npz = np.load(STUDY_SPECTRUM_PATH if path is None else path)
    return {k: npz[k] for k in npz.files}
