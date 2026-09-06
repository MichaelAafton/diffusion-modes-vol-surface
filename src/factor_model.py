"""
PCA-Based Factor Model for Option Risk
========================================

Turns the PCA modes into a practical factor model for the risk (P&L distribution)
of option portfolios — the practical payoff of the project.

Instead of modelling each option independently, decompose the portfolio's exposure
into a small number ``N`` of PCA factors (``N ≈ 5–7`` from the spectrum):

    PnL_portfolio ≈ Σᵢ wᵢ · Fᵢ

where ``wᵢ`` are the portfolio's factor loadings and ``Fᵢ`` are the factor returns
(time series of the i-th principal component).

The validation that matters (Phase 6) is **real and out-of-sample**:
  - split real data into in-sample (PCA + calibration) and out-of-sample;
  - construct test portfolios (butterfly, risk reversal, straddle);
  - predict each P&L distribution and compare predicted vs realised variance and
    correlations out-of-sample;
  - show whether the factor model beats a simple **SVI-surface baseline**.

Status: scaffolding. Implement once Phases 1–5 are complete.
"""

import numpy as np
from dataclasses import dataclass

from src.pca import PCAResult


@dataclass
class FactorModelResult:
    """Results from the factor model.

    Attributes
    ----------
    predicted_pnl_std : float
        Predicted standard deviation of portfolio P&L.
    actual_pnl_std : float
        Actual (realised) standard deviation.
    explained_ratio : float
        Fraction of portfolio variance explained by the factor model.
    factor_exposures : np.ndarray
        Portfolio's loading on each factor.
    residual_std : float
        Standard deviation of the unexplained (residual) P&L.
    """
    predicted_pnl_std: float
    actual_pnl_std: float
    explained_ratio: float
    factor_exposures: np.ndarray
    residual_std: float


class FactorModel:
    """PCA-based factor model for option portfolio risk.

    Example
    -------
    >>> pca_result = run_pca(returns_train, n_components=7)
    >>> model = FactorModel(pca_result)
    >>> model.fit(returns_train)
    >>> result = model.evaluate_portfolio(portfolio_weights, returns_test)
    """

    def __init__(self, pca_result: PCAResult, n_factors: int = 7):
        """Initialise the factor model.

        Parameters
        ----------
        pca_result : PCAResult
            PCA results from the training period.
        n_factors : int
            Number of factors to use (default 7, based on the
            presentation's eigenvalue spectrum).
        """
        self.n_factors = min(n_factors, len(pca_result.eigenvalues))
        self.eigenvectors = pca_result.eigenvectors[:self.n_factors]
        self.eigenvalues = pca_result.eigenvalues[:self.n_factors]
        self._factor_std = None

    def fit(self, returns_train: np.ndarray) -> None:
        """Fit the factor model on training data.

        Computes the factor return time series and their statistics.

        Parameters
        ----------
        returns_train : np.ndarray, shape (n_days, n_features)
            Training period returns.
        """
        # TODO: Implement in Phase 6
        # Steps:
        # 1. Project returns onto eigenvectors to get factor returns
        # 2. Compute factor return statistics (std, correlations)
        # 3. Store for later use in risk estimation
        raise NotImplementedError("Implement in Phase 6.")

    def portfolio_factor_exposure(
        self,
        portfolio_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute portfolio's exposure to each factor.

        Parameters
        ----------
        portfolio_weights : np.ndarray, shape (n_features,)
            Portfolio weight on each option/grid point.

        Returns
        -------
        exposures : np.ndarray, shape (n_factors,)
            Loading on each factor.
        """
        # TODO: Implement in Phase 6
        # exposure_k = portfolio_weights · eigenvector_k
        raise NotImplementedError("Implement in Phase 6.")

    def predict_portfolio_risk(
        self,
        portfolio_weights: np.ndarray,
    ) -> float:
        """Predict the standard deviation of portfolio P&L.

        Parameters
        ----------
        portfolio_weights : np.ndarray, shape (n_features,)
            Portfolio weights.

        Returns
        -------
        predicted_std : float
            Predicted P&L standard deviation.
        """
        # TODO: Implement in Phase 6
        # σ²_portfolio ≈ Σᵢ (wᵢ)² σ²_factor_i
        raise NotImplementedError("Implement in Phase 6.")

    def evaluate_portfolio(
        self,
        portfolio_weights: np.ndarray,
        returns_test: np.ndarray,
    ) -> FactorModelResult:
        """Evaluate the factor model out-of-sample.

        Parameters
        ----------
        portfolio_weights : np.ndarray, shape (n_features,)
            Portfolio weights.
        returns_test : np.ndarray, shape (n_days, n_features)
            Out-of-sample returns.

        Returns
        -------
        result : FactorModelResult
            Comparison of predicted vs actual risk.
        """
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6.")


# ---------------------------------------------------------------------------
# Example portfolios for testing
# ---------------------------------------------------------------------------

def make_butterfly(z_grid: np.ndarray, center: float = 0.0, width: float = 0.5) -> np.ndarray:
    """Create a butterfly spread portfolio in z-space.

    A butterfly is long the wings and short the body:
    long at (center - width), short 2x at center, long at (center + width).

    Parameters
    ----------
    z_grid : np.ndarray
        Moneyness grid.
    center : float
        Center strike in z-space.
    width : float
        Width of the butterfly in z-units.

    Returns
    -------
    weights : np.ndarray, shape (n_z,)
        Portfolio weights on the z-grid.
    """
    weights = np.zeros_like(z_grid)
    idx_low = np.argmin(np.abs(z_grid - (center - width)))
    idx_mid = np.argmin(np.abs(z_grid - center))
    idx_high = np.argmin(np.abs(z_grid - (center + width)))

    weights[idx_low] = 1.0
    weights[idx_mid] = -2.0
    weights[idx_high] = 1.0

    return weights


def make_risk_reversal(z_grid: np.ndarray, put_z: float = -1.0, call_z: float = 1.0) -> np.ndarray:
    """Create a risk reversal portfolio in z-space.

    Short a put, long a call — exposure to skew.

    Parameters
    ----------
    z_grid : np.ndarray
        Moneyness grid.
    put_z : float
        Moneyness of the short put.
    call_z : float
        Moneyness of the long call.

    Returns
    -------
    weights : np.ndarray, shape (n_z,)
    """
    weights = np.zeros_like(z_grid)
    idx_put = np.argmin(np.abs(z_grid - put_z))
    idx_call = np.argmin(np.abs(z_grid - call_z))

    weights[idx_put] = -1.0
    weights[idx_call] = 1.0

    return weights


def make_straddle(z_grid: np.ndarray, center: float = 0.0) -> np.ndarray:
    """Create a straddle portfolio in z-space.

    Long the at-the-money strike (a pure vol/convexity bet).

    Parameters
    ----------
    z_grid : np.ndarray
        Moneyness grid.
    center : float
        ATM strike in z-space.

    Returns
    -------
    weights : np.ndarray, shape (n_z,)
    """
    weights = np.zeros_like(z_grid)
    weights[np.argmin(np.abs(z_grid - center))] = 1.0
    return weights


# ---------------------------------------------------------------------------
# Baseline for comparison
# ---------------------------------------------------------------------------

def svi_baseline_risk(*args, **kwargs) -> float:
    """Predict portfolio P&L risk from a simple SVI-surface baseline.

    The honest test for the factor model is whether it beats a standard
    parametric surface fit (SVI) out-of-sample — not whether it works at all.

    TODO (Phase 6): fit SVI per maturity slice, propagate to portfolio P&L risk,
    and compare against ``FactorModel.evaluate_portfolio`` on the same OOS window.
    """
    raise NotImplementedError(
        "SVI baseline — implement in Phase 6 as the comparison benchmark."
    )

"""Factor risk models for the z-bin dSigma panel, + out-of-sample test."""


def train_test_split(changes, train_frac=0.7):
    n = int(len(changes) * train_frac)
    return changes[:n], changes[n:]


def empirical_factor_cov(train, n_factors=3):
    """PCA-truncated covariance + diagonal residual top-up."""
    c = np.cov(train, rowvar=False)
    lam, V = np.linalg.eigh(c)
    lam, V = lam[::-1], V[:, ::-1]                     # descending
    recon = (V[:, :n_factors] * lam[:n_factors]) @ V[:, :n_factors].T
    resid = np.diag(np.diag(c) - np.diag(recon))       # keep per-bin totals
    return recon + resid


def field_theory_cov(train, corr_model):
    """Physics correlations, train-data units."""
    s = train.std(axis=0, ddof=1)
    return np.outer(s, s) * corr_model


def diagonal_cov(train):
    return np.diag(train.var(axis=0, ddof=1))


def portfolio_report(covs: dict, test, portfolios: dict):
    """Realised OOS variance vs each model's prediction, per portfolio."""
    rows = []
    for pname, w in portfolios.items():
        w = np.asarray(w, float)
        realised = np.var(test @ w, ddof=1)
        row = {"portfolio": pname, "realised": realised}
        for mname, S in covs.items():
            row[mname] = realised / (w @ S @ w)        # ratio: 1 = perfect
        rows.append(row)
    return rows
