"""
PCA Analysis of Option Returns
================================

Implements Principal Component Analysis on the correlation matrix of
option returns (delta-hedged P&Ls) to extract the dominant modes of
variation of the volatility surface.

Expected shape (the heat-equation prediction — to be *tested* on real data, not
assumed):
    - Eigenmode 0: ~flat (parallel shift) — typically the bulk of the variance
    - Eigenmode 1: ~monotone (tilt/skew)
    - Eigenmode 2: ~one-node (smile)
    - Higher modes: increasingly oscillatory (more nodes)
    - Eigenvalues decay as ~k⁻² if the dynamics are diffusive

On synthetic harness data this comes out clean by construction. On real data,
report the *actual* variance shares, whether the leading modes are genuinely
stable across rolling windows, and how far the spectrum departs from k⁻² — those
differences are the data, not error.

The mathematical reason for the prediction: if the dynamics are governed by a
stochastic heat equation, PCA recovers the (sinusoidal) eigenfunctions of the
Laplacian operator.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PCAResult:
    """Container for PCA results.

    Attributes
    ----------
    eigenvalues : np.ndarray, shape (n_components,)
        Eigenvalues in descending order.
    eigenvectors : np.ndarray, shape (n_components, n_features)
        Each row is an eigenvector (eigenmode).
    explained_variance_ratio : np.ndarray, shape (n_components,)
        Fraction of total variance explained by each component.
    correlation_matrix : np.ndarray, shape (n_features, n_features)
        The empirical correlation matrix.
    n_samples : int
        Number of time samples used.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    correlation_matrix: np.ndarray
    n_samples: int


def run_pca(
    returns: np.ndarray,
    n_components: int | None = None,
    use_correlation: bool = True,
) -> PCAResult:
    """Run PCA on a matrix of option returns.

    Parameters
    ----------
    returns : np.ndarray, shape (n_days, n_features)
        Each row is one day's returns across all grid points.
        Features are typically moneyness bins (1D) or flattened
        (moneyness × maturity) bins (2D).
    n_components : int or None
        Number of components to keep. If None, keep all.
    use_correlation : bool
        If True, decompose the correlation matrix (standardised).
        If False, decompose the covariance matrix.
        We decompose the correlation matrix by default.

    Returns
    -------
    result : PCAResult
        Container with eigenvalues, eigenvectors, etc.

    Notes
    -----
    We use numpy's eigendecomposition directly (rather than sklearn's PCA)
    because we want access to the correlation matrix and because the
    presentation explicitly works with correlation, not covariance.
    """
    n_days, n_features = returns.shape

    if n_components is None:
        n_components = n_features

    # Remove mean
    returns_centered = returns - returns.mean(axis=0)

    if use_correlation:
        # Standardise each feature to unit variance
        stds = returns_centered.std(axis=0)
        # Avoid division by zero for dead grid points
        stds[stds < 1e-10] = 1.0
        returns_standardised = returns_centered / stds
        matrix = np.corrcoef(returns_standardised.T)
    else:
        matrix = np.cov(returns_centered.T)

    # Eigendecomposition (symmetric matrix → use eigh for stability)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # eigh returns in ascending order — reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

    # Keep only n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components].T  # now rows are eigenvectors

    # Explained variance ratio
    total_variance = matrix.trace()
    explained_variance_ratio = eigenvalues / total_variance

    return PCAResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=explained_variance_ratio,
        correlation_matrix=matrix,
        n_samples=n_days,
    )


def rolling_pca(
    returns: np.ndarray,
    window_size: int = 250,
    step_size: int = 50,
    n_components: int = 5,
) -> list[PCAResult]:
    """Run PCA on rolling windows to assess stability.

    On real data, the strong claim to test is whether the leading modes are
    genuinely stable over time, or only appear so in-sample.

    Parameters
    ----------
    returns : np.ndarray, shape (n_days, n_features)
        Full time series of returns.
    window_size : int
        Rolling window length in days (default 250 ≈ 1 year).
    step_size : int
        Step between windows (default 50 ≈ 2 months).
    n_components : int
        Number of components per window.

    Returns
    -------
    results : list of PCAResult
        One PCAResult per rolling window.
    """
    n_days = returns.shape[0]
    results = []

    for start in range(0, n_days - window_size + 1, step_size):
        window_returns = returns[start : start + window_size]
        result = run_pca(window_returns, n_components=n_components)
        results.append(result)

    return results


def align_eigenvector_signs(
    eigenvectors_list: list[np.ndarray],
    reference: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Align eigenvector signs across rolling windows.

    Eigenvectors are defined up to a sign flip. To compare them over
    time (the stability check), we need consistent orientation.

    Strategy: for each mode, ensure positive inner product with a
    reference (either the first window or a provided reference).

    Parameters
    ----------
    eigenvectors_list : list of np.ndarray, each shape (n_components, n_features)
        Eigenvectors from rolling PCA.
    reference : np.ndarray or None
        Reference eigenvectors. If None, use the first in the list.

    Returns
    -------
    aligned : list of np.ndarray
        Same eigenvectors with consistent signs.
    """
    if reference is None:
        reference = eigenvectors_list[0]

    aligned = []
    for evecs in eigenvectors_list:
        evecs_aligned = evecs.copy()
        for k in range(evecs.shape[0]):
            if np.dot(evecs[k], reference[k]) < 0:
                evecs_aligned[k] *= -1
        aligned.append(evecs_aligned)

    return aligned


def project_onto_factors(
    returns: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project returns onto PCA factors.

    Parameters
    ----------
    returns : np.ndarray, shape (n_days, n_features)
        Option returns matrix.
    eigenvectors : np.ndarray, shape (n_components, n_features)
        PCA eigenvectors (rows).

    Returns
    -------
    factor_returns : np.ndarray, shape (n_days, n_components)
        Time series of factor returns.
    """
    # Center returns
    returns_centered = returns - returns.mean(axis=0)
    # Standardise
    stds = returns_centered.std(axis=0)
    stds[stds < 1e-10] = 1.0
    returns_standardised = returns_centered / stds

    return returns_standardised @ eigenvectors.T

'''def slope(evr, floor=1e-10):
    evr = np.asarray(evr)
    k = np.arange(1, len(evr) + 1)
    m = evr > floor
    return -np.polyfit(np.log(k[m]), np.log(evr[m]), 1)[0]'''
def slope(evr, floor=1e-10, k_start=1):
    evr = np.asarray(evr)
    k = np.arange(k_start, k_start + len(evr))
    m = evr > floor
    return -np.polyfit(np.log(k[m]), np.log(evr[m]), 1)[0]

def sign_changes(v):
    return int(np.sum(np.diff(np.sign(v)) != 0))
