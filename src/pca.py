"""
PCA of daily surface changes
============================

Correlation-matrix PCA of the study panel's daily changes, factor projection,
and the spectral summaries used in the paper (tail slope, sign changes).
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
    """Run PCA on a matrix of daily changes.

    Parameters
    ----------
    returns : np.ndarray, shape (n_days, n_features)
        Each row is one day's changes across the moneyness bins.
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


def project_onto_factors(
    returns: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project standardised daily changes onto PCA eigenvectors (factor returns).

    Parameters
    ----------
    returns : np.ndarray, shape (n_days, n_features)
        Daily changes.
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

def slope(evr, floor=1e-10, k_start=1):
    """Power-law slope p of variance shares: evr_k ~ k^(-p), by log-log least squares.

    ``k_start`` is the mode index of ``evr[0]``; the tail slope used in the paper is
    ``slope(evr[1:], k_start=2)`` (modes 2..n). Shares <= ``floor`` are ignored.
    """
    evr = np.asarray(evr)
    k = np.arange(k_start, k_start + len(evr))
    m = evr > floor
    return -np.polyfit(np.log(k[m]), np.log(evr[m]), 1)[0]


def sign_changes(v):
    """Number of sign changes (nodes) along an eigenvector."""
    return int(np.sum(np.diff(np.sign(v)) != 0))
