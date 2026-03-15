"""
Field Theory Calibration
=========================

Fits the parameters μ and κ of the 2D field theory model to the
empirical correlation structure of option returns (Slide 11).

The 2D stochastic PDE on the (z, τ) surface:

    ∂ₜp = D_z ∂²_z p + D_τ ∂²_τ p + ξ(z,τ,t)

can be parameterised by:
    μ — controls the effective diffusion along moneyness z
    κ — controls the effective diffusion along psychological time τ

From the presentation:
    - μ ~ 0.5-0.9 for most asset classes
    - κ stays low and flat (~0.05-0.2)
    - Fit error ~5%
    - VIX options and KOSPI are outliers

These two parameters encode the overall "stiffness" of the volatility
surface membrane, and are remarkably consistent across equities,
commodities, and FX — a strong universality claim.

TODO: Implement this in Phase 5 of the project. The structure below
is scaffolding to be filled in once you've completed Phases 1-4.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class FieldTheoryParams:
    """Parameters of the 2D field theory model.

    Attributes
    ----------
    mu : float
        Diffusion parameter along moneyness z.
    kappa : float
        Diffusion parameter along psychological time τ.
    """
    mu: float = 0.7
    kappa: float = 0.1


def model_correlation_2d(
    z_grid: np.ndarray,
    tau_grid: np.ndarray,
    params: FieldTheoryParams,
    n_terms_z: int = 20,
    n_terms_tau: int = 10,
) -> np.ndarray:
    """Compute the model correlation matrix for the 2D field theory.

    The correlation between points (z₁,τ₁) and (z₂,τ₂) is predicted
    by the 2D stochastic heat equation with parameters μ and κ.

    Parameters
    ----------
    z_grid : np.ndarray, shape (n_z,)
        Moneyness grid.
    tau_grid : np.ndarray, shape (n_tau,)
        Psychological time grid.
    params : FieldTheoryParams
        Model parameters.
    n_terms_z : int
        Number of Fourier terms in z-direction.
    n_terms_tau : int
        Number of Fourier terms in τ-direction.

    Returns
    -------
    C : np.ndarray, shape (n_z * n_tau, n_z * n_tau)
        Model correlation matrix over the flattened 2D grid.
    """
    # TODO: Implement in Phase 5
    # Sketch of the approach:
    # 1. Build the 2D Fourier basis: φ_{k,l}(z,τ) = cos(kπz/Lz) cos(lπτ/Lτ)
    # 2. Each mode has variance ~ 1/(μ k² + κ l²)
    # 3. Compute C(i,j) = Σ_{k,l} w_{k,l} φ_{k,l}(zᵢ,τᵢ) φ_{k,l}(zⱼ,τⱼ)
    # 4. Normalise to unit diagonal
    raise NotImplementedError(
        "2D model correlation — implement in Phase 5 after completing "
        "the 1D analysis in Phases 1-4."
    )


def fit_field_theory(
    empirical_corr: np.ndarray,
    z_grid: np.ndarray,
    tau_grid: np.ndarray,
    initial_params: FieldTheoryParams | None = None,
) -> dict:
    """Fit μ and κ to the empirical 2D correlation matrix.

    Minimises ||C_empirical - C_model(μ, κ)||² over (μ, κ).

    Parameters
    ----------
    empirical_corr : np.ndarray, shape (n_z * n_tau, n_z * n_tau)
        Empirical correlation matrix.
    z_grid : np.ndarray
        Moneyness grid.
    tau_grid : np.ndarray
        Psychological time grid.
    initial_params : FieldTheoryParams or None
        Starting point for optimisation.

    Returns
    -------
    result : dict with keys:
        'params'         : FieldTheoryParams — best-fit parameters
        'model_corr'     : np.ndarray — model correlation at best fit
        'relative_error' : float — relative Frobenius error
    """
    # TODO: Implement in Phase 5
    raise NotImplementedError(
        "Field theory fitting — implement in Phase 5."
    )
