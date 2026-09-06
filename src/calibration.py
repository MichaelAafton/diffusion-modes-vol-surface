"""
Field-Theory Calibration
=========================

Calibrates the two effective stiffness parameters ``(κ, μ)`` of the 2D field
theory to the **empirical** correlation structure of real option returns, and
tests how well the heat-equation form matches the data.

The 2D stochastic PDE on the ``(z, τ)`` surface,

    ∂ₜp = κ ∂²_z p + μ ∂²_τ p + ξ(z,τ,t),

makes correlation depend on "distance" in ``(z, τ)`` space:
    κ — effective diffusion/stiffness along moneyness z
    μ — effective diffusion/stiffness along psychological time τ

Procedure (Phase 5)
-------------------
1. Compute the empirical correlation matrix from real data.
2. Write the model correlation matrix as a function of ``(κ, μ)``.
3. Minimise ‖C_empirical − C_model(κ, μ)‖²_F via ``scipy.optimize``; report fit
   quality and parameter confidence intervals.

On "universality" — frame it as a question, not a result
---------------------------------------------------------
The original presentation suggests the fitted parameters are roughly consistent
across assets. This is the part **most likely not to replicate** on real data:
equity-index skew, single-stock skew, and crypto-option skew genuinely differ in
shape. Treat cross-asset consistency as an *open hypothesis to test*, and report
the differences honestly. "Universality holds for the leading modes but the fitted
stiffness differs across asset classes" is a more credible — and more interesting —
finding than a forced claim of universality.

Status: scaffolding. Implement once Phases 1–4 are complete.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize  # noqa: F401  (used once fitting is implemented)
from src.simulate import StochasticHeatEquation, SPDEConfig


@dataclass
class FieldTheoryParams:
    """Effective stiffness parameters of the 2D field theory.

    Attributes
    ----------
    kappa : float
        Stiffness along moneyness ``z``.
    mu : float
        Stiffness along psychological time ``τ``.
    """
    kappa: float = 0.1
    mu: float = 0.7


def model_correlation_2d(
    z_grid: np.ndarray,
    tau_grid: np.ndarray,
    params: FieldTheoryParams,
    n_terms_z: int = 20,
    n_terms_tau: int = 10,
) -> np.ndarray:
    """Model correlation matrix for the 2D field theory over the flattened grid.

    Sketch of the approach:
      1. Build the 2D Fourier basis φ_{k,l}(z,τ) = cos(kπz/L_z)·cos(lπτ/L_τ).
      2. Each mode has variance ~ 1 / (κ k² + μ l²).
      3. C(i,j) = Σ_{k,l} w_{k,l} φ_{k,l}(zᵢ,τᵢ) φ_{k,l}(zⱼ,τⱼ), normalised to unit diagonal.

    TODO (Phase 5): implement after the 1D analysis (Phases 1–4) is complete.
    """
    raise NotImplementedError(
        "2D model correlation — implement in Phase 5 after the 1D analysis."
    )


def fit_field_theory(
    empirical_corr: np.ndarray,
    z_grid: np.ndarray,
    tau_grid: np.ndarray,
    initial_params: FieldTheoryParams | None = None,
) -> dict:
    """Fit ``(κ, μ)`` to the empirical 2D correlation matrix.

    Minimises ‖C_empirical − C_model(κ, μ)‖²_F over ``(κ, μ)``.

    Returns
    -------
    result : dict with keys
        'params'          : FieldTheoryParams — best-fit ``(κ, μ)``
        'confidence'      : dict — confidence intervals for each parameter
        'model_corr'      : np.ndarray — model correlation at best fit
        'relative_error'  : float — relative Frobenius error

    TODO (Phase 5): implement the fit and bootstrap/Hessian confidence intervals,
    then run it per asset class and *report cross-asset differences honestly*.
    """
    raise NotImplementedError("Field-theory fitting — implement in Phase 5.")

def model_spectrum(
    D,
    kappa,
    mean_inc_var,
    n_z=8,
    z_min=-1.15,
    z_max=1.10,
    dt=1.0,
    return_corr=False,
):
    """Exact variance shares of daily changes under the composite model.

    Covariance of increments, built from the OU increment law

        Var(dF_k) = 2 V_k (1 - exp(-gamma_k dt))

    on the sampled cosine basis, plus a uniform level-factor
    increment variance.

    By default, returns the correlation-normalised eigenvalue
    shares, matching ``run_pca``.

    If ``return_corr=True``, returns the 8x8 correlation matrix
    itself instead.
    """
    cfg = SPDEConfig(
        D=D,
        kappa=kappa,
        n_z=n_z,
        n_modes=n_z,
        z_min=z_min,
        z_max=z_max,
    )

    spde = StochasticHeatEquation(cfg)

    w = 2.0 * spde.stationary_variance * (
        1.0 - np.exp(-spde.decay_rates * dt)
    )  # per-mode increment variance

    Phi = spde.basis_functions  # (n_modes, n_z)

    cov = Phi.T @ np.diag(w) @ Phi + mean_inc_var

    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    if return_corr:
        return corr

    eig = np.linalg.eigvalsh(corr)[::-1]  # descending
    return eig / eig.sum()

def fit_composite(target, sigma, x0=(0.2, 0.02, 11.2)):
    """Fit (D, kappa, mean_inc_var) to a spectrum by weighted least squares.

    Fits log-parameters (positivity, scale-symmetric); modes 1-7 only
    (mode 8 is determined by the sum-to-one constraint). Returns dict
    with best-fit params and chi2 (dof = 7 - 3 = 4).
    """
    def chi2(logp):
        D, kappa, m = np.exp(logp)
        model = model_spectrum(D, kappa, m)
        return np.sum(((model[:7] - target[:7]) / sigma[:7]) ** 2)

    res = minimize(chi2, np.log(x0), method="Nelder-Mead")
    D, kappa, m = np.exp(res.x)
    return {"D": D, "kappa": kappa, "mean_inc_var": m,
            "chi2": res.fun, "success": res.success}