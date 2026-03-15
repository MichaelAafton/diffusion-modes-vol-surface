"""
Heat Equation: Analytical Solutions & Empirical Comparison
============================================================

Provides the theoretical predictions of the stochastic heat equation
and tools to compare them against empirical PCA results.

The stochastic heat equation on a domain z ∈ [-L/2, L/2]:

    ∂ₜp = D ∂²_z p + ξ(z,t)

has Fourier mode decomposition (Slide 9):

    p(z,t) = Σₖ fₖ(t) φₖ(z)

where:
    φₖ(z) = cos(kπz/L)  or  sin(kπz/L)    (Laplacian eigenfunctions)
    ∂ₜfₖ = -Dk² fₖ + ξₖ(t)                (OU process per mode)
    ⟨fₖ²⟩ ~ 1/(Dk²)                        (stationary variance)

This predicts:
    - PCA eigenvectors ≈ sinusoidal functions of z
    - PCA eigenvalues ∝ k⁻²
    - Correlation C(z₁,z₂) depends on |z₁-z₂| through a specific kernel

From Slide 10, the 2D Fourier modes were:
    Mode (z0,τ0): 1
    Mode (z1,τ0): sin(πz/4)
    Mode (z0,τ1): sin(π(τ - 8.6)/7.7)
"""

import numpy as np
from scipy.optimize import minimize


def fourier_eigenfunctions(
    z_grid: np.ndarray,
    n_modes: int,
    boundary: str = "neumann",
) -> np.ndarray:
    """Compute the analytical eigenfunctions of the Laplacian.

    Parameters
    ----------
    z_grid : np.ndarray, shape (n_z,)
        Spatial grid points.
    n_modes : int
        Number of modes to compute.
    boundary : str
        Boundary condition type:
        - 'neumann': ∂p/∂z = 0 at boundaries → cosine modes
        - 'dirichlet': p = 0 at boundaries → sine modes

    Returns
    -------
    modes : np.ndarray, shape (n_modes, n_z)
        Each row is one eigenfunction evaluated on the grid.
    """
    L = z_grid[-1] - z_grid[0]
    z_shifted = z_grid - z_grid[0]  # shift to [0, L]

    modes = np.zeros((n_modes, len(z_grid)))
    for k in range(n_modes):
        n = k + 1  # mode number starts at 1
        if boundary == "neumann":
            modes[k] = np.sqrt(2 / L) * np.cos(n * np.pi * z_shifted / L)
        elif boundary == "dirichlet":
            modes[k] = np.sqrt(2 / L) * np.sin(n * np.pi * z_shifted / L)
        else:
            raise ValueError(f"Unknown boundary type: {boundary}")

    return modes


def theoretical_eigenvalue_spectrum(
    n_modes: int,
    D: float = 1.0,
    L: float = 6.0,
) -> np.ndarray:
    """Theoretical eigenvalue spectrum: λₖ ∝ 1/(Dk²).

    Parameters
    ----------
    n_modes : int
        Number of eigenvalues.
    D : float
        Diffusion coefficient.
    L : float
        Domain length.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_modes,)
        Normalised eigenvalues (summing to 1).
    """
    k = np.arange(1, n_modes + 1)
    eigenvalues = 1.0 / (D * (k * np.pi / L) ** 2)
    return eigenvalues / eigenvalues.sum()


def fit_power_law(eigenvalues: np.ndarray) -> dict:
    """Fit a power law λₖ ~ A k^{-α} to the eigenvalue spectrum.

    The heat equation predicts α = 2. Deviations indicate departures
    from simple diffusion (e.g. anomalous diffusion, nonlocal effects).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Empirical eigenvalues in descending order.

    Returns
    -------
    result : dict with keys:
        'alpha'     : float — fitted power law exponent
        'A'         : float — fitted amplitude
        'r_squared' : float — goodness of fit
        'k_values'  : np.ndarray — mode numbers
    """
    n = len(eigenvalues)
    k = np.arange(1, n + 1).astype(float)

    # Fit in log-log space: log(λ) = log(A) - α log(k)
    log_k = np.log(k)
    log_lambda = np.log(np.maximum(eigenvalues, 1e-15))

    # Linear regression in log-log space
    coeffs = np.polyfit(log_k, log_lambda, 1)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])

    # R² in log-log space
    log_lambda_pred = coeffs[0] * log_k + coeffs[1]
    ss_res = np.sum((log_lambda - log_lambda_pred) ** 2)
    ss_tot = np.sum((log_lambda - log_lambda.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": alpha,
        "A": A,
        "r_squared": r_squared,
        "k_values": k,
    }


def mode_overlap(
    empirical_mode: np.ndarray,
    theoretical_mode: np.ndarray,
) -> float:
    """Compute the overlap (absolute cosine similarity) between modes.

    A value of 1.0 means perfect alignment (up to sign),
    0.0 means orthogonal.

    Parameters
    ----------
    empirical_mode : np.ndarray, shape (n_z,)
    theoretical_mode : np.ndarray, shape (n_z,)

    Returns
    -------
    overlap : float in [0, 1]
    """
    dot = np.dot(empirical_mode, theoretical_mode)
    norm1 = np.linalg.norm(empirical_mode)
    norm2 = np.linalg.norm(theoretical_mode)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return abs(dot / (norm1 * norm2))


def compare_modes(
    empirical_modes: np.ndarray,
    z_grid: np.ndarray,
    boundary: str = "neumann",
) -> dict:
    """Compare empirical PCA modes to theoretical Fourier modes.

    Parameters
    ----------
    empirical_modes : np.ndarray, shape (n_modes, n_z)
        Empirical eigenvectors from PCA.
    z_grid : np.ndarray, shape (n_z,)
        Spatial grid.
    boundary : str
        Boundary condition for theoretical modes.

    Returns
    -------
    result : dict with keys:
        'overlaps'           : np.ndarray — overlap for each mode
        'theoretical_modes'  : np.ndarray — the Fourier modes used
        'mean_overlap'       : float — average overlap across modes
    """
    n_modes = empirical_modes.shape[0]
    theoretical_modes = fourier_eigenfunctions(z_grid, n_modes, boundary)

    overlaps = np.array([
        mode_overlap(empirical_modes[k], theoretical_modes[k])
        for k in range(n_modes)
    ])

    return {
        "overlaps": overlaps,
        "theoretical_modes": theoretical_modes,
        "mean_overlap": overlaps.mean(),
    }


# ---------------------------------------------------------------------------
# Correlation kernel predicted by the heat equation
# ---------------------------------------------------------------------------

def heat_equation_correlation(
    z1: np.ndarray,
    z2: np.ndarray,
    D: float,
    L: float = 6.0,
    n_terms: int = 100,
) -> np.ndarray:
    """Theoretical correlation function from the stochastic heat equation.

    The correlation between returns at positions z₁ and z₂ is:

        C(z₁,z₂) ∝ Σₖ (1/k²) φₖ(z₁) φₖ(z₂)

    This is the Green's function of the biharmonic operator,
    and depends primarily on |z₁ - z₂|.

    Parameters
    ----------
    z1, z2 : np.ndarray
        Position arrays (will be broadcast).
    D : float
        Diffusion coefficient.
    L : float
        Domain length.
    n_terms : int
        Number of Fourier terms in the sum.

    Returns
    -------
    C : np.ndarray
        Correlation values.
    """
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)

    C = np.zeros(np.broadcast_shapes(z1.shape, z2.shape))
    z1_shifted = z1 - (-L / 2)  # shift to [0, L]
    z2_shifted = z2 - (-L / 2)

    for k in range(1, n_terms + 1):
        weight = 1.0 / (D * (k * np.pi / L) ** 2)
        basis1 = np.cos(k * np.pi * z1_shifted / L)
        basis2 = np.cos(k * np.pi * z2_shifted / L)
        C += weight * basis1 * basis2

    # Normalise to correlation (diagonal = 1)
    C_diag_1 = np.zeros_like(z1, dtype=float)
    C_diag_2 = np.zeros_like(z2, dtype=float)
    for k in range(1, n_terms + 1):
        weight = 1.0 / (D * (k * np.pi / L) ** 2)
        C_diag_1 += weight * np.cos(k * np.pi * z1_shifted / L) ** 2
        C_diag_2 += weight * np.cos(k * np.pi * z2_shifted / L) ** 2

    normalisation = np.sqrt(C_diag_1 * C_diag_2)
    normalisation[normalisation < 1e-10] = 1.0
    C /= normalisation

    return C


def fit_diffusion_coefficient(
    empirical_corr: np.ndarray,
    z_grid: np.ndarray,
) -> dict:
    """Fit the diffusion coefficient D from the empirical correlation matrix.

    Minimises ||C_empirical - C_model(D)||² over D.

    Parameters
    ----------
    empirical_corr : np.ndarray, shape (n_z, n_z)
        Empirical correlation matrix from PCA.
    z_grid : np.ndarray, shape (n_z,)
        Spatial grid.

    Returns
    -------
    result : dict with keys:
        'D'              : float — fitted diffusion coefficient
        'model_corr'     : np.ndarray — model correlation at best-fit D
        'residual_error' : float — Frobenius norm of the residual
        'relative_error' : float — residual / ||C_empirical||
    """
    L = z_grid[-1] - z_grid[0]
    z1, z2 = np.meshgrid(z_grid, z_grid)

    def objective(log_D):
        D = np.exp(log_D)
        C_model = heat_equation_correlation(z1, z2, D, L)
        residual = empirical_corr - C_model
        return np.sum(residual ** 2)

    # Optimise in log-space for positivity
    result = minimize(objective, x0=np.log(0.05), method="Nelder-Mead")
    D_fit = np.exp(result.x[0])

    C_model = heat_equation_correlation(z1, z2, D_fit, L)
    residual_error = np.sqrt(np.sum((empirical_corr - C_model) ** 2))
    empirical_norm = np.sqrt(np.sum(empirical_corr ** 2))

    return {
        "D": D_fit,
        "model_corr": C_model,
        "residual_error": residual_error,
        "relative_error": residual_error / empirical_norm if empirical_norm > 0 else float("inf"),
    }
