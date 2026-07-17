"""
Stochastic Heat Equation Simulator — the Validation Harness
============================================================

This simulator is the project's **validation harness**, not its study. By
generating data from a known stochastic heat equation and running the entire
downstream pipeline on it, we confirm the PCA / reparameterization / fitting code
recovers what we put in (sinusoidal modes, λₖ ~ k⁻², the correct diffusion
constant D). A clean fit here proves the *code* is correct — it is not a finding.
The findings come from real data.

Simulates the stochastic PDE hypothesized to govern volatility-surface dynamics:

    ∂ₜp(z,t) = D ∂²_z p(z,t) + ξ(z,t)

where:
    p(z,t) = perturbation of the vol surface at moneyness z, time t
    D      = diffusion coefficient ("stiffness" of the surface)
    ξ(z,t) = spatiotemporal white noise (random trading shocks)

The simulation works in Fourier space, where each mode evolves independently
as an Ornstein-Uhlenbeck process:

    ∂ₜfₖ = -Dk² fₖ + ξₖ(t)

with stationary variance ⟨fₖ²⟩ = σ²_noise / (2Dk²).

This is the same equation that describes heat diffusion in a rod, or the
fluctuations of an elastic string — the physical analogy at the heart of the
project.

References:
    - Ioselevich, P. "A Data-Driven Factor Model for Option Risk"
    - Le Coz, V. & Bouchaud, J.-P. — elastic string models for forward rates
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class SPDEConfig:
    """Configuration for the stochastic heat equation simulation.

    Parameters
    ----------
    D : float
        Diffusion coefficient. Controls the "stiffness" of the surface.
        Higher D = stiffer surface = perturbations spread/decay faster.
        Typical values from the presentation: D ~ 0.05 in normalised units.
    n_z : int
        Number of spatial grid points (moneyness axis).
    z_min : float
        Left boundary of the moneyness domain (e.g. -3 for 3σ puts).
    z_max : float
        Right boundary of the moneyness domain (e.g. +3 for 3σ calls).
    n_modes : int
        Number of Fourier modes to simulate. More modes = finer spatial
        resolution but also more noise. The presentation used ~50 grid
        points, so 20-30 modes is a good starting point.
    noise_amplitude : float
        Amplitude of the white noise driving term.
    dt : float
        Time step (in units of trading days). dt=1 means daily.
    seed : int or None
        Random seed for reproducibility.
    """
    D: float = 0.05
    n_z: int = 50
    z_min: float = -3.0
    z_max: float = 3.0
    n_modes: int = 25
    noise_amplitude: float = 1.0
    dt: float = 1.0
    seed: int | None = 42
    include_mean_mode: bool = False  # extra slow "market level" mode; OFF for the clean harness


class StochasticHeatEquation:
    """Simulator for the 1D stochastic heat equation.

    The simulation is performed in Fourier space for efficiency and
    numerical stability. Each Fourier mode is an independent OU process,
    which can be simulated exactly (no discretisation error in time).

    Example
    -------
    >>> config = SPDEConfig(D=0.05, n_z=50, n_modes=25)
    >>> spde = StochasticHeatEquation(config)
    >>> data = spde.simulate(n_days=2000)
    >>> print(data.shape)  # (2000, 50) — daily snapshots of the surface
    """

    def __init__(self, config: SPDEConfig | None = None):
        if config is None:
            config = SPDEConfig()
        self.config = config
        self.L = config.z_max - config.z_min  # domain length
        self.z_grid = np.linspace(config.z_min, config.z_max, config.n_z)

        # Wavenumbers for modes k = 1, 2, ..., n_modes
        # (k=0 is the mean/constant mode, handled separately)
        self.k_values = np.arange(1, config.n_modes + 1)
        self.wavenumbers = self.k_values * np.pi / self.L

        # Decay rates for each mode: γₖ = D * (kπ/L)²
        self.decay_rates = config.D * self.wavenumbers ** 2

        # Stationary variance of each mode: σ²_noise / (2γₖ)
        self.stationary_variance = (
            config.noise_amplitude ** 2 / (2 * self.decay_rates)
        )

        # Pre-compute basis functions (cosines) on the spatial grid
        # φₖ(z) = √(2/L) * cos(kπ(z - z_min)/L)
        # The √(2/L) normalisation makes them orthonormal
        z_shifted = self.z_grid - config.z_min
        self.basis_functions = np.zeros((config.n_modes, config.n_z))
        for i, k in enumerate(self.k_values):
            self.basis_functions[i, :] = np.sqrt(2 / self.L) * np.cos(
                k * np.pi * z_shifted / self.L
            )

        self.rng = np.random.default_rng(config.seed)

    def simulate(self, n_days: int) -> np.ndarray:
        """Simulate the SPDE and return daily snapshots of the surface.

        Parameters
        ----------
        n_days : int
            Number of trading days to simulate.

        Returns
        -------
        surface : np.ndarray, shape (n_days, n_z)
            Each row is the surface perturbation p(z) on that day.
        """
        cfg = self.config
        dt = cfg.dt

        # Exact OU simulation parameters
        # For each mode: fₖ(t+dt) = fₖ(t) * exp(-γₖ dt) + noise
        # where noise ~ N(0, σ²ₖ (1 - exp(-2γₖ dt)))
        exp_decay = np.exp(-self.decay_rates * dt)
        noise_std = np.sqrt(
            self.stationary_variance * (1 - exp_decay ** 2)
        )

        # Initialise modes from their stationary distribution
        f_modes = self.rng.normal(
            0, np.sqrt(self.stationary_variance), size=cfg.n_modes
        )

        # Also track a slowly-varying mean (mode 0)
        f_mean = 0.0
        mean_decay = np.exp(-cfg.D * 0.01 * dt)  # very slow decay for mean
        mean_noise_std = cfg.noise_amplitude * np.sqrt(dt) * 0.5

        # Storage for output
        surface = np.zeros((n_days, cfg.n_z))

        for t in range(n_days):
            # Reconstruct surface from Fourier modes
            # p(z) = f_mean + Σₖ fₖ φₖ(z)
            surface[t, :] = f_mean + f_modes @ self.basis_functions

            # Evolve each mode (exact OU step)
            f_modes = (
                f_modes * exp_decay
                + self.rng.normal(0, 1, size=cfg.n_modes) * noise_std
            )
            if cfg.include_mean_mode:
                f_mean = (
                        f_mean * mean_decay
                        + self.rng.normal(0, mean_noise_std)
                )

        return surface

    def compute_returns(self, surface: np.ndarray) -> np.ndarray:
        """Compute daily changes (returns) of the surface.

        Parameters
        ----------
        surface : np.ndarray, shape (n_days, n_z)
            Output from simulate().

        Returns
        -------
        returns : np.ndarray, shape (n_days - 1, n_z)
            Daily changes: returns[t] = surface[t+1] - surface[t].
        """
        return np.diff(surface, axis=0)

    def theoretical_eigenvalues(self, n_components: int) -> np.ndarray:
        """Return the theoretical eigenvalue spectrum.

        For the stochastic heat equation, the variance of mode k
        scales as 1/k². This provides the ground truth for comparison
        with empirical PCA eigenvalues.

        Parameters
        ----------
        n_components : int
            Number of eigenvalues to return.

        Returns
        -------
        eigenvalues : np.ndarray, shape (n_components,)
            Theoretical eigenvalues (proportional to 1/k²).
        """
        k = np.arange(1, n_components + 1)
        eigenvalues = 1.0 / (self.config.D * (k * np.pi / self.L) ** 2)
        # Normalise so they sum to the same as explained variance fractions
        eigenvalues = eigenvalues / eigenvalues.sum()
        return eigenvalues

    def theoretical_eigenvectors(self, n_components: int) -> np.ndarray:
        """Return the theoretical eigenvectors (cosine basis functions).

        Parameters
        ----------
        n_components : int
            Number of eigenvectors to return.

        Returns
        -------
        eigenvectors : np.ndarray, shape (n_components, n_z)
            Each row is one eigenfunction φₖ(z).
        """
        return self.basis_functions[:n_components, :]

    def theoretical_increment_eigenvalues(self, n_components: int) -> np.ndarray:
        """Theoretical spectrum for PCA on DAILY CHANGES (returns).

           The k^-2 law describes the stationary variance of mode LEVELS.
           Increments of an OU process obey a different law:

               Var(Δf_k) = 2 V_k (1 - exp(-γ_k dt)),   V_k = σ²/(2γ_k)

           which is ~flat (≈ σ² dt) for slow modes (γ_k dt << 1) and
           ~k^-2 (→ 2 V_k) for fast modes (γ_k dt >> 1). This is the
           correct ground truth for the validation harness, since the
           pipeline runs PCA on daily changes, not levels.
           """
        var_inc = 2 * self.stationary_variance * (1 - np.exp(-self.decay_rates * self.config.dt))
        var_inc = var_inc / var_inc.sum()
        return var_inc[:n_components]


class StochasticHeatEquation2D:
    """Simulator for the 2D stochastic heat equation on (z, τ) space.

    Extends the 1D model to include the maturity (time-to-expiry) dimension:

        ∂ₜp(z,τ,t) = D_z ∂²_z p + D_τ ∂²_τ p + ξ(z,τ,t)

    where τ is "psychological time" (log-transformed maturity).

    This corresponds to the full 2D ``(z, τ)`` analysis (Phases 3–5).

    TODO: Implement alongside the 2D PCA / field-theory work.
    """

    def __init__(self):
        raise NotImplementedError(
            "2D SPDE simulator — implement this in Phase 4. "
            "Start with the 1D version first."
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_days: int = 2000,
    D: float = 0.05,
    n_z: int = 50,
    seed: int = 42,
) -> dict:
    """Generate a complete synthetic dataset for the validation harness.

    Entry point for the Phase 1 harness: it produces known ground truth so the
    downstream pipeline can be verified before it is pointed at real data.

    Parameters
    ----------
    n_days : int
        Number of trading days (~8 years at 250 days/year).
    D : float
        Diffusion coefficient.
    n_z : int
        Spatial grid resolution.
    seed : int
        Random seed.

    Returns
    -------
    dataset : dict with keys:
        'z_grid'    : np.ndarray, shape (n_z,) — moneyness grid points
        'surface'   : np.ndarray, shape (n_days, n_z) — surface levels
        'returns'   : np.ndarray, shape (n_days-1, n_z) — daily changes
        'config'    : SPDEConfig — simulation parameters
    """
    config = SPDEConfig(D=D, n_z=n_z, seed=seed)
    spde = StochasticHeatEquation(config)
    surface = spde.simulate(n_days)
    returns = spde.compute_returns(surface)

    return {
        "z_grid": spde.z_grid,
        "surface": surface,
        "returns": returns,
        "config": config,
    }
