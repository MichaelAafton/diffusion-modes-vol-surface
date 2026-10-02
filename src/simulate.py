"""
Stochastic heat equation simulator (validation harness)
=======================================================

Simulates a 1D membrane on the moneyness window [z_min, z_max] with Neumann
boundaries:

    d_t p(z,t) = D d_z^2 p - kappa d_z^4 p + xi(z,t)

In the cosine basis each mode is an Ornstein-Uhlenbeck process with rate
gamma_k = D q_k^2 + kappa q_k^4, q_k = k pi / L, simulated exactly in time.
An optional uniform level factor (OU, slow) can be added.

The simulator generates known ground truth for checking the pipeline (PCA,
binning, the forward model in ``forward_model.model_spectrum``) and drives the
vendor-kernel harness in ``scripts/test49_vendor_kernel.py``. Agreement on
synthetic data validates code, not physics.

References:
    - Le Coz, V. & Bouchaud, J.-P. -- elastic string models for forward rates
"""
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Shared definitions (used by the simulator and by forward_model.model_spectrum)
# ---------------------------------------------------------------------------

def wavenumbers(n_modes: int, L: float) -> np.ndarray:
    """Physical wavenumbers q_k = k*pi/L for k = 1..n_modes (units: 1/z)."""
    return np.arange(1, n_modes + 1) * np.pi / L


def decay_rates(q: np.ndarray, D: float, kappa: float = 0.0) -> np.ndarray:
    """Relaxation rates gamma_k = D q_k^2 + kappa q_k^4 (per day).

    D has units z^2/day and kappa z^4/day, because q is a physical wavenumber.
    """
    return D * q ** 2 + kappa * q ** 4


def cosine_basis(q: np.ndarray, z: np.ndarray, z_min: float, L: float) -> np.ndarray:
    """Neumann cosine modes sqrt(2/L) cos(q_k (z - z_min)), shape (n_modes, len(z))."""
    return np.sqrt(2 / L) * np.cos(np.outer(q, np.asarray(z, dtype=float) - z_min))


def bin_centres(z_min: float, z_max: float, n_bins: int) -> np.ndarray:
    """Centres of n_bins uniform bins on [z_min, z_max] (the panel's sample points)."""
    edges = np.linspace(z_min, z_max, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


@dataclass
class SPDEConfig:
    """Configuration for the stochastic heat equation simulation.

    Parameters
    ----------
    D : float
        Diffusion coefficient (z^2/day).
    n_z : int
        Number of sample points when ``z_points`` is None (linspace on the window).
    z_min, z_max : float
        Window boundaries (Neumann). Pass them explicitly whenever ``z_points``
        is given: the defaults describe a wide synthetic window, not the study's.
    n_modes : int
        Number of cosine modes simulated.
    noise_amplitude : float
        Amplitude of the white-noise forcing.
    dt : float
        Time step in trading days (1 = daily).
    seed : int or None
        Random seed.
    kappa : float
        Bending stiffness (z^4/day); 0 gives pure diffusion.
    include_mean_mode, mean_mode_decay_rate, mean_mode_noise_std :
        Optional uniform level factor (OU): on/off, decay per day, daily kick s.d.
    z_points : tuple of float or None
        Sample points; None means linspace(z_min, z_max, n_z).
    """
    D: float = 0.05
    n_z: int = 50
    z_min: float = -3.0
    z_max: float = 3.0
    n_modes: int = 25
    noise_amplitude: float = 1.0
    dt: float = 1.0
    seed: int | None = 42
    kappa: float = 0.0
    include_mean_mode: bool = False  # extra slow "market level" mode; OFF for the clean harness
    mean_mode_decay_rate: float = 0.002  # per day
    mean_mode_noise_std: float = 0.5  # daily kick size of the level factor
    z_points: tuple[float, ...] | None = None  # sample points; None = linspace(z_min, z_max, n_z)


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
        if config.z_points is not None:
            self.z_grid = np.asarray(config.z_points, dtype=float)
        else:
            self.z_grid = np.linspace(config.z_min, config.z_max, config.n_z)
        self.n_points = len(self.z_grid)

        # Modes k = 1..n_modes (k = 0, the mean, is handled separately)
        self.k = np.arange(1, config.n_modes + 1)
        self.wavenumbers = wavenumbers(config.n_modes, self.L)

        # gamma_k = D q_k^2 + kappa q_k^4 with physical wavenumbers q_k = k*pi/L
        self.decay_rates = decay_rates(self.wavenumbers, config.D, config.kappa)

        # Stationary variance of each mode: sigma^2_noise / (2 gamma_k)
        self.stationary_variance = (
            config.noise_amplitude ** 2 / (2 * self.decay_rates)
        )

        # Orthonormal cosine basis evaluated at the sample points
        self.basis_functions = cosine_basis(
            self.wavenumbers, self.z_grid, config.z_min, self.L
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
        mean_decay = np.exp(-cfg.mean_mode_decay_rate * dt)
        mean_noise_std = cfg.mean_mode_noise_std * np.sqrt(dt)

        # Storage for output
        surface = np.zeros((n_days, self.n_points))

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

        Stationary level variances, proportional to 1/gamma_k (1/k² for
        pure diffusion). This is the ground truth for PCA on LEVELS; for
        daily changes use theoretical_increment_eigenvalues.

        Parameters
        ----------
        n_components : int
            Number of eigenvalues to return.

        Returns
        -------
        eigenvalues : np.ndarray, shape (n_components,)
            Theoretical eigenvalues (proportional to 1/gamma_k).
        """
        eigenvalues = 1.0 / self.decay_rates[:n_components]
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

    Produces known ground truth so the downstream pipeline can be checked.

    Parameters
    ----------
    n_days : int
        Number of trading days.
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
