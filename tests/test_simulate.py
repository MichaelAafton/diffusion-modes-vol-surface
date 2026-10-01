"""Tests for the stochastic heat equation simulator."""

import numpy as np
import pytest

from src.pca import run_pca
from src.calibration import model_spectrum
from src.simulate import (
    SPDEConfig, StochasticHeatEquation, bin_centres, generate_synthetic_dataset,
)


class TestSPDEConfig:
    def test_default_config(self):
        config = SPDEConfig()
        assert config.D == 0.05
        assert config.n_z == 50
        assert config.z_min == -3.0
        assert config.z_max == 3.0

    def test_custom_config(self):
        config = SPDEConfig(D=0.1, n_z=100)
        assert config.D == 0.1
        assert config.n_z == 100


class TestStochasticHeatEquation:
    @pytest.fixture
    def spde(self):
        config = SPDEConfig(D=0.05, n_z=50, n_modes=25, seed=42)
        return StochasticHeatEquation(config)

    def test_simulate_shape(self, spde):
        surface = spde.simulate(n_days=100)
        assert surface.shape == (100, 50)

    def test_simulate_reproducible(self, spde):
        """Same seed should give same results."""
        s1 = spde.simulate(n_days=50)
        spde2 = StochasticHeatEquation(SPDEConfig(seed=42))
        s2 = spde2.simulate(n_days=50)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_returns_shape(self, spde):
        surface = spde.simulate(n_days=100)
        returns = spde.compute_returns(surface)
        assert returns.shape == (99, 50)

    def test_z_grid(self, spde):
        assert len(spde.z_grid) == 50
        assert spde.z_grid[0] == pytest.approx(-3.0)
        assert spde.z_grid[-1] == pytest.approx(3.0)

    def test_stationary_variance_scaling(self, spde):
        """Pure diffusion: V_k = 1/(2 D q_k^2), so V_1 / V_2 = 4."""
        var = spde.stationary_variance
        ratio = var[0] / var[1]
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_theoretical_eigenvalues(self, spde):
        evals = spde.theoretical_eigenvalues(5)
        assert len(evals) == 5
        assert evals.sum() == pytest.approx(1.0, rel=0.01)
        # Should be descending
        assert all(evals[i] >= evals[i + 1] for i in range(len(evals) - 1))

    def test_theoretical_eigenvectors_orthonormal(self, spde):
        evecs = spde.theoretical_eigenvectors(5)
        # Check approximate orthogonality (numerical quadrature)
        dz = spde.z_grid[1] - spde.z_grid[0]
        for i in range(5):
            for j in range(5):
                inner = np.sum(evecs[i] * evecs[j]) * dz
                if i == j:
                    assert inner == pytest.approx(1.0, abs=0.1)
                else:
                    assert abs(inner) < 0.15

    def test_pca_recovers_increment_spectrum(self):
        """PCA on daily changes must match the OU increment-variance theory,
        Var(Δf_k) ∝ 2 V_k (1 - exp(-γ_k dt)) — NOT the naive k^-2 law."""
        cfg = SPDEConfig(D=0.05, n_z=50, seed=42)  # mean mode off by default
        spde = StochasticHeatEquation(cfg)
        surface = spde.simulate(20000)
        pca = run_pca(np.diff(surface, axis=0), n_components=10)
        theory = spde.theoretical_increment_eigenvalues(10)
        np.testing.assert_allclose(
            pca.explained_variance_ratio, theory, atol=0.012
        )


class TestGenerateSyntheticDataset:
    def test_output_keys(self):
        dataset = generate_synthetic_dataset(n_days=100)
        assert "z_grid" in dataset
        assert "surface" in dataset
        assert "returns" in dataset
        assert "config" in dataset

    def test_output_shapes(self):
        dataset = generate_synthetic_dataset(n_days=200, n_z=30)
        assert dataset["surface"].shape == (200, 30)
        assert dataset["returns"].shape == (199, 30)
        assert len(dataset["z_grid"]) == 30


class TestForwardModel:
    """calibration.model_spectrum against direct simulation on the study instrument."""

    Z = bin_centres(-1.15, 1.10, 8)[1:]          # the 7 kept bin centres

    def test_shares_sum_to_one_and_descend(self):
        s = model_spectrum(0.35, 0.003, 7.0, self.Z)
        assert s.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(s) <= 0)

    def test_matches_simulation(self):
        D, kappa, m = 0.3529, 0.002868, 6.99
        analytic = model_spectrum(D, kappa, m, self.Z)
        sims = []
        for seed in range(3):
            cfg = SPDEConfig(D=D, kappa=kappa, n_modes=40, z_min=-1.15, z_max=1.10,
                             z_points=tuple(self.Z), include_mean_mode=True,
                             mean_mode_noise_std=np.sqrt(m), mean_mode_decay_rate=0.0,
                             seed=seed)
            surface = StochasticHeatEquation(cfg).simulate(20000)
            sims.append(run_pca(np.diff(surface, axis=0)).explained_variance_ratio)
        np.testing.assert_allclose(np.mean(sims, axis=0)[:4], analytic[:4], rtol=0.1)
