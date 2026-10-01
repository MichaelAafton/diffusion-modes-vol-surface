"""Tests for PCA analysis."""

import numpy as np
import pytest
from src.pca import run_pca, project_onto_factors, sign_changes, slope
from src.simulate import generate_synthetic_dataset


class TestRunPCA:
    @pytest.fixture
    def synthetic_returns(self):
        dataset = generate_synthetic_dataset(n_days=500, D=0.05, n_z=30, seed=42)
        return dataset["returns"]

    def test_output_structure(self, synthetic_returns):
        result = run_pca(synthetic_returns, n_components=5)
        assert result.eigenvalues.shape == (5,)
        assert result.eigenvectors.shape == (5, 30)
        assert result.explained_variance_ratio.shape == (5,)
        assert result.correlation_matrix.shape == (30, 30)

    def test_eigenvalues_descending(self, synthetic_returns):
        result = run_pca(synthetic_returns, n_components=10)
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] >= result.eigenvalues[i + 1]

    def test_shares_sum_to_one_with_all_components(self, synthetic_returns):
        result = run_pca(synthetic_returns)
        assert result.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-10)

    def test_eigenvectors_orthogonal(self, synthetic_returns):
        result = run_pca(synthetic_returns, n_components=5)
        for i in range(5):
            for j in range(i + 1, 5):
                dot = np.abs(np.dot(result.eigenvectors[i], result.eigenvectors[j]))
                assert dot < 0.1  # approximately orthogonal

    def test_correlation_matrix_symmetric(self, synthetic_returns):
        result = run_pca(synthetic_returns)
        np.testing.assert_array_almost_equal(
            result.correlation_matrix, result.correlation_matrix.T
        )

    def test_correlation_matrix_unit_diagonal(self, synthetic_returns):
        result = run_pca(synthetic_returns)
        diag = np.diag(result.correlation_matrix)
        np.testing.assert_array_almost_equal(diag, np.ones_like(diag), decimal=5)


class TestProjectOntoFactors:
    def test_projection_shape(self):
        dataset = generate_synthetic_dataset(n_days=200, n_z=20, seed=42)
        result = run_pca(dataset["returns"], n_components=5)
        factors = project_onto_factors(dataset["returns"], result.eigenvectors)
        assert factors.shape == (199, 5)


class TestSpectralSummaries:
    def test_slope_recovers_power_law(self):
        k = np.arange(2, 8)
        assert slope(3.0 * k ** -4.4, k_start=2) == pytest.approx(4.4, rel=1e-9)

    def test_slope_ignores_floor(self):
        evr = np.array([0.5, 0.125, 0.0])
        assert slope(evr) == pytest.approx(2.0, rel=1e-9)

    def test_sign_changes(self):
        assert sign_changes(np.array([-1, -1, 1, 1])) == 1
        assert sign_changes(np.array([1, -1, 1, -1])) == 3
