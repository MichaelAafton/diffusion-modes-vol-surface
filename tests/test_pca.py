"""Tests for PCA analysis."""

import numpy as np
import pytest
from src.pca import run_pca, rolling_pca, align_eigenvector_signs, project_onto_factors
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

    def test_explained_variance_sums_to_less_than_one(self, synthetic_returns):
        result = run_pca(synthetic_returns, n_components=5)
        assert result.explained_variance_ratio.sum() <= 1.0 + 1e-6

    def test_first_mode_dominates(self, synthetic_returns):
        """For heat equation data, mode 0 should explain the most variance."""
        result = run_pca(synthetic_returns, n_components=5)
        # Mode 0 should have the largest eigenvalue
        assert result.explained_variance_ratio[0] == result.explained_variance_ratio.max()

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


class TestRollingPCA:
    def test_rolling_output_length(self):
        dataset = generate_synthetic_dataset(n_days=600, seed=42)
        results = rolling_pca(dataset["returns"], window_size=250, step_size=50)
        # (599 - 250) / 50 + 1 = 7.98 → 7 windows
        assert len(results) >= 5

    def test_rolling_consistency(self):
        """Each window should produce valid PCA results."""
        dataset = generate_synthetic_dataset(n_days=600, seed=42)
        results = rolling_pca(dataset["returns"], window_size=250, step_size=100, n_components=3)
        for r in results:
            assert r.eigenvalues.shape == (3,)
            assert all(r.eigenvalues[i] >= r.eigenvalues[i + 1] for i in range(2))


class TestAlignSigns:
    def test_sign_alignment(self):
        """Flipped vectors should be realigned."""
        v1 = np.array([[1, 2, 3], [4, 5, 6]])
        v2 = np.array([[-1, -2, -3], [4, 5, 6]])  # mode 0 flipped
        aligned = align_eigenvector_signs([v1, v2])
        # After alignment, both should point the same way as v1
        assert np.dot(aligned[1][0], v1[0]) > 0


class TestProjectOntoFactors:
    def test_projection_shape(self):
        dataset = generate_synthetic_dataset(n_days=200, n_z=20, seed=42)
        result = run_pca(dataset["returns"], n_components=5)
        factors = project_onto_factors(dataset["returns"], result.eigenvectors)
        assert factors.shape == (199, 5)
