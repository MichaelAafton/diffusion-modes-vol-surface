"""Tests for the (z, τ) reparameterization, including the √T moneyness scaling."""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    ReparamConfig,
    compute_moneyness,
    compute_psychological_time,
    inverse_psychological_time,
    build_grid,
    preprocess_options_data,
)


class TestMoneyness:
    def test_atm_is_zero(self):
        """At-the-money (K = S) gives z = 0 for any σ, T."""
        z = compute_moneyness(K=100.0, S=100.0, sigma=0.2, T=0.5)
        assert z == pytest.approx(0.0, abs=1e-12)

    def test_sqrt_T_makes_z_comparable_across_maturities(self):
        """The √T scaling is the whole point: a strike that is one standard
        deviation to expiry must map to z ≈ 1 regardless of maturity."""
        S, sigma = 100.0, 0.2
        for T in (0.25, 1.0, 2.0):
            K = S * np.exp(sigma * np.sqrt(T))  # exactly 1 s.d. to expiry
            z = compute_moneyness(K=K, S=S, sigma=sigma, T=T)
            assert z == pytest.approx(1.0, rel=1e-9)

    def test_without_sqrt_T_would_differ(self):
        """Guard against regressing to the (wrong) log(K/S)/σ shortcut."""
        S, sigma = 100.0, 0.2
        K = S * np.exp(sigma * np.sqrt(2.0))
        z_correct = compute_moneyness(K=K, S=S, sigma=sigma, T=2.0)
        z_shortcut = np.log(K / S) / sigma  # missing the √T
        assert not np.isclose(z_correct, z_shortcut)

    def test_vectorised(self):
        K = np.array([90.0, 100.0, 110.0])
        z = compute_moneyness(K=K, S=100.0, sigma=0.2, T=1.0)
        assert z.shape == (3,)
        assert z[0] < 0 < z[2]


class TestPsychologicalTime:
    def test_roundtrip(self):
        ttoexp = np.array([3.0, 20.0, 60.0, 250.0])
        tau = compute_psychological_time(ttoexp, psi=25.0)
        recovered = inverse_psychological_time(tau, psi=25.0)
        np.testing.assert_allclose(recovered, ttoexp, rtol=1e-9)

    def test_compresses_long_horizons(self):
        """Short maturities are spread out; long ones are compressed."""
        tau = compute_psychological_time(np.array([20.0, 40.0, 200.0, 220.0]), psi=25.0)
        near_gap = tau[1] - tau[0]   # 20 → 40 business days
        far_gap = tau[3] - tau[2]    # 200 → 220 business days
        assert near_gap > far_gap


class TestGrid:
    def test_shapes(self):
        grid = build_grid(ReparamConfig(n_z_bins=30, n_tau_bins=10))
        assert grid["z_edges"].shape == (31,)
        assert grid["z_centers"].shape == (30,)
        assert grid["tau_edges"].shape == (11,)
        assert grid["tau_centers"].shape == (10,)


class TestPreprocess:
    def test_adds_columns_and_filters(self):
        # One-year options (√T = 1). z = ±3 ⇒ K/S ∈ [≈0.55, ≈1.82], so K = 500
        # is far out of range and must be dropped.
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02"] * 3),
                "spot": [100.0, 100.0, 100.0],
                "strike": [100.0, 130.0, 500.0],  # last is far OTM → filtered out
                "ttoexp": [252.0, 252.0, 252.0],
                "mid_price": [8.0, 2.0, 0.01],
                "option_type": ["call", "call", "call"],
                "implied_vol": [0.2, 0.2, 0.2],
                "sigma_atm": [0.2, 0.2, 0.2],
            }
        )
        out = preprocess_options_data(df, ReparamConfig(z_min=-3.0, z_max=3.0))
        assert {"z", "tau", "T_years"} <= set(out.columns)
        assert out["T_years"].iloc[0] == pytest.approx(1.0)
        assert len(out) == 2  # the z out-of-range row is dropped
