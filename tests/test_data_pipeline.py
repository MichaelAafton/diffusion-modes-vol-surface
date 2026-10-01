"""Tests for moneyness, loading and the study panel."""

import numpy as np
import pandas as pd
import pytest

from scipy.stats import norm

from src.data_pipeline import (
    ReparamConfig,
    build_study_panel,
    build_surface_panel,
    compute_moneyness,
    load_optionmetrics,
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
        assert {"z", "T_years"} <= set(out.columns)
        assert out["T_years"].iloc[0] == pytest.approx(1.0)
        assert len(out) == 2  # the z out-of-range row is dropped


def _vendor_like_raw(days_list=(10, 30, 60), n_dates=40, seed=0):
    """Pillars in load_optionmetrics schema: deltas x100, calls and puts, several maturities."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2020-01-02", periods=n_dates):
        atm = 0.2 + 0.01 * rng.normal()
        for days in days_list:
            T = days / 365
            for delta in list(range(10, 55, 5)) + list(range(-50, -5, 5)):
                ce = delta / 100 if delta > 0 else 1 + delta / 100
                lnK = -norm.ppf(ce) * atm * np.sqrt(T) + 0.5 * atm ** 2 * T
                rows.append(dict(days=days, date=d, spot=100.0, strike=100 * np.exp(lnK),
                                 ttoexp=days * 252 / 365, mid_price=1.0,
                                 option_type="C" if delta > 0 else "P",
                                 implied_vol=atm, sigma_atm=atm, delta=delta))
    return pd.DataFrame(rows)


class TestLoadOptionmetrics:
    def test_coerces_numeric_columns_and_keeps_otm(self, tmp_path):
        """Vendor columns arrive as strings/Decimals; the loader must coerce them."""
        surf = pd.DataFrame({
            "date": ["2020-01-02"] * 4,
            "days": ["30", "30", "30", "30"],
            "delta": ["50", "-50", "25", "75"],
            "cp_flag": ["C", "P", "C", "C"],
            "impl_volatility": ["0.20", "0.22", "0.18", "0.25"],
            "impl_strike": ["100", "100", "105", "95"],
            "impl_premium": ["2.0", "2.1", "0.8", "5.5"],
        })
        spot = pd.DataFrame({"date": ["2020-01-02"], "spot": ["100.0"]})
        surf.to_parquet(tmp_path / "s.parquet")
        spot.to_parquet(tmp_path / "p.parquet")

        df = load_optionmetrics(tmp_path / "s.parquet", tmp_path / "p.parquet")
        for col in ["spot", "strike", "implied_vol", "mid_price", "delta", "days", "sigma_atm"]:
            assert pd.api.types.is_numeric_dtype(df[col]), col
        assert (df["delta"].abs() <= 50).all()          # |delta| = 75 (ITM) dropped
        assert df["sigma_atm"].iloc[0] == pytest.approx(0.21)  # mean of the two 50-delta vols


class TestSurfacePanel:
    def test_refuses_mixed_maturities(self):
        df = preprocess_options_data(_vendor_like_raw(), ReparamConfig())
        with pytest.raises(ValueError, match="maturities"):
            build_surface_panel(df, ReparamConfig())

    def test_drops_bins_below_coverage(self):
        raw = _vendor_like_raw(days_list=(30,))
        df = preprocess_options_data(raw, ReparamConfig())
        first_bin_hi = -1.15 + 2.25 / 8
        sparse = df[df["z"] < first_bin_hi]
        df = df.drop(sparse.index[sparse["date"].isin(df["date"].unique()[::3])])
        out = build_surface_panel(df, ReparamConfig(), min_coverage=0.95)
        assert 0 not in out["kept_bins"]
        assert out["n_bins_dropped"] == 1
        assert len(out["z_centers"]) == out["panel"].shape[1]


class TestStudyPanel:
    def test_uses_only_the_30_day_slice(self):
        out = build_study_panel(_vendor_like_raw())
        assert out["maturities"] == [30]
        assert out["panel"].shape[0] == 40
