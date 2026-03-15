"""Tests for Black-Scholes pricing and implied volatility."""

import numpy as np
import pytest
from src.black_scholes import bs_price, bs_delta, bs_vega, implied_vol


class TestBSPrice:
    def test_atm_call(self):
        """ATM call with S=K should have a known approximate value."""
        price = bs_price(S=100, K=100, T=1.0, sigma=0.2)
        # BS ATM call ≈ S * σ * √T * 0.4 (rough approximation)
        assert 5 < price < 12

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        S, K, T, sigma, r = 100, 105, 0.5, 0.25, 0.03
        call = bs_price(S, K, T, sigma, r, "call")
        put = bs_price(S, K, T, sigma, r, "put")
        parity = S - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=1e-8)

    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic value."""
        price = bs_price(S=200, K=100, T=0.01, sigma=0.2)
        assert price == pytest.approx(100.0, abs=1.0)

    def test_deep_otm_put(self):
        """Deep OTM put should be near zero."""
        price = bs_price(S=200, K=100, T=0.1, sigma=0.2, option_type="put")
        assert price < 0.01

    def test_expired_call(self):
        assert bs_price(S=110, K=100, T=0, sigma=0.2, option_type="call") == 10.0
        assert bs_price(S=90, K=100, T=0, sigma=0.2, option_type="call") == 0.0

    def test_expired_put(self):
        assert bs_price(S=90, K=100, T=0, sigma=0.2, option_type="put") == 10.0


class TestBSDelta:
    def test_atm_call_delta(self):
        """ATM call delta should be close to 0.5."""
        delta = bs_delta(S=100, K=100, T=1.0, sigma=0.2)
        assert 0.45 < delta < 0.65

    def test_call_delta_range(self):
        """Call delta should be in [0, 1]."""
        delta = bs_delta(S=100, K=120, T=0.5, sigma=0.3)
        assert 0 <= delta <= 1

    def test_put_delta_negative(self):
        """Put delta should be negative."""
        delta = bs_delta(S=100, K=100, T=1.0, sigma=0.2, option_type="put")
        assert -1 <= delta <= 0


class TestBSVega:
    def test_vega_positive(self):
        """Vega should always be positive."""
        vega = bs_vega(S=100, K=100, T=1.0, sigma=0.2)
        assert vega > 0

    def test_vega_atm_max(self):
        """Vega is highest ATM."""
        vega_atm = bs_vega(S=100, K=100, T=1.0, sigma=0.2)
        vega_otm = bs_vega(S=100, K=150, T=1.0, sigma=0.2)
        assert vega_atm > vega_otm


class TestImpliedVol:
    def test_roundtrip(self):
        """implied_vol(bs_price(σ)) should return σ."""
        sigma_in = 0.25
        price = bs_price(S=100, K=100, T=1.0, sigma=sigma_in)
        sigma_out = implied_vol(price, S=100, K=100, T=1.0)
        assert sigma_out == pytest.approx(sigma_in, abs=1e-6)

    def test_roundtrip_put(self):
        sigma_in = 0.30
        price = bs_price(S=100, K=110, T=0.5, sigma=sigma_in, option_type="put")
        sigma_out = implied_vol(price, S=100, K=110, T=0.5, option_type="put")
        assert sigma_out == pytest.approx(sigma_in, abs=1e-6)

    def test_invalid_price(self):
        """Price below intrinsic should return None."""
        result = implied_vol(market_price=-1.0, S=100, K=100, T=1.0)
        assert result is None
