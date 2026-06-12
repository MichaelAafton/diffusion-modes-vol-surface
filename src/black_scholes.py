"""
Black-Scholes Pricing and Implied Volatility
=============================================

Implements the Black-Scholes model — not because it's accurate (it isn't),
but because it serves as the *lingua franca* of options markets.

Options are quoted in terms of implied volatility: the σ you'd plug into
Black-Scholes to reproduce the observed market price. The collection of
implied vols across all strikes and maturities forms the "volatility surface".

Key functions:
    bs_price       — Black-Scholes option price
    bs_delta       — Delta hedge ratio (∂P/∂S)
    bs_vega        — Sensitivity to volatility (∂P/∂σ)
    implied_vol    — Invert BS to find σ from market price
    delta_hedged_pnl — Compute delta-hedged P&L
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    option_type: str = "call",
) -> float:
    """Black-Scholes European option price.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry (in years).
    sigma : float
        Volatility (annualised).
    r : float
        Risk-free interest rate (default 0 for simplicity).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    price : float
        Black-Scholes option price.

    Notes
    -----
    The formula assumes geometric Brownian motion for the underlying:
        dS/S = r dt + σ dW

    which gives:
        C = S N(d₁) - K e^{-rT} N(d₂)
        P = K e^{-rT} N(-d₂) - S N(-d₁)

    where d₁ = [log(S/K) + (r + σ²/2)T] / (σ√T) and d₂ = d₁ - σ√T.
    """
    if T <= 0:
        # At expiry, return intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    option_type: str = "call",
) -> float:
    """Black-Scholes delta: ∂P/∂S.

    This is the hedge ratio used to construct delta-hedged P&L:
        Δ ≡ ∂P_BS(K, S, T, σ) / ∂S

    Parameters
    ----------
    S, K, T, sigma, r, option_type : same as bs_price.

    Returns
    -------
    delta : float
        The delta of the option.
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def bs_vega(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """Black-Scholes vega: ∂P/∂σ.

    Useful for understanding the sensitivity of option prices to
    volatility changes. Same for calls and puts.

    Returns
    -------
    vega : float
        The vega of the option.
    """
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "call",
    bounds: tuple[float, float] = (0.001, 5.0),
) -> float | None:
    """Invert Black-Scholes to find implied volatility.

    Given a market price, find the σ such that BS(σ) = market_price.
    Uses Brent's root-finding method.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S, K, T, r, option_type : same as bs_price.
    bounds : tuple
        Search interval for σ.

    Returns
    -------
    sigma : float or None
        Implied volatility, or None if no solution found.
    """
    def objective(sigma):
        return bs_price(S, K, T, sigma, r, option_type) - market_price

    try:
        return brentq(objective, bounds[0], bounds[1], xtol=1e-8)
    except ValueError:
        # No root in the interval — price may be below intrinsic
        return None


def delta_hedged_pnl(
    prices_today: np.ndarray,
    prices_yesterday: np.ndarray,
    spot_today: float,
    spot_yesterday: float,
    deltas_yesterday: np.ndarray,
) -> np.ndarray:
    """Compute single-step delta-hedged P&L for a set of options.

    The delta-hedged P&L removes the directional exposure to the
    underlying, isolating the "pure volatility" component of returns:

        P&L_t = P_t - P_{t-1} - Δ_{t-1} (S_t - S_{t-1})

    This is the quantity whose correlations across strikes reveal (or fail to
    reveal) the heat-equation structure.

    Note: this is one hedge step. The *hedging frequency* used to build a full
    P&L series is a modelling choice that injects noise and bias — document it
    explicitly (see ``data_pipeline.ReparamConfig.hedge_frequency``).

    Parameters
    ----------
    prices_today : np.ndarray, shape (n_options,)
        Option mid-prices today.
    prices_yesterday : np.ndarray, shape (n_options,)
        Option mid-prices yesterday.
    spot_today : float
        Underlying spot price today.
    spot_yesterday : float
        Underlying spot price yesterday.
    deltas_yesterday : np.ndarray, shape (n_options,)
        Delta of each option as of yesterday.

    Returns
    -------
    pnl : np.ndarray, shape (n_options,)
        Delta-hedged P&L for each option.
    """
    price_change = prices_today - prices_yesterday
    hedge_pnl = deltas_yesterday * (spot_today - spot_yesterday)
    return price_change - hedge_pnl


# ---------------------------------------------------------------------------
# Vectorised versions for arrays of options
# ---------------------------------------------------------------------------

def bs_price_vec(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.0,
    option_type: str = "call",
) -> np.ndarray:
    """Vectorised Black-Scholes price over arrays of K, T, sigma.

    Parameters
    ----------
    S : float
        Spot price (scalar).
    K : np.ndarray
        Array of strike prices.
    T : np.ndarray
        Array of times to expiry (must broadcast with K).
    sigma : np.ndarray
        Array of volatilities (must broadcast with K).
    r : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    prices : np.ndarray
        Black-Scholes prices.
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle T=0 cases
    mask = T > 0
    prices = np.zeros_like(K, dtype=float)

    if option_type == "call":
        prices[~mask] = np.maximum(S - K[~mask], 0)
    else:
        prices[~mask] = np.maximum(K[~mask] - S, 0)

    if mask.any():
        d1 = (np.log(S / K[mask]) + (r + 0.5 * sigma[mask] ** 2) * T[mask]) / (
            sigma[mask] * np.sqrt(T[mask])
        )
        d2 = d1 - sigma[mask] * np.sqrt(T[mask])

        if option_type == "call":
            prices[mask] = S * norm.cdf(d1) - K[mask] * np.exp(-r * T[mask]) * norm.cdf(d2)
        else:
            prices[mask] = K[mask] * np.exp(-r * T[mask]) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return prices
