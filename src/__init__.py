"""
Heat-Equation Dynamics in the Volatility Surface.

A pipeline for testing whether option-surface dynamics behave like a stochastic
heat equation — validated on synthetic ground truth, then tested on real data.

Modules
-------
- simulate       : stochastic heat-equation simulator (the validation harness)
- black_scholes  : BS pricing, Greeks, IV inversion, delta-hedged P&L
- data_pipeline  : load real + synthetic data; (z, τ) reparameterization
- pca            : PCA, rolling-window stability, factor projection
- heat_equation  : analytical Fourier modes, eigenvalue scaling, mode overlap
- calibration    : field-theory (κ, μ) calibration
- factor_model   : PCA-based factor model for option-portfolio risk
- plotting       : consistent, publication-quality figures
"""
