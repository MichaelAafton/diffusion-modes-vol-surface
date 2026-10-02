"""
Spectral structure of the 30-day SPX smile in vendor (IvyDB) surfaces.

Modules
-------
- data_pipeline : load IvyDB vsurfd surfaces; z-moneyness; the study panel
- pca           : correlation PCA of daily changes, factor projection, tail slope
- simulate      : stochastic heat equation simulator (validation harness)
- forward_model : analytic spectrum of daily changes for the membrane model
"""
