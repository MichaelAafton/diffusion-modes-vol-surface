# Discovering Heat Equation Dynamics in the Volatility Surface

A data-driven investigation into how option returns co-move across the volatility surface, revealing that the correlation structure is consistent with a stochastic heat (diffusion) equation — the same PDE that governs thermal diffusion in physics.

Inspired by the work of **Pavel Ioselevich** (Capital Fund Management) presented at the Cambridge Quant Conference 2026, and building on the elastic string models of **Victor Le Coz & Jean-Philippe Bouchaud**.

---

## Overview

Option markets produce a 2D surface of implied volatilities across strikes and maturities. This project studies **how that surface moves over time** by:

1. **Reparameterizing** strikes into normalised moneyness $z = \log(K/S) / \sigma$ and maturities into psychological time $\tau = \psi \log(1 + T/\psi)$
2. **Running PCA** on delta-hedged option returns to extract the dominant modes of variation
3. **Showing** that these eigenmodes are sinusoidal — consistent with eigenfunctions of the Laplacian $\partial^2/\partial z^2$
4. **Demonstrating** that eigenvalues decay as $\sim k^{-2}$, the signature of a diffusion/heat process
5. **Fitting** a stochastic heat equation $\partial_t p = D\partial_z^2 p + \xi(z,t)$ to the empirical correlation structure
6. **Building** a practical factor model for option portfolio risk

## Physics Connection

The volatility surface behaves like a **vibrating elastic membrane**:
- Random trading shocks perturb it locally (the noise term $\xi$)
- Perturbations diffuse across nearby strikes and maturities (the $D\partial_z^2$ term)
- Short-wavelength disturbances decay faster than long-wavelength ones ($\sim k^{-2}$ eigenvalue scaling)
- The "stiffness" parameter $D$ is remarkably consistent across equity indices, commodities, and FX

This is an example of **universality**: the same mathematical structure emerges in finance as in physics — not because markets imitate nature, but because both systems involve the aggregation of many small, local, random perturbations.

## Project Structure

```
vol-surface-dynamics/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── setup.py
│
├── src/                        # Core library
│   ├── __init__.py
│   ├── simulate.py             # Stochastic PDE simulation
│   ├── black_scholes.py        # BS pricing, Greeks, IV inversion
│   ├── data_pipeline.py        # Data loading, cleaning, reparameterization
│   ├── pca.py                  # PCA analysis and mode extraction
│   ├── heat_equation.py        # Analytical solutions, Fourier modes, comparison
│   ├── calibration.py          # Field theory parameter fitting (μ, κ)
│   ├── factor_model.py         # PCA-based risk factor model
│   └── plotting.py             # Consistent, publication-quality plot styling
│
├── notebooks/                  # Analysis notebooks (narrative + code)
│   ├── 01_data_generation.ipynb
│   ├── 02_reparameterization.ipynb
│   ├── 03_pca_analysis.ipynb
│   ├── 04_physics_connection.ipynb
│   ├── 05_field_theory_fit.ipynb
│   └── 06_factor_model.ipynb
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_simulate.py
│   ├── test_black_scholes.py
│   └── test_pca.py
│
├── data/                       # Data directory (not committed to git)
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
└── report/                     # Written report
    └── (report.pdf)
```

## Getting Started

### Prerequisites

- Python 3.10+
- PyCharm (recommended IDE)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/vol-surface-dynamics.git
cd vol-surface-dynamics

# Create a virtual environment (PyCharm can do this for you)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

### Quick Start

```python
from src.simulate import StochasticHeatEquation
from src.pca import run_pca, plot_eigenmodes

# Generate synthetic volatility surface data
spde = StochasticHeatEquation(D=0.05, n_z=50, z_range=(-3, 3))
data = spde.simulate(n_days=2000)

# Run PCA on the simulated returns
eigenvalues, eigenvectors = run_pca(data, n_components=10)

# Visualize eigenmodes
plot_eigenmodes(eigenvectors, spde.z_grid)
```

## Key Results

*(To be filled in as you complete each phase)*

- [ ] Phase 1: Synthetic data generation from stochastic heat equation
- [ ] Phase 2: Moneyness z and psychological time τ reparameterization
- [ ] Phase 3: PCA reveals sinusoidal eigenmodes, ~k⁻² eigenvalue decay
- [ ] Phase 4: Quantitative match between PCA modes and Laplacian eigenfunctions
- [ ] Phase 5: Field theory fit with μ, κ parameters
- [ ] Phase 6: Factor model for option portfolio risk

## References

- Ioselevich, P. — *A Data-Driven Factor Model for Option Risk*, Cambridge Quant Conference
- Le Coz, V. & Bouchaud, J.-P. — Elastic string models for forward interest rates
- Gatheral, J. — *The Volatility Surface: A Practitioner's Guide*
- Bouchaud, J.-P. & Potters, M. — *Theory of Financial Risk and Derivative Pricing*
- Hull, J. — *Options, Futures, and Other Derivatives*

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
