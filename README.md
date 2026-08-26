# Heat-Equation Dynamics in the Volatility Surface

**Testing a field-theory factor model against real options data.**

A physics / data-science / quantitative-finance research project that asks a single
honest question: *does the option volatility surface really move like a diffusive
membrane — and where does that picture break?*

Inspired by Pavel **Ioselevich** (Capital Fund Management), *A Data-Driven Factor Model
for Option Risk*, and building on the elastic-string models of **Le Coz & Bouchaud**.

---

## The idea in one paragraph

Option markets quote a 2D surface of implied volatilities across strikes and maturities.
If you reparameterize that surface into standard-deviation moneyness `z` and a compressed
"psychological time" `τ`, the daily co-movement of delta-hedged P&Ls looks suspiciously
like the dynamics of a **stochastic heat equation** — the same PDE that governs diffusion
in physics. This project builds the full pipeline (data → delta-hedged P&L → PCA →
eigenmodes → field-theory fit → factor model) and tests that hypothesis. The deliverable
is a small, validated factor model for option-portfolio risk together with an honest
account of where the diffusive picture holds and where it fails.

## The reframe (why this project is structured the way it is)

A naive version of this study has a fatal flaw: if you *generate* data from a stochastic
heat equation and then "discover" that it has the eigenstructure of a stochastic heat
equation, you have proved nothing — you put the answer in. A clean `λ_k ~ k⁻²` fit on
synthetic data is a **unit test of the simulator**, not a finding.

So the roles are inverted on purpose:

- **Synthetic data is the validation harness.** It exists only to prove the pipeline
  recovers known ground truth (sinusoidal modes, `λ_k ~ k⁻²`, the correct diffusion
  constant `D`). This belongs in *methodology*, labelled as exactly that.
- **Real options data is the actual study.** The research contribution is precisely
  *where real surfaces match the diffusive-membrane picture and where they deviate from
  it.* The deviations are the result, not a failure.

> *"The first three modes match the heat-equation prediction, but the eigenvalue tail
> departs from `k⁻²` because real skew has non-diffusive structure"* is a far stronger,
> more credible sentence than any clean fit on simulated data.

## Background you need (and where to get it)

You need working intuition, not mastery, in four areas:

| Area | The one-line version | Resource |
|------|----------------------|----------|
| **Options** | A call/put gives nonlinear exposure to the underlying; payoff `max(±(S−K), 0)`. | Hull, ch. 1, 10 |
| **Black–Scholes & implied vol** | Prices are quoted as the `σ` you'd plug into BS to reproduce them. | Hull ch. 13, 15; `py_vollib` |
| **The vol surface** | Implied vol `σ(K,T)` for every traded strike/maturity; it smiles/skews, and it *moves*. | Gatheral, *The Volatility Surface* |
| **Delta-hedged P&L** | Hedging away delta isolates vol/convexity exposure — but "continuously" is a fiction; the hedging frequency is a modelling choice that injects noise and bias. | (documented in `data_pipeline.py`) |
| **PCA** | Eigendecomposition of the return correlation matrix; eigenvectors are independent modes of variation, eigenvalues their variance share. | StatQuest / any linear-algebra text |
| **The heat equation** | `∂ₜu = D ∂²ₓu`; on a finite domain its eigenfunctions are sinusoidal with eigenvalues `~ k²`. | Bouchaud & Potters |

**The physics connection.** If vol-surface returns obey a stochastic heat equation

```
∂ₜ p(z,t) = D ∂²_z p(z,t) + ξ(z,t)            (ξ = white noise)
```

then decomposing in Fourier modes `p = Σ fₖ cos(kz)` makes each mode an
Ornstein–Uhlenbeck process `∂ₜfₖ = −D k² fₖ + ξₖ(t)` with stationary variance `~ 1/k²`.
The k^-2 law describes stationary variance for mode levels; PCA on daily changes sees the increment spectrum - 2Vₖ(1−e^(−γₖdt)) - flat for slow modes,
k^-2 for fast ones, and it is this corrected prediction the validation harness tests.

## Repository structure

```
diffusion-modes-vol-surface/
├── README.md                     # this file
├── LICENSE                       # MIT
├── requirements.txt
├── setup.py
│
├── src/                          # reusable library (tools, not the study)
│   ├── simulate.py               # stochastic heat-equation simulator (validation harness)
│   ├── black_scholes.py          # BS pricing, Greeks, IV inversion, delta-hedged P&L
│   ├── data_pipeline.py          # load real + synthetic data; (z, τ) reparameterization
│   ├── pca.py                    # PCA, rolling-window stability, factor projection
│   ├── heat_equation.py          # analytical Fourier modes, λ_k scaling, mode overlap
│   ├── calibration.py            # field-theory (κ, μ) calibration
│   ├── factor_model.py           # PCA-based portfolio risk model
│   └── plotting.py               # consistent, publication-quality figures
│
├── notebooks/                    # narrative: synthetic harness vs real data, compared
│   ├── 01_data.ipynb             # Phase 1 — validation harness, then real data
│   ├── 02_reparam.ipynb          # Phase 2 — (z, τ) reparameterization (with the √T fix)
│   ├── 03_pca.ipynb              # Phase 3 — PCA on both, side by side
│   ├── 04_physics.ipynb          # Phase 4 — eigenvalue scaling; where it deviates
│   ├── 05_field_theory.ipynb     # Phase 5 — fit (κ, μ); universality as a question
│   └── 06_factor_model.ipynb     # Phase 6 — factor model + out-of-sample test
│
├── tests/                        # unit tests, incl. the synthetic-recovery harness
│   ├── test_simulate.py
│   ├── test_black_scholes.py
│   ├── test_data_pipeline.py
│   └── test_pca.py
│
├── data/                         # data lives here (contents git-ignored)
│   ├── raw/                      # raw options data (real)
│   ├── processed/                # cleaned data in (z, τ) coordinates
│   └── synthetic/                # simulator output (the validation harness)
│
└── report/                       # written report (10–15 pages)
    └── README.md                 # report outline; report.pdf goes here
```

## Getting started

```bash
git clone https://github.com/michaelaafton/diffusion-modes-vol-surface.git
cd diffusion-modes-vol-surface

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\scripts\activate

pip install -r requirements.txt
pip install -e .
```

Run the validation harness (synthetic recovery) and the test suite:

```bash
pytest                           # unit tests, including synthetic recovery
jupyter lab notebooks/           # work through 01 → 06
```

```python
# The validation harness: simulate known ground truth, confirm the pipeline recovers it.
from src.simulate import generate_synthetic_dataset
from src.pca import run_pca

data = generate_synthetic_dataset(n_days=2000, D=0.05, n_z=50, seed=42)
pca = run_pca(data["returns"], n_components=10)
theory = spde.theoretical_increment_eigenvalues(10)
# PCA runs on DAILY CHANGES, so the correct ground truth is the OU
# increment spectrum Var(Δf_k) ∝ 2V_k(1 − exp(−γ_k dt)):
# ~flat for slow modes, → k⁻² for fast ones. See tests/test_simulate.py.
```

## Conventions (state them explicitly — they matter)

- **Moneyness** uses standard-deviation-to-expiry scaling:
  `z = log(K/S) / (σ·√T)`. The `√T` is *not* optional — implied vol `σ` is annualized, so
  without it `z` is not comparable across maturities and the whole `τ`-dimension analysis is
  distorted. (Equivalently: scale by total implied variance over the option's life.)
- **Psychological time** compresses the maturity axis: `τ = ψ·log(1 + T/ψ)` with
  `ψ ≈ 20–30` business days, matching the intuition that 1m-vs-2m feels larger than 1y-vs-2y.
- **Delta-hedged P&L** requires a chosen **hedging frequency**. This is a modelling decision
  that injects both noise and bias into every downstream number; it is documented, not buried.

## Design decisions
D1. Strike from impl_strike, not from inverting delta. Two roads to z: 
analytically invert the Black–Scholes delta formula, or trust the vendor's impl_strike. 
Take the vendor's. Reason: their inversion used the exact same model, rates, and dividend assumptions that produced the surface
— self-consistent by construction. Rolling your own risks a subtle mismatch that would masquerade as structure in your spectrum.

D2. Put/call stitching: keep OTM only. At every strike two options exist (put and call), but market information lives in the liquid one, 
which is the out-of-the-money one. The clean rule: keep rows with |delta| ≤ 0.50 — that keeps OTM puts (which populate z < 0, the crash side) and OTM calls (z > 0, the rally side), 
meeting at the money. One continuous moneyness axis, each half from its liquid representative. Make it a flag (otm_only=True) so the choice is visible, not buried.
The grid is 10–90 and the filter keeps 18 pillars per maturity.

D3. Calendar days → business days. Surface days are calendar days; your pipeline (and BDAYS_PER_YEAR = 252) thinks in business days. 
Convert: ttoexp = days × 252/365. Miss this and every √T is silently wrong by √(365/252) ≈ 1.20 — a 20% distortion of the z-axis that would corrupt the spectrum while looking perfectly plausible. 
This is the classic quiet-unit-bug of options work.

D4. The z-range shrinks — and that's a finding, not a bug. Here's something the repo's config doesn't know yet. 
Deltas ±0.20…±0.80 translate to z spanning roughly ±1.15 (a 0.20-delta option sits about 0.84 standard deviations OTM — the delta and the z-quantile are near-mirrors). 
But ReparamConfig defaults to z ∈ [−3, 3]! On real surface data most of that domain is empty. So: for real data, 
construct ReparamConfig(z_min=-1.15, z_max=1.10, n_z_bins=8)  
Physics consequence worth writing down now: the accessible z-window sets the longest wavelength — and hence which modes k you can resolve. 
The deep wings (crash tail) would need the raw per-option file (opprcd), a possible later extension. This is essentially the same as documenting an instrument's field of view.

D5. Panel variable: implied vol, differenced; not delta-hedged P&L. PCA input is Δσ(z) day-over-day. 
Rationale: (i) stationarity — levels are near-random-walk; (ii) matches the harness's validated increment theory; (iii) avoids importing a hedging_frequency assumption (see hedge_frequency note) into the measurement stage. 
Caveat: daily Δσ at fixed z differs from delta-hedged P&L by gamma–theta carry terms, which are approximately a rank-one contamination concentrated in the leading mode, so mode-1 interpretation carries an asterisk;
modes ≥ 2 are robust. Future robustness check: vega-weighted panel.

D6. Phase 3 results & specification. PCA on daily Δσ panel (2499×8, full coverage). 
Headline: mode 1 = level (91.5%, 0 nodes), mode 2 = skew see-saw (1 node), spectrum slope p = 2.99 vs diffusive prediction 2. 
Robustness (deep-put bin excluded): p = 3.20, modes 1–2 unchanged ⇒ stiffness anomaly is specification-stable. 
Mode 3 rejected as membrane harmonic: edge-following dipole, node count unstable (2→4) under bin removal. 
Corrections during analysis: (i) earlier "(2498, 7), 1 bin dropped" panel was an artifact of object-dtype null semantics — clean-dtype truth is full coverage; dtype now guaranteed at panel construction + pytest. 
(ii) Predicted smooth curvature mode 3 falsified by edge-tracking diagnostic. Node-counting evidence stops at mode 2 by design, not oversight.

D7. Spectral adjudication (Phase 4).
Candidate operators compared via matched-instrument sweeps: identical 8-bin grid on [−1.15, 1.10], 2500-day horizon, daily differencing, correlation-matrix PCA, and slope estimator for synthetic and real data. 5 seeds per cell.
 -Diffusion-only (κ = 0): falsified twice. Full-spectrum slope saturates at p ≤ 1.78 and tail slope at p ≤ 1.87 across the D sweep, vs real p = 2.99–3.20 (full) and 2.53–2.64 (tail).
 -Single k⁴ operator: slope-sufficient (κ ≈ 0.2–0.3 reproduces full p ≈ 3) but profile-falsified: synthetic spectra are concave on log-log (p_tail > p_full), real is convex (p_tail < p_full); mode-2 share off by 6×.
 -Composite = external level factor + membrane (D = 0.2, κ ≈ 0.02, k* ≈ 3, crossover inside the observable band): reproduces the 8-point profile. Mode-1 share 0.9182 ± 0.0022 vs real 0.9150 (+1.5σ); rank-7 degeneracy filled, mode 8 = 0.0012 ± 0.0001 vs real 0.0014, untuned agreement. 
  Residuals (z-scores vs seed spread): mode-2 deficit −7.8σ; mode-3–4 excess +31σ / +9.6σ; missing mode-4–5 shelf (real 0.0067 -> 0.0064 flat; model cascades), mode 5 −10.5σ; modes 6–8 mildly starved (−4 to −8σ), consistent with tail slightly too steep (synthetic p_tail 2.72 ± 0.02 vs real 2.53–2.64), κ marginally high, deferred to formal calibration (Phase 5).
  However, with block-bootstrap errors on the real spectrum, residual significance shrinks to mode-4 +3.2σ and the 4–5 shelf; mode-3 excess not significant.
 -Caveats: shares sum to 1, so residual z-scores are not independent; the denominator is synthetic seed-spread only (real-spectrum estimation error not yet included), so significance is overstated. Establishes sufficiency, not uniqueness, other mechanisms (fat-tailed noise, non-stationarity) could reproduce a finite-resolution spectrum.
Methodological notes
 -Aliasing: simulated modes matched to grid resolution (n_modes = n_z); coarse sampling of extra modes would fold high-k variance into low modes and spuriously steepen slopes.
 -Underflow guard: re-derived rather than raised, exp underflow (a→0) is the benign exact-update limit (mode redraws from its stationary law); guard replaced with a finiteness assert.
 -Mode-1 inflation: the level factor inflates apparent stiffness by ~10× (full-slope match demanded κ ≈ 0.2–0.3; tail-only demands κ ≈ 0.02), motivating tail-slope analysis as the operator-sensitive statistic.
 -Prediction ledger: modes 6–8 predicted within ±2σ, observed −4 to −8σ (explained by global tail steepness); 4–5 shelf predicted missing, confirmed.

D8. Formal calibration (Phase 5). Analytic forward model: exact increment-covariance Φᵀdiag(2Vₖ(1−e^(−γₖdt)))Φ + m, correlation-normalised, eigendecomposed: validated against simulation to within seed noise. 
Real-spectrum uncertainty via moving-block bootstrap (block 25 d, 1000 replicates); with these errors, D7's seed-only z-scores collapse (mode-3 "+31σ" → +1.9σ, not significant). Fit: weighted χ² on modes 1–7 (mode 8 sum-determined), log-parameters, Nelder–Mead; errors by bootstrap-refit. 
Result: D = 0.69 [0.29, 1.25], κ = 0.011 [0.006, 0.020], level kick σ = 2.64 [2.25, 3.35], k* = 7.9 [5.1, 11.2] - crossover at the top of the observable band. Hand-fit (D = 0.2, κ = 0.02) rejected: Δχ² = 17.1, D was never fitted by hand, and the optimiser traded 3.5× more diffusion against half the bending. 
corr(log D, log κ) = +0.32: parameters separately identified, CIs honestly wide. χ²/dof = 3.6; sole significant residual: mode-5 deficit z = −3.0 - the real 4-5 shelf, unreproducible by any smooth level+operator composite; leading candidate for future work (mode 8 z = +2.3, unfitted, noted). 
Prediction ledger: flat-valley hypothesis rejected by the data; predicted D–κ degeneracy (+0.6–0.9) not observed; predicted surviving mode-4 residual absorbed by the fit.
## Data sources

Deep historical full-surface options data is expensive. Viable routes, in order of
preference:

1. **WRDS / OptionMetrics (IvyDB)** — the standard academic source for historical equity
   implied vols and Greeks; free if your institution subscribes. Cleanest path to real
   equity surfaces.
2. **Deribit BTC/ETH options** — free public API, several years of liquid crypto-option
   history. Younger and weirder than equity index, which makes the "where does the membrane
   break?" question *more* interesting.
3. **yfinance** — current chains only, *not* deep history; fine for a snapshot sanity check,
   insufficient to drive the time-series PCA at the core of this project.

## Roadmap

| Weeks | Phase | What happens | Key module |
|------:|-------|--------------|------------|
| 1–2 | Prep | Options, implied vol, PCA, heat equation, delta-hedging mechanics | — |
| 3 | **1** | Synthetic validation harness; secure + load a real data source | `simulate.py`, `data_pipeline.py` |
| 3–4 | **2** | `(z, τ)` reparameterization with the √T fix; document conventions | `data_pipeline.py` |
| 4–5 | **3** | Identical PCA on synthetic and real; reproduce + compare key plots | `pca.py` |
| 5–7 | **4** | Eigenvalue scaling `λₖ ~ k⁻ᵅ`, Fourier overlap — *quantify the deviation* | `heat_equation.py` |
| 7–8 | **5** | Field-theory calibration `(κ, μ)`; test universality across assets | `calibration.py` |
| 8–9 | **6** | Factor model + out-of-sample validation on real data vs an SVI baseline | `factor_model.py` |
| 9–10 | Polish | Report, code cleanup, README | — |

## Status / key results

Results are filled in honestly as each phase completes. The synthetic recovery is a
*validation step*; the headline results come from real data, deviations included.

- [ ] **Phase 1** — Synthetic harness recovers ground truth; real data acquired and loaded
- [ ] **Phase 2** — `(z, τ)` reparameterization with documented conventions
- [ ] **Phase 3** — PCA eigenmodes & spectra, synthetic vs real, side by side
- [ ] **Phase 4** — Power-law exponent `α` for both sources; where real surfaces leave `k⁻²`
- [ ] **Phase 5** — `(κ, μ)` fit with confidence intervals; the universality question
- [ ] **Phase 6** — Out-of-sample factor-model risk vs realised, vs SVI baseline

## References

- Ioselevich, P. — *A Data-Driven Factor Model for Option Risk*.
- Le Coz, V. & Bouchaud, J.-P. — Elastic-string models for forward rates.
- Gatheral, J. — *The Volatility Surface: A Practitioner's Guide*.
- Bouchaud, J.-P. & Potters, M. — *Theory of Financial Risk and Derivative Pricing*.
- Hull, J. — *Options, Futures, and Other Derivatives*.

## License

MIT — see [LICENSE](LICENSE).
