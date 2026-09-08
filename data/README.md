# Heat-Equation Dynamics in the Volatility Surface

**Testing a field-theory factor model against real options data.**

A physics / data-science / quantitative-finance research project that asks a single
honest question: *does the option volatility surface really move like a diffusive
membrane — and where does that picture break?*

Inspired by an unpublished presentation by Pavel **Ioselevich** (Capital Fund Management),
*A Data-Driven Factor Model for Option Risk*, and building on the elastic-string models of
**Le Coz & Bouchaud**.

**Full write-up:** [`report/report.pdf`](report/report.pdf)

---

## Key results (SPX, 2015–2024, OptionMetrics IvyDB)

- **Pure diffusion is falsified.** Through a matched instrument (identical grid, horizon,
  differencing and estimator for synthetic and real data), no diffusion constant reaches the
  observed spectrum: synthetic slopes saturate at `p ≤ 1.87` vs real tail slopes of `2.53–2.64`.
- **A 3-parameter composite works:** an external volatility-level factor (91.5% of daily-change
  variance) riding on a membrane with tension *and* bending rigidity, `γₖ = Dk² + κk⁴`.
  Calibrated with block-bootstrap errors: `D = 0.69 [0.29, 1.25]`, `κ = 0.011 [0.006, 0.020]`,
  crossover `k* = 7.9 [5.1, 11.2]` at the top of the observable band (`χ²/dof = 3.6`).
- **Out of sample (train ≤ 2022-01-05, test 2022–2024):** the 3-parameter field theory predicts
  portfolio risk within **1%** of a 36-number empirical factor model on level, straddle and
  risk-reversal exposures. A butterfly portfolio — loading exactly modes 3–5 — exposes the
  model's one significant spectral residual (the mode-4–5 "shelf") in money terms, over-predicted 2.2×.
- **The honest part:** the shelf is measured twice by independent routes (spectral fit `z = −3.0`;
  butterfly ratio) and reported as the leading open feature, not smoothed over.

## The idea in one paragraph

Option markets quote a 2D surface of implied volatilities across strikes and maturities.
Reparameterized into standard-deviation moneyness `z`, the daily co-movement of implied-vol
changes `Δσ(z)` looks suspiciously like the dynamics of a **stochastic heat equation** — the
same PDE that governs diffusion in physics. This project builds the full pipeline
(data → (z, τ) panel → PCA → eigenmodes → field-theory calibration → factor model) and tests
that hypothesis. The deliverable is a small, validated factor model for option-portfolio risk
together with an honest account of where the diffusive picture holds and where it fails.

## The reframe (why this project is structured the way it is)

A naive version of this study has a fatal flaw: if you *generate* data from a stochastic
heat equation and then "discover" that it has the eigenstructure of a stochastic heat
equation, you have proved nothing — you put the answer in. A clean fit on synthetic data is
a **unit test of the simulator**, not a finding.

So the roles are inverted on purpose:

- **Synthetic data is the validation harness.** It exists only to prove the pipeline
  recovers known ground truth — and the correct ground truth for PCA on *daily changes* is
  the OU increment spectrum `2Vₖ(1−e^(−γₖdt))`, not the naive `k⁻²` level law (see below).
  This belongs in *methodology*, labelled as exactly that.
- **Real options data is the actual study.** The research contribution is precisely
  *where real surfaces match the diffusive-membrane picture and where they deviate from
  it.* The deviations are the result, not a failure.

> *"The level and skew modes are membrane-like, but the spectrum falls steeper than
> diffusion allows — and the excess is quantitatively consistent with bending rigidity"*
> is a far stronger, more credible sentence than any clean fit on simulated data.

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

**The physics connection.** If vol-surface perturbations obey a stochastic heat equation

```
∂ₜ p(z,t) = D ∂²_z p(z,t) + ξ(z,t)            (ξ = white noise)
```

then decomposing in Fourier modes `p = Σ fₖ cos(kz)` makes each mode an
Ornstein–Uhlenbeck process `∂ₜfₖ = −D k² fₖ + ξₖ(t)` with stationary variance `~ 1/k²`.
**But** the `k⁻²` law describes the stationary variance of mode *levels*; PCA on daily
*changes* sees the increment spectrum `2Vₖ(1−e^(−γₖdt))` — flat for slow modes, `k⁻²` for
fast ones — and it is this corrected prediction the validation harness tests. Getting this
distinction right materially changes the conclusions; see `tests/test_simulate.py` and
Sec. 2 of the report.

## Repository structure

```
diffusion-modes-vol-surface/
├── README.md                     # this file
├── LICENSE                       # MIT
├── requirements.txt
├── setup.py
│
├── src/                          # reusable library (tools, not the study)
│   ├── simulate.py               # SPDE simulator: exact OU updates, D k² + κ k⁴ operator
│   ├── black_scholes.py          # BS pricing, Greeks, IV inversion, delta-hedged P&L
│   ├── data_pipeline.py          # WRDS/IvyDB loader; (z, τ) reparameterization; panel builder
│   ├── pca.py                    # PCA, slope estimators, factor projection
│   ├── heat_equation.py          # analytical Fourier modes, λ_k scaling, mode overlap
│   ├── calibration.py            # analytic forward model + weighted χ² fit of (D, κ, m)
│   ├── factor_model.py           # covariance models + out-of-sample portfolio risk test
│   └── plotting.py               # consistent figures
│
├── scripts/                      # the study itself, in execution order
│   ├── pull_wrds.py              # one-time WRDS pull (SPX secid 108105, 2015–2024)
│   ├── build_panel.py            # day × z-bin implied-vol panel
│   ├── run_pca_real.py           # Phase 3: empirical spectrum, slopes, mode shapes
│   ├── adjudicate_p.py           # Phase 4: matched-instrument operator sweeps
│   ├── fingerprint_composite.py  # Phase 4: composite model vs real 8-point spectrum
│   ├── bootstrap_real_spectrum.py# Phase 5: block-bootstrap errors on the real spectrum
│   ├── fit_real_spectrum.py      # Phase 5: formal calibration with bootstrap-refit CIs
│   ├── oos_risk_test.py          # Phase 6: out-of-sample portfolio risk test
│   └── make_figures.py           # report figures
│
├── tests/                        # unit tests, incl. the synthetic-recovery harness
│   ├── test_simulate.py
│   ├── test_black_scholes.py
│   ├── test_data_pipeline.py
│   └── test_pca.py
│
├── data/                         # data lives here (contents git-ignored — licensed)
│   ├── raw/                      # raw options data (real)
│   ├── processed/                # bootstrap outputs, processed panels
│   └── synthetic/                # simulator output (the validation harness)
│
└── report/                       # the written paper
    ├── report.pdf                # compiled report (10 pages)
    ├── report.tex                # LaTeX source
    └── figures/                  # spectrum, mode shapes, OOS ratios
```

## Getting started

```bash
git clone https://github.com/michaelaafton/diffusion-modes-vol-surface.git
cd diffusion-modes-vol-surface

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Run the test suite (includes the synthetic-recovery harness):

```bash
pytest
```

The validation harness in five lines — simulate known ground truth, confirm the pipeline
recovers the *correct* (increment-law) spectrum:

```python
import numpy as np
from src.simulate import StochasticHeatEquation, SPDEConfig
from src.pca import run_pca

spde = StochasticHeatEquation(SPDEConfig(D=0.05, n_z=50, seed=42))
surface = spde.simulate(20_000)
pca = run_pca(np.diff(surface, axis=0), n_components=10)   # PCA on DAILY CHANGES
theory = spde.theoretical_increment_eigenvalues(10)         # 2Vₖ(1 − exp(−γₖ dt))
# empirical and theory agree to <0.2 pp per mode; frozen in tests/test_simulate.py
```

Real-data pipeline: run `scripts/pull_wrds.py` once (requires WRDS credentials; raw data
lands in `data/raw/`, git-ignored under licence), then the scripts in the order listed above.

## Conventions (state them explicitly — they matter)

- **Moneyness** uses standard-deviation-to-expiry scaling with an ATM ruler:
  `z = log(K/S) / (σ_ATM·√T)`. The `√T` is *not* optional — implied vol is annualized, so
  without it `z` is not comparable across maturities. The *ATM* vol (not each option's own
  vol) is essential: normalising by own-vol makes the coordinate self-referential (D4).
- **Psychological time** compresses the maturity axis: `τ = ψ·log(1 + T/ψ)` with
  `ψ ≈ 20–30` business days. (This study uses the 30-calendar-day slice; the τ-dimension
  is deferred future work.)
- **Delta-hedged P&L** requires a chosen **hedging frequency**. This is a modelling decision
  that injects both noise and bias into every downstream number; the study side-steps it by
  measuring `Δσ` directly (D5), and documents the resulting gamma–theta caveat.

## Design decisions (D-log)

**D1. Strike from vendor `impl_strike`, not from inverting delta.** The vendor's inversion
used the same model, rates and dividend assumptions that produced the surface —
self-consistent by construction. Rolling your own risks a subtle mismatch that would
masquerade as structure in the spectrum.

**D2. Put/call stitching: keep OTM only.** At every strike two options exist, but market
information lives in the liquid, out-of-the-money one. Rule: keep `|delta| ≤ 0.50` — OTM
puts populate `z < 0` (crash side), OTM calls `z > 0` (rally side), meeting at the money.
One continuous moneyness axis, each half from its liquid representative; exposed as a flag
(`otm_only=True`) so the choice is visible. The pillar grid runs `|Δ| ∈ [0.10, 0.90]` in
steps of 0.05; the OTM filter keeps 9 pillars per side (18 per maturity).

**D3. Calendar days → business days.** Surface maturities are calendar days; the pipeline
(and `BDAYS_PER_YEAR = 252`) thinks in business days. Convert: `ttoexp = days × 252/365`.
Miss this and every `√T` is silently wrong by `√(365/252) ≈ 1.20` — a ~17% compression of
the z-axis that corrupts the spectrum while looking perfectly plausible.

**D4. The usable z-window is an instrument property: `[−1.15, 1.10]`, 8 bins.** The IvyDB
delta-pillar grid, with z normalised by the per-(date, maturity) σ_ATM (wing vols exceed
ATM vol, stretching the put side), maps to this asymmetric window; 13.3% of raw rows
(nearly all deep OTM puts) fall outside it. Pillar spacing supports ≈8 uniform bins —
finer binning would manufacture resolution the instrument does not possess. The accessible
window sets which modes k are resolvable (k ≲ 8); the deep crash wing would need the raw
per-option file (`opprcd`), a possible extension. This is documenting an instrument's
field of view.

**D5. Panel variable: implied vol, differenced — not delta-hedged P&L.** PCA input is
`Δσ(z)` day-over-day. Rationale: (i) stationarity — levels are near-random-walk;
(ii) matches the harness-validated increment theory; (iii) avoids importing a
hedging-frequency assumption into the measurement stage. Caveat: daily `Δσ` differs from
delta-hedged P&L by gamma–theta carry, approximately a rank-one contamination concentrated
in the leading mode — mode-1 interpretation carries an asterisk; modes ≥ 2 are robust.
Future robustness check: vega-weighted panel.

**D6. Phase-3 results & specification.** PCA on the daily `Δσ` panel (2499×8, full
coverage). Headline: mode 1 = level (91.5%, 0 nodes), mode 2 = skew see-saw (1 node);
spectrum slopes p = 2.99 (8-bin) / 3.20 (edge bin excluded), tail slopes 2.53 / 2.64.
Mode 3 rejected as a membrane harmonic: edge-following dipole, node count unstable (2→4)
under bin removal. Corrections during analysis: (i) an earlier "(2498, 7), 1 bin dropped"
panel was an artifact of object-dtype null semantics — clean-dtype truth is full coverage;
dtypes now guaranteed at panel construction + pytest. (ii) A predicted smooth curvature
mode 3 was falsified by the edge-tracking diagnostic. Node-counting evidence stops at
mode 2 by diagnosis, not oversight.

**D7. Spectral adjudication (Phase 4).** Candidate operators compared via
matched-instrument sweeps (identical 8-bin grid on [−1.15, 1.10], 2500-day horizon, daily
differencing, correlation-matrix PCA and slope estimator; 5 seeds/cell).
Diffusion-only (κ=0): falsified twice — slopes saturate at p ≤ 1.78 (full) and ≤ 1.87
(tail) vs real 2.99–3.20 / 2.53–2.64. Single k⁴ operator: slope-sufficient but
profile-falsified — synthetic spectra are concave on log-log (p_tail > p_full), real is
convex; mode-2 share off 6×. Composite (external level factor + membrane): reproduces the
8-point profile, including the untuned mode-8 agreement (level factor fills the rank-7
sampling degeneracy: 0.0012 ± 0.0001 vs real 0.0014). With block-bootstrap errors on the
real spectrum, residual significance shrinks to the mode-4–5 shelf region; the seed-only
"+31σ mode-3 excess" collapses to +1.9σ (not significant) — *seed spread is not
measurement error*. Methodological notes: aliasing control (`n_modes = n_z`); underflow
guard re-derived (exp underflow is the benign exact-update limit); mode-1 dominance
inflates apparent stiffness ~10× (full-slope match demanded κ ≈ 0.2–0.3; tail-only ≈ 0.02),
motivating tail-slope analysis as the operator-sensitive statistic.

**D8. Formal calibration (Phase 5).** Analytic forward model — exact increment covariance
`Φᵀ diag(2Vₖ(1−e^(−γₖdt))) Φ + m`, correlation-normalised, eigendecomposed — validated
against simulation to within seed noise. Real-spectrum uncertainty via moving-block
bootstrap (block 25 d, 1000 replicates). Fit: weighted χ² on modes 1–7 (mode 8
sum-determined), log-parameters, Nelder–Mead; CIs by bootstrap-refit. **Result:
D = 0.69 [0.29, 1.25], κ = 0.011 [0.006, 0.020], level kick √m = 2.64 [2.25, 3.35],
k\* = 7.9 [5.1, 11.2]** — crossover at the top of the observable band. Hand-fit
(D=0.2, κ=0.02) rejected: Δχ² = 17.1 (D was never fitted by hand). corr(log D, log κ) =
+0.32: parameters separately identified, CIs honestly wide. χ²/dof = 3.6; sole significant
residual: mode-5 deficit z = −3.0 — the real 4–5 shelf, unreproducible by any smooth
level+operator composite (mode-8 z = +2.3, unfitted, noted). Prediction ledger:
flat-valley hypothesis rejected; predicted strong D–κ degeneracy not observed; predicted
surviving mode-4 residual absorbed by the fit.

**D9. Out-of-sample factor-risk test (Phase 6).** Chronological split: train 2015-01-05 →
2022-01-05 (1748 changes, COVID in-sample), test 2022-01-06 → 2024-12-31 (750). Predicted
vs realised variance of four vega-sketch portfolios; all structured models share train
per-bin stds, so the contest is correlation structure only. Results: (i) common scale
factor ~0.44–0.65 (train contains COVID-2020, test calmer; dynamic rescaling = future
work); (ii) diagonal baseline fails two-sided — 3.1× under on level, 6–14× over on
spreads; (iii) **field theory (3 parameters) matches the empirical 3-factor model
(36 numbers) to ≤1%** on level, straddle and risk-reversal, insensitive to factor count
(1-factor fails the risk reversal — mode 2 carries real skew risk); (iv) butterfly:
field theory over-predicts 2.2× — the portfolio loads precisely modes 3–5, independently
rediscovering the D8 spectral shelf in money terms. Prediction ledger: "all ratios > 1"
wrong (COVID-in-train); "equal-weight insensitive to structure" wrong (most sensitive);
spread-direction diagonal failure and field-empirical agreement right (agreement predicted
at 20%, delivered at 1%).

## Data sources

Deep historical full-surface options data is expensive. This study uses route 1:

1. **WRDS / OptionMetrics (IvyDB)** — the standard academic source for historical equity
   implied vols and Greeks; free if your institution subscribes. Used here: SPX smoothed
   surfaces (`vsurfd`) + spot (`secprd`), secid 108105, 2015–2024, pulled once to Parquet.
2. **Deribit BTC/ETH options** — free public API; a future cross-asset test of the
   "where does the membrane break?" question.
3. **yfinance** — current chains only; insufficient for time-series PCA.

## Project history

| Phase | What happened | Key module |
|------:|---------------|------------|
| **1** | Synthetic validation harness (exact OU; increment-law correction frozen as pytest); WRDS data secured and loaded | `simulate.py`, `data_pipeline.py` |
| **2** | `(z, τ)` reparameterization with σ_ATM ruler and documented conventions; day × z-bin panel | `data_pipeline.py` |
| **3** | PCA on real data: level + skew modes, mode-3 edge artifact diagnosed, slope anomaly found | `pca.py` |
| **4** | Matched-instrument adjudication: diffusion falsified; level-factor + bending composite sufficient | `simulate.py` (κk⁴ operator) |
| **5** | Analytic forward model; block-bootstrap errors; formal (D, κ, m) fit with CIs | `calibration.py` |
| **6** | Out-of-sample portfolio risk vs empirical-factor and diagonal baselines | `factor_model.py` |

**Deferred (honest list):** rolling-window mode stability; the τ-dimension / 2D membrane;
vega-weighted panel robustness; dynamic variance rescaling; deep wings via `opprcd`;
cross-asset universality as a question, not a claim.

## Status / key results

- [x] **Phase 1** — Harness recovers the (corrected) increment-law ground truth; SPX data loaded
- [x] **Phase 2** — `(z, τ)` panel: 2499 days × 8 bins, full coverage, decisions D1–D5 logged
- [x] **Phase 3** — Modes 1–2 membrane-like; mode 3 = edge artifact; tail slope 2.53–2.64
- [x] **Phase 4** — Diffusion falsified (p ≤ 1.87); composite level + `Dk²+κk⁴` reproduces the spectrum
- [x] **Phase 5** — D = 0.69 [0.29, 1.25], κ = 0.011 [0.006, 0.020], k\* = 7.9 [5.1, 11.2]; χ²/dof = 3.6
- [x] **Phase 6** — Field theory within 1% of empirical factor model OOS; butterfly localises the shelf

## References

- Bouchaud, J.-P. and Potters, M. (2003) *Theory of financial risk and derivative pricing*. 2nd edn. Cambridge: Cambridge University Press.
- Cont, R. and da Fonseca, J. (2002) 'Dynamics of implied volatility surfaces', *Quantitative Finance*, 2(1), pp. 45–60.
- Gatheral, J. (2006) *The volatility surface: a practitioner's guide*. Hoboken, NJ: John Wiley & Sons.
- Le Coz, V. and Bouchaud, J.-P. (2024) 'Revisiting elastic string models of forward interest rates', *Quantitative Finance*, 24, pp. 1561–1578.
- Ioselevich, P. — *A Data-Driven Factor Model for Option Risk* (unpublished presentation, Capital Fund Management).
- Hull, J. — *Options, Futures, and Other Derivatives* (background).

## License

MIT — see [LICENSE](LICENSE).
