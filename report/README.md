# Report

The written report (10–15 pages) lives here as `report.pdf`.

## Outline

1. **Introduction** — why volatility-surface dynamics matter.
2. **Background** — options, implied vol, PCA, the heat equation.
3. **Methodology** — the `(z, τ)` reparameterization (with the √T scaling); PCA;
   and the *synthetic-harness-vs-real-data* design. The synthetic recovery appears
   here as a **validation step**, framed as exactly that — not as a result.
4. **Results** — eigenmodes, eigenvalue scaling, and Fourier comparison, shown
   **synthetic vs real, side by side**.
5. **Field-theory fit and the universality question** — fitted `(κ, μ)` with
   confidence intervals; cross-asset consistency reported honestly as an open question.
6. **Factor model and out-of-sample performance** — predicted vs realised risk on
   real data, benchmarked against an SVI baseline.
7. **Discussion** — what holds, what breaks, and extensions.

## Framing

**Lead the abstract and discussion with the real-data result and its deviations —
not the synthetic fit.** "The first three modes match the heat-equation prediction
but the eigenvalue tail departs from `k⁻²`" is the kind of honest empirical claim
this report is built to make.

## Cite

Ioselevich; Bouchaud & Potters; Le Coz & Bouchaud; Gatheral; Hull.
