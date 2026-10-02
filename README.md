# Spectral structure of the 30 day SPX volatility smile

A measurement study on OptionMetrics implied volatility surfaces for the S&P 500 index, 2015 to 2024.
Full report: [`report/report.pdf`](report/report.pdf)

This project started as a test of a physical picture: that the implied volatility smile moves like an elastic membrane, so the principal components of its daily changes should look like the modes of a diffusion equation. Testing that picture carefully ended up being mostly a study of the measuring instrument. The surfaces I use are smoothed by the data vendor before I ever see them, and the smoothing shapes the spectrum strongly enough that the membrane question cannot be settled with this data. The repository contains the pipeline, the tests that led to that conclusion, and a description of the experiment that would settle it.

## Results

All intervals are 95% moving block bootstrap intervals (blocks of 25 days, 1000 replicates).

* **Factor structure.** Over 2498 trading days, a level factor carries 94.4% of the variance of daily implied volatility changes (93.6 to 95.2) and a second factor carries 4.7% (3.9 to 5.5). The other five modes fall away quickly. A power law fitted to modes 2 to 7 has exponent p = 4.44 (4.18 to 4.67).

* **Unsmoothed diffusion is ruled out.** Simulated diffusion plus a level factor, put through exactly the same panel construction, never produces a tail steeper than p = 2.63 anywhere on the parameter grid.

* **The vendor's smoothing reproduces the slope of the tail but not its shape.** OptionMetrics builds these surfaces with a Gaussian kernel in delta and maturity. When simulated diffusion is passed through that kernel, the tail exponent lands between 4.2 and 4.5, inside the interval measured on real data. The shape is wrong, though. The smoothed spectra bend downward, with local slopes rising from below 2 to about 12 between modes 2 and 7, while the real spectrum stays close to a straight line on log scales (local slopes between 3.7 and 5.0). Figure 1 of the report shows the comparison.

* **The second factor is partly tied to the call/put seam.** The vendor smooths calls and puts separately, and in this panel the put side of the smile comes from puts and the call side from calls. The second factor's loadings jump at the money. Its daily returns have R² = 0.57 (0.43 to 0.68) against the daily change in the gap between the ATM call and ATM put volatilities, a gap that would stay at zero for a single consistent surface. Under the rule I fixed before running this test the result counts as ambiguous, since the interval contains 0.5, but somewhere between 43% and 68% of this factor's variance moves with the seam.

* **Negative autocorrelation over a few days.** Factor returns for modes 2, 4, 5 and 7 are negatively autocorrelated at lags 2 to 5, which measurement noise on its own does not produce. I could not identify the source. Both the call/put seam and pillars drifting between bins are plausible candidates.

* **What would settle the membrane question.** Rebuilding the 30 day slice from raw option quotes (OptionMetrics `opprcd`) as one surface that respects put/call parity would remove both the kernel and the seam. That is the natural next step for this work.

## How the tests were run

There are three tests: the kernel test (`scripts/test49_vendor_kernel.py`), and the seam and relaxation tests (`scripts/test47_48.py`). The numbers in the file names are the numbers of the items in the review of the earlier version that proposed them. Each script states its decision rules in its docstring, and the rules were written down before the tests were run. For the seam and relaxation tests the commit history confirms this, and the rule that decided which direction the paper would take was also committed before their results were known. The kernel test's rules were committed together with later follow up analyses, so for that test the history cannot confirm the order.

One rule had to change after its results existed: the validity check in the kernel test was derived from an older version of the instrument and failed for that reason. The script still prints the original verdict and then the amended check, labelled as such. The report gives the original rule, why it no longer applied, and the amended rule, in that order. The conclusions do not depend on the amended check.

## Corrections to the first version

An earlier version of this project reported a fitted membrane with bending stiffness, a feature in modes 4 and 5 of the spectrum, and an out of sample risk comparison. A review of that version found that the analysis scripts averaged implied volatility across all eleven vendor maturities instead of using the 30 day slice, because the maturity filter was present in one script and missing from the others. The averaged panel produced those results. I had also attributed a dropped bin to a data type problem; the real cause was the same missing filter. The fits and everything built on them have been removed, and the panel is now built in one function (`build_study_panel`) that checks it contains a single maturity. The report lists every correction.

## Data

SPX (OptionMetrics secid 108105), January 2015 to December 2024, from WRDS:

* `optionm.vsurfd`: the vendor's standardised volatility surface (pillars at fixed delta and maturity)
* `optionm.secprd`: daily closing level of the index

The data is licensed and is not included. Running `scripts/pull_wrds.py` needs a WRDS account with OptionMetrics access and saves both tables to `data/raw/`.

## Method in brief

* **Moneyness.** Each pillar is placed at z = log(K/S) / (σ_ATM √T), where σ_ATM is the average of the 50 delta call and put volatilities on that day and T is the maturity in years. The strike is the vendor's own `impl_strike`.
* **Panel.** Only the 30 calendar day pillars are used, and only out of the money ones (|delta| ≤ 0.50), so puts cover z < 0 and calls cover z > 0. Pillars are averaged within 8 equal bins on z in [−1.15, 1.10]. Bins filled on fewer than 95% of days are dropped (the deepest put bin, filled on 89.1% of days), and then days with any gap are dropped. The result is 2498 days by 7 bins.
* **PCA.** Eigendecomposition of the correlation matrix of daily changes in implied volatility.
* **Vendor kernel.** Per the IvyDB reference manual (also quoted by Avellaneda et al. 2020), each pillar is a vega weighted average of option volatilities with a Gaussian weight in log maturity and call equivalent delta (bandwidths h₁ = 0.05 and h₂ = 0.005). A third bandwidth h₃ = 0.001 on the call/put indicator means calls and puts are smoothed independently. `scripts/test49_vendor_kernel.py` reproduces this kernel on simulated quotes.

## Reproducing the results

```bash
git clone https://github.com/MichaelAafton/diffusion-modes-vol-surface.git
cd diffusion-modes-vol-surface
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Then, from the repository root:

```bash
python -m scripts.pull_wrds                 # once, needs WRDS access
python -m scripts.build_panel               # panel, spectrum, mode loadings
python -m scripts.bootstrap_real_spectrum   # bootstrap intervals (a few minutes)
python -m scripts.test49_vendor_kernel      # vendor kernel test (about 10 minutes)
python -m scripts.test49_followups          # ceiling decomposition and shape comparison (about 3 minutes)
python -m scripts.test47_48                 # seam and relaxation tests
python -m scripts.make_figures              # Figure 1 of the report
```

### Where each number comes from

| Numbers | Script |
|---|---|
| Panel size, coverage, dropped bin, bin centres, variance shares, p = 4.44, local slopes, mode loadings | `build_panel` |
| Intervals on the shares, on p, on the local slopes and on the tail shares | `bootstrap_real_spectrum` |
| Diffusion ceiling 2.63, smoothed diffusion exponents and their seed spread, both validity checks | `test49_vendor_kernel` |
| Decomposition of the ceiling, shape comparison between real and smoothed spectra | `test49_followups` |
| Seam R², autocorrelations and the outcome of the decision rule | `test47_48` |
| Figure 1 | `make_figures` |
| Kernel bandwidths | IvyDB US Reference Manual v7.0 (quoted, not computed) |

## Project structure

```
diffusion-modes-vol-surface/
├── data/
│   ├── processed/                spectra and bootstrap output (not tracked)
│   └── raw/                      downloaded data (not tracked)
├── report/
│   ├── figures/
│   │   └── fig1_spectrum.pdf
│   ├── report.pdf
│   └── report.tex
├── scripts/
│   ├── bootstrap_real_spectrum.py
│   ├── build_panel.py            study panel and spectrum
│   ├── make_figures.py           Figure 1
│   ├── pull_wrds.py              one time download from WRDS
│   ├── test47_48.py              seam and relaxation tests
│   ├── test49_followups.py       ceiling decomposition, shape comparison
│   ├── test49_vendor_kernel.py   vendor kernel test
│   └── visualize_surface.py      optional animation of the smile
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          loading, moneyness, build_study_panel
│   ├── forward_model.py          analytic spectrum of the membrane model
│   ├── pca.py                    correlation PCA, factor returns, tail slope
│   └── simulate.py               membrane simulator used by the tests
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_pca.py
│   └── test_simulate.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Use of AI tools

I used AI assistants (mainly Claude, through chat and Claude Code) throughout this project. They wrote a large share of the code from my specifications, including the first version of the library and most of the test scripts, and they helped me learn the methods I was using. I also used an AI model as a critical reviewer of the first version, and that review found the errors described above. The research question, the choice of data, the decisions about what to test and what to claim, and the interpretation of the results are mine. I reran every script listed above on my own machine, and every number in this README and in the report comes from those runs.

## Acknowledgements

This project was inspired by a conference talk by P. Ioselevich on factor models for option risk. The analysis here is independent and differs substantially from that work.

## References

* Avellaneda, M., Healy, B., Papanicolaou, A. and Papanicolaou, G. (2020). PCA for implied volatility surfaces. arXiv:2002.00085.
* Cont, R. and da Fonseca, J. (2002). Dynamics of implied volatility surfaces. *Quantitative Finance* 2(1), 45 to 60.
* Le Coz, V. and Bouchaud, J.P. (2024). Revisiting elastic string models of forward interest rates. *Quantitative Finance* 24, 1561 to 1578.
* OptionMetrics. *IvyDB US Reference Manual*, version 7.0.

## License

MIT, see [LICENSE](LICENSE).
