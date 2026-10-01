"""Tests 47 (seam) and 48 (relaxation) on the study panel.

PRE-REGISTRATION (fixed before any run on real data; sign-off: supervisor)
--------------------------------------------------------------------------
Common inputs. Study panel (build_study_panel): daily changes, correlation PCA,
factor returns f_m = standardised changes projected on eigenvector m (m = 1..7).
Uncertainty: moving-block bootstrap, block 25 days, 1000 replicates, seed 0
(same scheme as scripts/bootstrap_real_spectrum.py), resampling whole days.

Test 47 - is mode 2 the put/call seam?
  Seam series: s_t = sigma(call, delta +0.50, 30d) - sigma(put, delta -0.50, 30d),
  from the raw vendor pillars (calls and puts are smoothed separately by the vendor
  kernel, h3 = 0.001). Statistic: R^2_m = corr(f_m, Delta s)^2 for m = 1..4.
  Under "mode 2 is genuine skew": R^2_2 small (<~0.2).
  Under "mode 2 is substantially a stitching artifact": R^2_2 > 0.5.
  Reading of R^2_2: upper 95% bound < 0.5 -> NOT substantially seam;
                    lower 95% bound > 0.5 -> substantially seam;
                    interval straddles 0.5 -> ambiguous.

Test 48 - is there resolvable relaxation? (a measurement, not a comparison
  with any fitted gamma; the pre-Step-1 gammas are not used)
  Statistic: autocorrelation of f_m at lags 1..5, and S_m = sum of lags 2..5.
  Readings, stated before running:
   - no relaxation:            all lags ~ 0
   - measurement noise only:   lag 1 negative (up to -0.5), lags >= 2 ~ 0
   - relaxation with gamma << 1/day (slow): small negative lags decaying slowly
   - relaxation with gamma ~ 1/day:  lag 1 negative AND lags 2-5 negative,
                                     decaying ~geometrically
   - relaxation with gamma >> 1/day: indistinguishable from measurement noise at
     daily sampling (lag-2 correlation ~ -(1-a)a/2 with a = e^-gamma -> 0)
  Operational criterion for the scope rule ("significant negative lag-1 structure
  not attributable to measurement noise"): for modes 2 AND 3 the lag-1 95% interval
  lies below 0, AND for at least one of modes 2-3 the 95% interval of S_m lies below 0.
  Criterion check on synthetic data (test-49 harness, kernel on, 1000 days, 200
  bootstrap replicates; run with --validate): a fast membrane (gamma_1 ~ 5.8/day,
  noise-like at daily sampling) gives lag 1 ~ -0.5 for modes 2-3 but S ~ 0 -> NOT MET;
  gamma_1 ~ 0.2/day gives lag 1 ~ -0.25 and S ~ -0.14 -> MET. Lag 1 alone cannot
  separate relaxation from noise; S can.
  Caveat: S detects relaxation on timescales >~ 1 day. If the real gammas are
  >> 1/day, test 48 fails even though an operator exists, because daily sampling
  cannot see it. A failed test 48 therefore closes the dynamics claim at this
  sampling frequency only; it does not adjudicate statics.

Scope rule (supervisor, pre-registered): a kernel-convolved forward model
  (Step 3 refit, kernel o (level + operator)) is undertaken only if test 47 is
  "NOT substantially seam" AND test 48 meets the criterion above. Otherwise the
  operator question is recorded as open and unresolved (not refuted), and the
  paper ships as the measurement study.

Run:  python -m scripts.test47_48             (real data)
      python -m scripts.test47_48 --validate  (criterion check on synthetic data)
"""
import sys
import numpy as np
import pandas as pd

from src.data_pipeline import build_study_panel, load_optionmetrics, STUDY_MATURITY_DAYS
from src.pca import run_pca, project_onto_factors

BLOCK, N_BOOT, SEED = 25, 1000, 0
MAX_LAG = 5
N_R2_MODES = 4


def seam_series(df_raw):
    atm = df_raw[(df_raw["days"] == STUDY_MATURITY_DAYS) & np.isclose(df_raw["delta"].abs(), 50)]
    wide = atm.pivot_table(index="date", columns="option_type", values="implied_vol")
    calls = [c for c in wide.columns if str(c).upper().startswith("C")]
    puts = [c for c in wide.columns if str(c).upper().startswith("P")]
    if len(calls) != 1 or len(puts) != 1:
        raise ValueError(f"unexpected option_type values: {list(wide.columns)}")
    return (wide[calls[0]] - wide[puts[0]]).rename("seam")


def factors_and_seam(df_raw):
    panel = build_study_panel(df_raw)["panel"]
    chg = np.diff(panel.to_numpy(dtype=float), axis=0)
    pca = run_pca(chg)
    f = project_onto_factors(chg, pca.eigenvectors)          # (T, n_modes)
    dseam = seam_series(df_raw).diff().reindex(panel.index[1:]).to_numpy()
    return f, dseam, pca.explained_variance_ratio


def r2(f, x):
    ok = np.isfinite(x)
    return np.array([np.corrcoef(f[ok, m], x[ok])[0, 1] ** 2 for m in range(N_R2_MODES)])


def acf(f, max_lag=MAX_LAG):
    fc = f - f.mean(axis=0)
    var = (fc ** 2).mean(axis=0)
    return np.array([(fc[h:] * fc[:-h]).mean(axis=0) / var for h in range(1, max_lag + 1)])  # (lag, mode)


def block_indices(T, rng):
    n_blocks = int(np.ceil(T / BLOCK))
    starts = rng.integers(0, T - BLOCK + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]


def bootstrap(f, dseam):
    rng = np.random.default_rng(SEED)
    T = len(f)
    R2, A = [], []
    for _ in range(N_BOOT):
        idx = block_indices(T, rng)
        R2.append(r2(f[idx], dseam[idx]))
        # autocorrelation within blocks only: concatenation seams are rare (1/BLOCK)
        A.append(acf(f[idx]))
    return np.array(R2), np.array(A)


def criterion_met(A_boot):
    a_ci = np.percentile(A_boot, [2.5, 97.5], axis=0)
    S_ci = np.percentile(A_boot[:, 1:].sum(axis=1), [2.5, 97.5], axis=0)
    lag1_neg = all(a_ci[1, 0, m] < 0 for m in (1, 2))
    s_neg = any(S_ci[1, m] < 0 for m in (1, 2))
    return lag1_neg, s_neg


def validate():
    global N_BOOT
    import scripts.test49_vendor_kernel as t49
    t49.N_DAYS, N_BOOT = 1000, 200
    for D, label in ((3.0, "fast, noise-like"), (0.1, "resolvable relaxation")):
        p, lvl = t49.simulate_truth(D, 1)
        f, ds, _ = factors_and_seam(t49.vendor_pillars(p, lvl, 0.002, t49.H_VENDOR, 1))
        A = acf(f)
        _, A_b = bootstrap(f, ds)
        lag1_neg, s_neg = criterion_met(A_b)
        print(f"D={D} ({label}): lag1 m2,m3 = {A[0, 1]:+.2f},{A[0, 2]:+.2f}  "
              f"S m2,m3 = {A[1:, 1].sum():+.2f},{A[1:, 2].sum():+.2f}  "
              f"-> {'MET' if lag1_neg and s_neg else 'NOT MET'}")


def main():
    df_raw = load_optionmetrics()
    f, dseam, evr = factors_and_seam(df_raw)
    np.set_printoptions(precision=3, suppress=True)
    print(f"factors: {f.shape}, seam changes finite: {np.isfinite(dseam).sum()}")

    R2_hat, A_hat = r2(f, dseam), acf(f)
    R2_b, A_b = bootstrap(f, dseam)
    r2_ci = np.percentile(R2_b, [2.5, 97.5], axis=0)
    a_ci = np.percentile(A_b, [2.5, 97.5], axis=0)
    S_hat = A_hat[1:].sum(axis=0)
    S_ci = np.percentile(A_b[:, 1:].sum(axis=1), [2.5, 97.5], axis=0)

    print("\nTest 47: R^2 of factor returns on Delta(ATM call vol - ATM put vol)")
    for m in range(N_R2_MODES):
        print(f"  mode {m+1}: R^2 = {R2_hat[m]:.3f}  95% [{r2_ci[0, m]:.3f}, {r2_ci[1, m]:.3f}]")
    lo, hi = r2_ci[:, 1]
    t47 = ("NOT substantially seam" if hi < 0.5 else
           "substantially seam" if lo > 0.5 else "ambiguous")
    print(f"  reading (mode 2): {t47}")

    print("\nTest 48: autocorrelation of factor returns, lags 1..5  [95% interval]")
    for m in range(f.shape[1]):
        cells = "  ".join(f"{A_hat[h, m]:+.3f} [{a_ci[0, h, m]:+.3f},{a_ci[1, h, m]:+.3f}]"
                          for h in range(MAX_LAG))
        print(f"  mode {m+1} (share {evr[m]:.4f}): {cells}")
        print(f"          S (lags 2-5) = {S_hat[m]:+.3f} [{S_ci[0, m]:+.3f},{S_ci[1, m]:+.3f}]")
    lag1_neg, s_neg = criterion_met(A_b)
    t48 = lag1_neg and s_neg
    print(f"  criterion: lag-1 < 0 for modes 2 and 3: {lag1_neg}; "
          f"S < 0 for mode 2 or 3: {s_neg}  ->  {'MET' if t48 else 'NOT MET'}")

    go = (t47 == "NOT substantially seam") and t48
    print(f"\nScope rule: {'kernel-convolved refit (Step 3)' if go else 'measurement-study branch'}")


if __name__ == "__main__":
    validate() if "--validate" in sys.argv else main()