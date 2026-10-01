"""Figure 1 of the report: tail of the daily-change spectrum, real vs synthetic.

Tail-normalised variance shares (modes 2-7 divided by their sum, so the tuned
level factor plays no role) on log-log axes:
  - real 30-day panel, with 95% moving-block bootstrap intervals;
  - unsmoothed pure diffusion at its ceiling (kernel off, D = 3, q = 0);
  - kernel-smoothed pure diffusion in the four cells whose p_tail lies inside the
    real interval (test 49).
Synthetic spectra are 3-seed means from the test-49 harness (deterministic).

Requires data/processed/study_spectrum.npz (scripts/build_panel.py) and
data/processed/real_spectrum_boot.npy (scripts/bootstrap_real_spectrum.py).

    python -m scripts.make_figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scripts.test49_vendor_kernel as t49
from src.data_pipeline import PROCESSED_DIR, REPO_ROOT, load_study_spectrum
from src.pca import slope

FIG_DIR = REPO_ROOT / "report" / "figures"
MATCHED_CELLS = [(0.0, 0.03), (0.002, 0.03), (0.005, 0.1), (0.005, 0.3)]   # (q, D)
CEILING_CELL = (0.0, 3.0)


def tail(s):
    s = np.asarray(s)
    return s[1:] / s[1:].sum()


def synthetic(q, D, h):
    spectra = []
    for seed in t49.SEEDS:
        p, level = t49.simulate_truth(D, seed)
        spectra.append(t49.instrument(t49.vendor_pillars(p, level, q, h, seed))[0])
    return np.mean(spectra, axis=0)


def main():
    real = load_study_spectrum()["shares"]
    boot = np.load(PROCESSED_DIR / "real_spectrum_boot.npy")
    lo, hi = np.percentile(np.array([tail(b) for b in boot]), [2.5, 97.5], axis=0)
    k = np.arange(2, len(real) + 1)

    ceiling = synthetic(*CEILING_CELL, t49.H_OFF)
    smoothed = {cell: synthetic(*cell, t49.H_VENDOR) for cell in MATCHED_CELLS}

    np.set_printoptions(precision=4, suppress=True)
    print(f"real            tail {tail(real)}  p_tail {slope(real[1:], k_start=2):.3f}")
    print(f"unsmoothed      tail {tail(ceiling)}  p_tail {slope(ceiling[1:], k_start=2):.3f}")
    for (q, D), s in smoothed.items():
        print(f"smoothed q={q:<5} D={D:<4} tail {tail(s)}  p_tail {slope(s[1:], k_start=2):.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    p_sm = [slope(s[1:], k_start=2) for s in smoothed.values()]
    for i, s in enumerate(smoothed.values()):
        ax.plot(k, tail(s), color="tab:blue", lw=1, alpha=0.8,
                label=(f"kernel-smoothed diffusion, 4 cells ($p$ = {min(p_sm):.2f}-{max(p_sm):.2f})"
                       if i == 0 else None))
    ax.plot(k, tail(ceiling), "k--", lw=1,
            label=f"unsmoothed diffusion, ceiling ($p$ = {slope(ceiling[1:], k_start=2):.2f})")
    t = tail(real)
    ax.errorbar(k, t, yerr=[t - lo, hi - t], fmt="o", color="tab:red", ms=4, capsize=2,
                label=f"SPX 30-day, 95% CI ($p$ = {slope(real[1:], k_start=2):.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xticklabels([str(i) for i in k])
    ax.minorticks_off()
    ax.set_xlabel("mode $k$")
    ax.set_ylabel("share of tail variance (modes 2-7)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig1_spectrum.pdf"
    fig.savefig(out)
    print("saved:", out)


if __name__ == "__main__":
    main()
