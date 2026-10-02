"""Test 49 follow-ups (POST HOC, descriptive; not part of the pre-registration).

1. Ceiling decomposition: why the unsmoothed-diffusion ceiling moved 1.87 -> 2.63.
   Analytic forward model in the fast-regime limit (kappa = 0, D -> large), adding one
   instrument change at a time (one ordering; contributions are order-dependent).
2. Shape: does kernel-smoothed pure diffusion reproduce the spectrum's shape
   (mode-share profile, log-log curvature), or only its tail slope?
3. Tail-internal comparison: modes 2-7 only (shares renormalised within the tail,
   and local slopes, which are ratios of adjacent tail modes), against block-bootstrap
   95% intervals of the same quantities on the real data. Independent of mode 1.

Run:  python -m scripts.test49_followups
"""
import sys
from pathlib import Path

import numpy as np

import scripts.test49_vendor_kernel as t49
from src.forward_model import model_spectrum
from src.pca import slope
from src.simulate import bin_centres, cosine_basis, wavenumbers

K_TAIL = np.arange(2, 8)


def p_tail(s):
    return slope(s[1:], k_start=2)


def local_slopes(s):
    k = np.arange(2, len(s) + 1)
    return -np.diff(np.log(s[1:])) / np.diff(np.log(k))


def old_model(D, n_pts=8):
    """Pre-Step-1 forward model: index-unit rates, linspace points, n_modes = n_pts."""
    z = np.linspace(-1.15, 1.10, n_pts)
    L = 2.25
    k = np.arange(1, n_pts + 1)
    g = D * k ** 2
    w = (1 - np.exp(-g)) / g
    Phi = cosine_basis(k * np.pi / L, z, -1.15, L)
    cov = Phi.T @ (w[:, None] * Phi)
    d = np.sqrt(np.diag(cov))
    e = np.linalg.eigvalsh(cov / np.outer(d, d))[::-1]
    return e / e.sum()


def decomposition():
    D = 1e3
    c8 = bin_centres(-1.15, 1.10, 8)
    c7 = c8[1:]
    lin8 = np.linspace(-1.15, 1.10, 8)
    # level variance giving mode-1 share ~0.91 (the kernel-OFF harness value)
    ms = np.linspace(0, 1, 2001)
    m91 = ms[np.argmin([abs(model_spectrum(D, 0, m, c7)[0] - 0.91) for m in ms])]
    steps = [
        ("S0 old instrument (8 linspace pts, 8 modes, no level)", old_model(D)),
        ("S1 + continuum field (40 modes)", model_spectrum(D, 0, 0, lin8)),
        ("S2 + bin-centre sampling", model_spectrum(D, 0, 0, c8)),
        ("S3 + bin 0 dropped (7 pts, 6 tail modes)", model_spectrum(D, 0, 0, c7)),
        ("S4 + level factor (mode 1 = 0.91)", model_spectrum(D, 0, m91, c7)),
    ]
    print("1. Ceiling decomposition (analytic, fast-regime limit)")
    prev = None
    for name, s in steps:
        v = p_tail(s)
        delta = "" if prev is None else f"  ({v - prev:+.2f})"
        print(f"   {name:55s} p_tail = {v:.3f}{delta}")
        prev = v
    print("   S5 full kernel-OFF harness (pillars, quotes, binning; D=3, q=0) = 2.630 (test 49)")


def real_reference():
    root = Path(__file__).resolve().parent.parent / "data" / "processed"
    try:
        from src.data_pipeline import load_study_spectrum
        real = load_study_spectrum()["shares"]
        boot = np.load(root / "real_spectrum_boot.npy")
        return real, np.percentile(boot, [2.5, 97.5], axis=0)
    except (FileNotFoundError, OSError):
        return None, None


def shape():
    np.set_printoptions(precision=4, suppress=True)
    real, ci = real_reference()
    print("\n2. Shape of kernel-ON spectra")
    if real is not None:
        print(f"   real                 shares {real}  local slopes {np.round(local_slopes(real), 2)}")
    print("   (a) cells whose mean p_tail lies in the real interval, 3 seeds:")
    cells = {}
    for q, D in [(0.0, 0.03), (0.002, 0.03), (0.005, 0.1), (0.005, 0.3)]:
        S = []
        for seed in t49.SEEDS:
            p, lvl = t49.simulate_truth(D, seed)
            S.append(t49.instrument(t49.vendor_pillars(p, lvl, q, t49.H_VENDOR, seed))[0])
        m = np.mean(S, axis=0)
        cells[(q, D)] = m
        line = f"   q={q:<5} D={D:<5} shares {m}  local slopes {np.round(local_slopes(m), 2)}"
        if ci is not None:
            line += f"  in real CI {((m >= ci[0]) & (m <= ci[1])).astype(int)}"
        print(line, flush=True)
    print("   (b) whole kernel-ON grid, seed 0: first -> last local slope")
    for q in t49.Q_GRID:
        cells_b = []
        for D in t49.D_GRID:
            p, lvl = t49.simulate_truth(D, 0)
            ls = local_slopes(t49.instrument(t49.vendor_pillars(p, lvl, q, t49.H_VENDOR, 0))[0])
            cells_b.append(f"D={D}: {ls[0]:.1f}->{ls[-1]:.1f}")
        print(f"   q={q}: " + " | ".join(cells_b), flush=True)
    return cells


def tail_internal(cells):
    np.set_printoptions(precision=3, suppress=True)
    tail = lambda s: np.asarray(s)[1:] / np.asarray(s)[1:].sum()
    print("\n3. Tail-internal comparison (modes 2-7 only; mode 1 plays no role)")
    real, _ = real_reference()
    root = Path(__file__).resolve().parent.parent / "data" / "processed"
    boot_path = root / "real_spectrum_boot.npy"
    if real is not None and boot_path.exists():
        boot = np.load(boot_path)
        bt = np.array([tail(b) for b in boot])
        bl = np.array([local_slopes(b) for b in boot])
        t_ci = np.percentile(bt, [2.5, 97.5], axis=0)
        l_ci = np.percentile(bl, [2.5, 97.5], axis=0)
        print(f"   real  tail shares {tail(real)}")
        print(f"         95% lo      {t_ci[0]}\n         95% hi      {t_ci[1]}")
        print(f"   real  local slopes {np.round(local_slopes(real), 2)}")
        print(f"         95% lo       {np.round(l_ci[0], 2)}\n         95% hi       {np.round(l_ci[1], 2)}")
    else:
        t_ci = l_ci = None
    for (q, D), m in cells.items():
        ts, ls = tail(m), local_slopes(m)
        line = f"   q={q:<5} D={D:<5} tail shares {ts}  local slopes {np.round(ls, 2)}"
        if t_ci is not None:
            line += (f"  in CI: shares {((ts >= t_ci[0]) & (ts <= t_ci[1])).astype(int)}"
                     f" slopes {((ls >= l_ci[0]) & (ls <= l_ci[1])).astype(int)}")
        print(line)


if __name__ == "__main__":
    decomposition()
    tail_internal(shape())