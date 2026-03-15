"""
Plotting Utilities
===================

Consistent, publication-quality plotting functions for all project
visualisations. Designed to reproduce the style of the presentation
slides (clean, minimal, with clear labels).

All plot functions return (fig, ax) tuples so you can further
customise them in notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.pca import PCAResult


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colour palette (matching the presentation's style)
COLORS = {
    "mode0": "#d62728",   # red
    "mode1": "#1f77b4",   # blue
    "mode2": "#bcbd22",   # yellow-green
    "mode3": "#9467bd",   # purple
    "mode4": "#8c564b",   # brown
    "theoretical": "#7f7f7f",  # grey for theoretical overlays
    "data": "#2ca02c",    # green for empirical data
}

MODE_COLORS = [COLORS[f"mode{i}"] for i in range(5)] + ["#e377c2", "#17becf"]


def apply_style():
    """Apply the project's matplotlib style globally."""
    mpl.rcParams.update(STYLE_CONFIG)


def plot_eigenmodes(
    pca_result: PCAResult,
    z_grid: np.ndarray,
    n_modes: int = 5,
    theoretical_modes: np.ndarray | None = None,
    title: str = "PCA Eigenmodes",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot PCA eigenmodes as functions of moneyness z (Slide 6 style).

    Parameters
    ----------
    pca_result : PCAResult
        PCA results.
    z_grid : np.ndarray, shape (n_z,)
        Moneyness grid.
    n_modes : int
        Number of modes to plot.
    theoretical_modes : np.ndarray or None, shape (n_modes, n_z)
        If provided, overlay theoretical (Fourier) modes in grey.
    title : str
        Overall figure title.

    Returns
    -------
    fig, axes : matplotlib figure and axes array.
    """
    apply_style()
    n_modes = min(n_modes, len(pca_result.eigenvalues))

    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for k in range(n_modes):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]

        # Empirical mode
        color = MODE_COLORS[k % len(MODE_COLORS)]
        ax.plot(z_grid, pca_result.eigenvectors[k], "o-",
                color=color, markersize=3, linewidth=1.5,
                label="empirical")

        # Theoretical overlay
        if theoretical_modes is not None and k < theoretical_modes.shape[0]:
            # Align sign
            sign = np.sign(np.dot(
                pca_result.eigenvectors[k], theoretical_modes[k]
            ))
            ax.plot(z_grid, sign * theoretical_modes[k], "--",
                    color=COLORS["theoretical"], linewidth=1.5,
                    label="Fourier", alpha=0.7)

        var_pct = pca_result.explained_variance_ratio[k] * 100
        ax.set_title(f"eigenmode {k} ({var_pct:.1f}%)")
        ax.set_xlabel("moneyness z")
        ax.set_ylabel("")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        if k == 0 and theoretical_modes is not None:
            ax.legend(fontsize=8)

    # Hide unused axes
    for k in range(n_modes, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_eigenvalue_spectrum(
    pca_result: PCAResult,
    theoretical_eigenvalues: np.ndarray | None = None,
    title: str = "Eigenvalue Spectrum",
) -> tuple[plt.Figure, tuple]:
    """Plot the eigenvalue spectrum: bar chart + log-log scaling.

    Reproduces the style of Slides 6 (bar chart) and 9 (log-log).

    Parameters
    ----------
    pca_result : PCAResult
        PCA results.
    theoretical_eigenvalues : np.ndarray or None
        If provided, overlay theoretical spectrum.
    title : str
        Figure title.

    Returns
    -------
    fig, (ax_bar, ax_loglog) : figure and two axes.
    """
    apply_style()
    fig, (ax_bar, ax_loglog) = plt.subplots(1, 2, figsize=(12, 5))

    n = len(pca_result.eigenvalues)
    k = np.arange(n)
    var_pct = pca_result.explained_variance_ratio * 100

    # Bar chart of explained variance
    colors = [MODE_COLORS[i % len(MODE_COLORS)] for i in range(n)]
    ax_bar.bar(k, var_pct, color=colors, alpha=0.8, edgecolor="white")
    for i, v in enumerate(var_pct):
        if v > 2:
            ax_bar.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax_bar.set_xlabel("mode number")
    ax_bar.set_ylabel("explained variance (%)")
    ax_bar.set_title("Explained Variance")

    # Log-log plot of eigenvalues
    k_plot = np.arange(1, n + 1)
    ax_loglog.loglog(k_plot, pca_result.eigenvalues, "o-",
                     color=COLORS["data"], markersize=5, label="empirical")

    if theoretical_eigenvalues is not None:
        ax_loglog.loglog(k_plot[:len(theoretical_eigenvalues)],
                         theoretical_eigenvalues, "s--",
                         color=COLORS["theoretical"], markersize=4,
                         label="theoretical ~k⁻²")

    # Reference k^-2 line
    ref_line = pca_result.eigenvalues[0] * (k_plot.astype(float)) ** (-2)
    ax_loglog.loglog(k_plot, ref_line, ":", color="grey", alpha=0.5,
                     label="~k⁻²")

    ax_loglog.set_xlabel("mode number k")
    ax_loglog.set_ylabel("eigenvalue")
    ax_loglog.set_title("Eigenvalue Scaling")
    ax_loglog.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, (ax_bar, ax_loglog)


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    z_grid: np.ndarray | None = None,
    title: str = "Correlation Matrix",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a correlation matrix as a heatmap.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (n, n)
        Correlation matrix.
    z_grid : np.ndarray or None
        If provided, use as tick labels.
    title : str
        Plot title.

    Returns
    -------
    fig, ax : figure and axes.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="correlation")

    if z_grid is not None:
        n_ticks = 6
        tick_idx = np.linspace(0, len(z_grid) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{z_grid[i]:.1f}" for i in tick_idx])
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{z_grid[i]:.1f}" for i in tick_idx])
        ax.set_xlabel("moneyness z")
        ax.set_ylabel("moneyness z")

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_rolling_stability(
    rolling_results: list[PCAResult],
    z_grid: np.ndarray,
    n_modes: int = 5,
    title: str = "Eigenmode Stability Over Time",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot eigenmodes from rolling windows overlaid (Slide 8 style).

    Shows all rolling-window eigenvectors as faint lines, with the
    mean highlighted, to demonstrate stability.

    Parameters
    ----------
    rolling_results : list of PCAResult
        Results from rolling_pca().
    z_grid : np.ndarray
        Spatial grid.
    n_modes : int
        Number of modes to show.
    title : str
        Figure title.

    Returns
    -------
    fig, axes : figure and axes.
    """
    apply_style()
    from src.pca import align_eigenvector_signs

    # Extract and align eigenvectors
    all_evecs = [r.eigenvectors[:n_modes] for r in rolling_results]
    all_evecs = align_eigenvector_signs(all_evecs)

    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for k in range(n_modes):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]
        color = MODE_COLORS[k % len(MODE_COLORS)]

        # Plot each window as a faint line
        for evecs in all_evecs:
            ax.plot(z_grid, evecs[k], color="grey", alpha=0.15, linewidth=0.8)

        # Overlay the mean
        mean_mode = np.mean([e[k] for e in all_evecs], axis=0)
        ax.plot(z_grid, mean_mode, color=color, linewidth=2.0, label="mean")

        ax.set_title(f"eigenmode {k}")
        ax.set_xlabel("moneyness z")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)

    for k in range(n_modes, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_string_simulation(
    z_grid: np.ndarray,
    surface_snapshot: np.ndarray,
    title: str = "Volatility Surface as a Vibrating String",
) -> tuple[plt.Figure, plt.Axes]:
    """Visualise a snapshot of the surface as a displaced string (Slide 9).

    Parameters
    ----------
    z_grid : np.ndarray, shape (n_z,)
    surface_snapshot : np.ndarray, shape (n_z,)
        One row from the simulated surface.
    title : str

    Returns
    -------
    fig, ax : figure and axes.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(z_grid, surface_snapshot, "o-", color=COLORS["mode1"],
            markersize=4, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=1, alpha=0.5, linestyle="--",
               label="equilibrium")
    ax.fill_between(z_grid, 0, surface_snapshot, alpha=0.1,
                    color=COLORS["mode1"])

    ax.set_xlabel("position z (moneyness)")
    ax.set_ylabel("displacement from equilibrium")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
