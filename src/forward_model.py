"""
Forward model for the daily-change spectrum of the membrane + level model.

``model_spectrum`` gives the exact correlation-PCA variance shares of daily
changes for a membrane (rates gamma_k = D q_k^2 + kappa q_k^4) plus a uniform
level factor, observed at given sample points. It is used for the analytic
diffusion-ceiling decomposition in ``scripts/test49_followups.py``.

It does not include the vendor's smoothing kernel, which is part of the
measurement (report, Sec. 3); it is therefore not a model of the observed surface.
"""

import numpy as np

from src.simulate import wavenumbers, decay_rates, cosine_basis


def model_spectrum(
    D,
    kappa,
    mean_inc_var,
    z_points,
    z_min=-1.15,
    z_max=1.10,
    n_modes=40,
    dt=1.0,
    return_corr=False,
):
    """Exact variance shares of daily changes under the composite model.

    The membrane lives on [z_min, z_max] (Neumann boundaries) and is observed
    at ``z_points``: the centres of the panel bins that survive the coverage
    rule (see ``data_pipeline.build_study_panel``). The mode sum runs to
    ``n_modes`` well above the number of sample points, so the field is not
    truncated to the grid.

    Covariance of increments, built from the OU increment law

        Var(dF_k) = 2 V_k (1 - exp(-gamma_k dt)),
        gamma_k = D q_k^2 + kappa q_k^4,  q_k = k pi / L,

    plus a uniform level-factor increment variance ``mean_inc_var``.

    By default, returns the correlation-normalised eigenvalue shares,
    matching ``run_pca``. If ``return_corr=True``, returns the correlation
    matrix itself instead.
    """
    L = z_max - z_min
    q = wavenumbers(n_modes, L)
    gamma = decay_rates(q, D, kappa)
    stationary_variance = 1.0 / (2.0 * gamma)          # unit noise amplitude
    w = 2.0 * stationary_variance * (1.0 - np.exp(-gamma * dt))  # per-mode increment variance

    Phi = cosine_basis(q, z_points, z_min, L)          # (n_modes, n_points)
    cov = Phi.T @ (w[:, None] * Phi) + mean_inc_var

    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    if return_corr:
        return corr

    eig = np.linalg.eigvalsh(corr)[::-1]  # descending
    return eig / eig.sum()
