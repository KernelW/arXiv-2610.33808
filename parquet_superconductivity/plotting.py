"""Scientific diagnostic figures on the actual open hexagonal mesh."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_screening(crpa):
    q = np.linalg.norm(crpa.transfer_coords @ crpa.system.basis, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for values, label in ((crpa.bare_interaction, "Bare V"),
                          (crpa.screened_interaction, r"$V^c_{RPA}$")):
        axes[0].scatter(q, np.asarray(values).real, s=8, label=label, alpha=.6)
    for values, label in ((crpa.polarization_full, "All 64 pairs"),
                          (crpa.polarization_active, "Active–active"),
                          (crpa.polarization_rest, "Remaining 63 pairs")):
        axes[1].scatter(q, -np.asarray(values).real, s=8, label=label, alpha=.6)
    for ax in axes:
        ax.set_xlabel(r"$|Q|a$")
        ax.legend()
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Interaction (eV, reference momentum units)")
    axes[1].set_ylabel(r"$-\Pi$ (eV$^{-1}$)")
    fig.suptitle("Constrained screening · free bands · one isospin")
    return fig


def plot_gap(system, result):
    xy = system.points
    mode = result.linearized.gap_mode
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    lim = max(np.max(np.abs(mode.real)), np.max(np.abs(mode.imag)), 1e-15)
    for ax, values, title, cmap, vmin, vmax in (
        (axes[0], mode.real, r"Re $\Delta_{lin}$", "RdBu_r", -lim, lim),
        (axes[1], mode.imag, r"Im $\Delta_{lin}$", "RdBu_r", -lim, lim),
        (axes[2], np.abs(mode), r"$|\Delta_{lin}|$", "viridis", 0, 1),
    ):
        pic = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap,
                         vmin=vmin, vmax=vmax, s=max(12, 4000 / len(xy)))
        if result.xi.min() < 0 < result.xi.max() and len(xy) >= 3:
            ax.tricontour(xy[:, 0], xy[:, 1], result.xi, levels=[0],
                          colors="limegreen", linewidths=.7)
        ax.set(title=title, xlabel=r"$k_x a$", ylabel=r"$k_y a$")
        ax.set_aspect("equal")
        fig.colorbar(pic, ax=ax)
    fig.suptitle(rf"Odd Cooper mode: $\lambda_{{max}}={result.linearized.leading_eigenvalue:.6g}$"
                 + f" · {result.linearized.classification}")
    return fig
