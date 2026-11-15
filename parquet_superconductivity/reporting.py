"""Export computed quantities without requiring a graphical display."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from .plotting import plot_gap
from .fixed_vertex_temperature import plot_order_parameter
from .phase_winding import plot_saved_phase_winding


def save_pairing_figures(system, result, directory, *, phase=True):
    """Save PNG and vector PDF figures; phase winding requires a nonzero gap."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figures = {"linearized_gap": plot_gap(system, result)}
    if result.nonlinear is not None and result.nonlinear.converged:
        figures["order_parameter"] = plot_order_parameter(system, result)
    paths = []
    for name, fig in figures.items():
        try:
            for ext in ("png", "pdf"):
                path = directory / f"{name}.{ext}"
                fig.savefig(path, dpi=180)
                paths.append(path)
        finally:
            plt.close(fig)
    if (phase and result.nonlinear is not None and result.nonlinear.converged
            and result.nonlinear.is_superconducting):
        try:
            fig, report = plot_saved_phase_winding(directory / "superconductivity.npz")
        except ValueError as error:
            # A coarse mesh, an orbital-reference zero, or an open FS can prevent
            # a phase winding assignment. Never invent a winding in that case.
            (directory / "phase_winding_status.json").write_text(json.dumps(
                {"available": False, "reason": str(error)}, indent=2) + "\n")
            print(f"Phase winding unavailable: {error}")
        else:
            try:
                path = directory / "phase_winding.pdf"
                fig.savefig(path)
                paths.append(path)
            finally:
                plt.close(fig)
    return paths
