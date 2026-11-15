"""Parameters and provenance shared by the notebook and command-line runner."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


REFERENCE_ROOT = Path(__file__).resolve().parent.parent

from free_fermion_gamma.model import ABCAParams, FreeFermionConfig, build_free_fermion_system
from static_parquet_memory.solver import estimate_problem_memory
from static_parquet_memory.gamma_ph_only import estimate_gamma_ph_memory


def default_config() -> dict:
    """Single-point settings matching the D=110 meV, n=6.375e11 reference run.

    The 5e-3 parquet tolerance is a numerical acceptance threshold, not an
    energy error bar. Tighten it and refine the mesh for convergence studies.
    """
    model = FreeFermionConfig(
        radius=10, spacing=0.012, isospin="K up", active_band_index=4,
        carrier_density_cm2=6.375e11, temperature_eV=1e-4,
        dielectric_constant=15.0, gate_distance_nm=25.0,
        band_params=ABCAParams(delta_d=0.11),
    )
    return {
        "schema": "constrained_rpa_parquet_superconductivity_v1",
        "model": asdict(model),
        "solver": {
            "tolerance": 5e-3, "maximum_iterations": 100,
            "krylov_method": "lgmres", "krylov_restart": 24,
            "krylov_maximum_iterations": 3, "krylov_outer_k": 3,
            "maximum_backtracks": 20, "minimum_line_search_step": 0.25,
            "pole_ceiling": 0.8, "preserve_physical_symmetry": True,
            "stagnation_window": 0, "stagnation_relative_improvement": 0.01,
            "stagnation_minimum_iterations": 5, "memory_mode": "streamed",
            "maximum_memory_mb": 12000,
        },
        "continuation": {
            "target_scale": 1.0, "starting_scale": 0.05,
            "stages": [round(i / 100, 2) for i in range(5, 101)],
            "minimum_scale_step": 1e-4, "maximum_attempts": 300,
            "nonconvergence_policy": "adaptive", "tolerance_relaxation_factor": 3.0,
        },
        "pairing": {"run_nonlinear": True, "nonlinear_kwargs": {
            "gap_tolerance": 1e-8, "max_iterations": 2000,
            "initial_gap": 1e-3, "mixing": 0.25, "minimum_mixing": 0.02,
            "filling_tolerance": 1e-11,
        }},
    }


def point_config(delta_d_meV=110.0, density_1e11_cm2=6.375, *, mesh_radius=10):
    """Convenient controls: displacement in meV and density in 1e11 cm^-2."""
    import math
    if not math.isfinite(delta_d_meV):
        raise ValueError("delta_d_meV must be finite")
    if not math.isfinite(density_1e11_cm2) or density_1e11_cm2 <= 0:
        raise ValueError("density_1e11_cm2 must be positive and finite")
    if isinstance(mesh_radius, bool) or int(mesh_radius) != mesh_radius or mesh_radius < 1:
        raise ValueError("mesh_radius must be a positive integer")
    config = default_config()
    config["model"]["band_params"]["delta_d"] = float(delta_d_meV) / 1000
    config["model"].update(carrier_density_cm2=float(density_1e11_cm2) * 1e11,
                           radius=int(mesh_radius), spacing=0.12 / mesh_radius)
    return config


def smoke_config():
    """Small, thermally broadened full-coupling workflow check; not physical results."""
    config = point_config(mesh_radius=2)
    config["model"]["temperature_eV"] = 0.005
    config["solver"].update(tolerance=1e-7, maximum_iterations=40,
                           minimum_line_search_step=0.05, maximum_memory_mb=512)
    config["continuation"]["stages"] = [.05, .1, .2, .4, .6, .8, 1.]
    return config

def make_system(config: dict):
    model = dict(config["model"])
    model["band_params"] = ABCAParams(**model["band_params"])
    return build_free_fermion_system(FreeFermionConfig(**model))


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def result_directory(config: dict, base: Path | str | None = None) -> Path:
    base = Path(base) if base is not None else Path.cwd() / "results"
    model = config["model"]
    return base / (f"M{model['radius']}_D{model['band_params']['delta_d']:g}"
                   f"_n{model['carrier_density_cm2'] / 1e11:g}e11_{config_digest(config)[:12]}")


def preview(config: dict) -> dict:
    system = make_system(config)
    problem = estimate_problem_memory(system, memory_profile="gamma-ph")
    solver = config["solver"]
    memory = estimate_gamma_ph_memory(problem, **{
        name: solver[name] for name in (
            "krylov_restart", "krylov_method", "krylov_outer_k", "memory_mode",
            "preserve_physical_symmetry",
        )
    })
    return {
        "momenta": len(system.coords), "transfers": problem["number_of_sectors"],
        "chemical_potential_eV": system.chemical_potential,
        "quadrature_weight": system.weight,
        "estimated_peak_GB": memory["estimated_peak_mb"] / 1000,
        "within_memory_budget": memory["estimated_peak_mb"] <= solver["maximum_memory_mb"],
        "each_vertex_GB": problem["vertex_bytes"] / 1e9,
        "hf_used": False, "screening_species_count": 1,
        "excluded_rpa_pair": [system.active_band_index, system.active_band_index],
    }


def reference_provenance() -> dict:
    """Hash bundled source files without machine-specific paths."""
    paths = sorted(p for package in ("parquet_superconductivity", "free_fermion_gamma",
                                     "static_parquet_memory")
                   for p in (REFERENCE_ROOT / package).glob("*.py"))
    return {"source": "bundled standalone code", "sha256": {
        p.relative_to(REFERENCE_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths}}
