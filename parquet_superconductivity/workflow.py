"""Three separately runnable, resumable stages of the no-HF calculation."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import uuid

import numpy as np

from .config import config_digest, make_system, preview, reference_provenance, result_directory
from .crpa import compute_constrained_rpa, load_constrained_rpa, save_constrained_rpa


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize(config, directory=None):
    directory = Path(directory) if directory is not None else result_directory(config)
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError("This result directory belongs to another configuration; choose a new directory.")
    write_json(config_path, config)
    provenance = reference_provenance()
    provenance["config_sha256"] = config_digest(config)
    provenance_path = directory / "provenance.json"
    if provenance_path.is_file() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Reference files changed since this run. Use a new output directory.")
    write_json(provenance_path, provenance)
    return directory


def stage1_screening(config, directory=None, *, progress=True):
    directory = initialize(config, directory)
    system = make_system(config)
    path = directory / "01_screening" / "constrained_rpa.npz"
    if path.is_file():
        crpa = load_constrained_rpa(path, system)
    else:
        crpa = compute_constrained_rpa(system, progress=progress)
        save_constrained_rpa(crpa, path)
    return system, crpa, directory


def stage2_vertex(config, directory=None, *, progress=True, resume=True):
    from .parquet import build_screened_problem, solve_screened_parquet, reconstruct_full_gamma
    from .progress import GammaPhProgressPrinter

    printer = GammaPhProgressPrinter(30) if progress else None

    def show_progress(event):
        printer(event)
        if event['event'] == 'stage_start' and 'requested_tolerance' in event:
            print(f"  Stage tolerance: {event['requested_tolerance']:.6g}", flush=True)
        elif (event['event'] == 'stage_end' and not event['converged']
                and event.get('next_tolerance') is not None):
            print(f"  Saved failed vertex; next stage tolerance: {event['next_tolerance']:.6g}",
                  flush=True)

    plan = preview(config)
    if not plan["within_memory_budget"]:
        raise MemoryError(f"Planned solver working set {plan['estimated_peak_GB']:.3f} GB exceeds budget.")
    system, crpa, directory = stage1_screening(config, directory, progress=progress)
    problem = build_screened_problem(system, crpa,
                                    maximum_memory_mb=config["solver"]["maximum_memory_mb"])
    result, continuation = solve_screened_parquet(
        problem, **config["continuation"], **config["solver"],
        checkpoint_directory=directory / "02_vertex" / "checkpoints",
        failed_iterate_directory=directory / "02_vertex" / "failed_iterates",
        resume=resume, progress=show_progress if progress else None,
    )
    write_json(directory / "02_vertex" / "continuation.json", continuation)
    if not continuation["target_reached"]:
        raise RuntimeError(
            f"Parquet retained g={continuation['retained_scale']:g}; target not reached "
            f"({continuation['status']}). Saved stage state can be resumed; "
            "failed iterates remain labeled as unconverged. "
            "Superconductivity has not been evaluated."
        )
    gamma = reconstruct_full_gamma(result, directory / "02_vertex",
                                   target_scale=config["continuation"]["target_scale"])
    return system, problem, result, gamma, directory


def stage3_pairing(config, directory=None, *, progress=True, problem=None, gamma=None):
    from .parquet import build_screened_problem, ensure_full_gamma, FULL_GAMMA_CONSTRUCTION
    from .pairing import extract_cooper_vertex, solve_superconductivity, save_superconductivity

    system, crpa, directory = stage1_screening(config, directory, progress=progress)
    if problem is None:
        problem = build_screened_problem(system, crpa,
                                        maximum_memory_mb=config["solver"]["maximum_memory_mb"])
    owns_archive = gamma is None or gamma.metadata.get('construction') != FULL_GAMMA_CONSTRUCTION
    if owns_archive:
        # Legacy K checkpoints solve the same parquet equation. Reconstruct the
        # explicit full vertex without repeating Newton when only Gamma is old.
        gamma = ensure_full_gamma(problem, directory / "02_vertex",
                                  target_scale=config["continuation"]["target_scale"])
    try:
        if gamma.metadata.get('interaction_scale') != config['continuation']['target_scale']:
            raise ValueError('Full Gamma does not match the requested target coupling.')
        pair = extract_cooper_vertex(problem, gamma.block)
        gamma_metadata = dict(gamma.metadata)
    finally:
        if owns_archive:
            gamma.flat._mmap.close()
    result = solve_superconductivity(system, pair, **config["pairing"], progress=progress)
    result.diagnostics.update(
        source_directory=str(directory), source_config_sha256=config_digest(config),
        source_gamma_file_sha256=gamma_metadata['gamma_file_sha256'],
        vertex_temperature_eV=config['model']['temperature_eV'],
    )
    destination = directory / "03_superconductivity"
    destination.mkdir(parents=True, exist_ok=True)
    preserve_replaced_pairing_outputs(destination, gamma_metadata)
    np.savez_compressed(destination / "cooper_vertex.npz", interaction=pair.interaction,
                        raw_gamma=pair.raw_gamma, coords=system.coords,
                        metadata_json=np.asarray(json.dumps(pair.diagnostics, sort_keys=True)))
    save_superconductivity(destination / "superconductivity.npz", result)
    summary = {
        "hf_used": False, "config_sha256": config_digest(config),
        "lambda_max": float(result.linearized.leading_eigenvalue),
        "classification": result.linearized.classification,
        "eigen_residual": float(result.linearized.residual),
        "target_scale": config["continuation"]["target_scale"],
        "vertex_construction": gamma_metadata['construction'],
        "parquet_diagnostics": gamma_metadata,
        "cooper_diagnostics": pair.diagnostics,
        "nonlinear": None if result.nonlinear is None else {
            "status": result.nonlinear.status, "converged": bool(result.nonlinear.converged),
            "is_superconducting": bool(result.nonlinear.is_superconducting),
            "max_gap_eV": float(result.nonlinear.max_gap),
            "gap_residual": float(result.nonlinear.gap_residual),
            "filling_residual": float(result.nonlinear.filling_residual),
        },
    }
    write_json(destination / "summary.json", summary)
    return pair, result, directory


def preserve_replaced_pairing_outputs(destination, gamma_metadata):
    """Preserve outputs of a different Gamma before committing a replacement solve."""
    destination = Path(destination).expanduser().resolve()
    names = ('cooper_vertex.npz', 'superconductivity.npz', 'summary.json',
             'linearized_gap.png', 'order_parameter.png',
             'phase_winding.png', 'phase_winding.json', 'phase_winding.npz')
    existing = [destination / name for name in names if (destination / name).is_file()]
    if not existing:
        return None
    saved = {}
    try:
        with np.load(destination / 'superconductivity.npz', allow_pickle=False) as archive:
            saved = json.loads(str(archive['metadata_json'].item()))
    except (OSError, ValueError, KeyError):
        pass  # Preserve unreadable or incomplete old artifacts as well.
    old_gamma = saved.get('source_gamma_metadata', {})
    if (saved.get('vertex_construction') == gamma_metadata['construction']
            and old_gamma.get('construction') == gamma_metadata['construction']
            and old_gamma.get('gamma_file_sha256') == gamma_metadata['gamma_file_sha256']):
        return None
    history = (destination / 'previous_results').resolve()
    if destination not in history.parents:
        raise ValueError('Pairing backup must remain inside its output directory.')
    for path in existing:
        if path.resolve().parent != destination:
            raise ValueError('Refusing to move an output that resolves outside its directory.')
    history.mkdir(parents=True, exist_ok=True)
    # Keep deep Windows result paths short; mkdir reserves each generation so
    # a shortened UUID collision never mixes or overwrites earlier artifacts.
    while True:
        backup = history / uuid.uuid4().hex[:12]
        try:
            backup.mkdir()
        except FileExistsError:
            continue
        break
    for path in existing:
        path.replace(backup / path.name)
    return backup


def run(config, directory=None, *, stage="all", progress=True, resume=True):
    if stage == "screening":
        return stage1_screening(config, directory, progress=progress)
    if stage == "vertex":
        return stage2_vertex(config, directory, progress=progress, resume=resume)
    if stage == "pairing":
        return stage3_pairing(config, directory, progress=progress)
    if stage != "all":
        raise ValueError("stage must be all, screening, vertex, or pairing")
    system, problem, vertex, gamma, directory = stage2_vertex(
        config, directory, progress=progress, resume=resume)
    del vertex
    try:
        return stage3_pairing(config, directory, progress=progress, problem=problem, gamma=gamma)
    finally:
        gamma.flat._mmap.close()
