"""Finite-temperature gap/number solve with a previously converged full vertex fixed."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .config import config_digest, make_system
from .crpa import load_constrained_rpa
from .pairing import (
    PairingProblem, SuperconductivityResult, extract_cooper_vertex,
    inversion_indices, save_superconductivity, solve_nonlinear_gap,
)
from .parquet import build_screened_problem, ensure_full_gamma, FULL_GAMMA_CONSTRUCTION
from .workflow import write_json, preserve_replaced_pairing_outputs

KB_EV_PER_K = 8.617333262145e-5


def normal_mu_at_filling(energies, target_filling, temperature_eV):
    """Normal active-band chemical potential at finite T and fixed mean occupation."""
    energies = np.asarray(energies, dtype=float)
    temperature = float(temperature_eV)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature_eV must be positive and finite; exact T=0 is not implemented.')
    if energies.ndim != 1 or not energies.size or not np.all(np.isfinite(energies)):
        raise ValueError('energies must be a nonempty finite vector')
    if not np.isfinite(target_filling) or not 0 < target_filling < 1:
        raise ValueError('target_filling must lie strictly between zero and one')
    padding = max(float(np.ptp(energies)), 40 * temperature, 1e-6)
    def number_residual(mu):
        return float(np.mean(expit((mu - energies) / temperature)) - target_filling)
    for _ in range(64):
        left, right = float(energies.min() - padding), float(energies.max() + padding)
        if number_residual(left) < 0 < number_residual(right):
            return float(brentq(number_residual, left, right, xtol=1e-14,
                                rtol=4 * np.finfo(float).eps))
        padding *= 2
    raise RuntimeError('Could not bracket the normal-state number equation.')


def _saved_gap_seed(path, coords, signature, target_filling, gamma_metadata):
    """A validated prior solution is only a starting guess, never a new result."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'].item()))
        if (metadata.get('source_signature') != signature
                or metadata.get('vertex_construction') != FULL_GAMMA_CONSTRUCTION
                or metadata.get('source_gamma_metadata', {}).get('construction') != FULL_GAMMA_CONSTRUCTION
                or metadata.get('source_gamma_metadata', {}).get('gamma_file_sha256')
                    != gamma_metadata['gamma_file_sha256']
                or not metadata.get('converged')
                or not metadata.get('is_superconducting')
                or abs(float(metadata.get('active_filling', -1)) - target_filling) > 1e-10
                or not np.array_equal(archive['pair_coords'], coords)
                or 'delta' not in archive.files):
            return None
        delta = np.asarray(archive['delta'], dtype=np.complex128).copy()
    if delta.shape != (len(coords),) or not np.all(np.isfinite(delta)) or not np.any(np.abs(delta) > 0):
        return None
    return delta


def solve_fixed_vertex_temperature(source_directory, temperature_eV=1e-5, *,
                                   output_directory=None, progress=True,
                                   nonlinear_kwargs=None):
    """Reuse Gamma(T_vertex); change pairing temperature at fixed active filling.

    Loads the explicit full Gamma, or reconstructs it from a validated converged
    K checkpoint when an older Gamma construction is present. No Newton solve
    is repeated. The original free band/gauge is retained; normal and superconducting
    chemical potentials are recomputed in the active-band number constraint.
    This is a fixed-vertex approximation, not a full recalculation at the new T.
    """
    temperature = float(temperature_eV)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Use a positive finite temperature_eV, e.g. 1e-5; exact T=0 is not implemented.')
    source = Path(source_directory).expanduser().resolve()
    config = json.loads((source / 'config.json').read_text())
    vertex_temperature = float(config['model']['temperature_eV'])
    system = make_system(config)  # Original temperature is essential for vertex validation.
    crpa = load_constrained_rpa(source / '01_screening' / 'constrained_rpa.npz', system)
    if progress:
        print(f'Loading saved full Gamma at kBT_vertex={vertex_temperature:g} eV; no parquet solve.')
    ph_problem = build_screened_problem(system, crpa,
        maximum_memory_mb=config['solver']['maximum_memory_mb'])
    archive = ensure_full_gamma(ph_problem, source / '02_vertex',
                               target_scale=config['continuation']['target_scale'])
    try:
        pair = extract_cooper_vertex(ph_problem, archive.block)
        gamma_metadata = dict(archive.metadata)
    finally:
        # All Cooper entries are copied; release the large map before the gap solve.
        archive.flat._mmap.close()
    del archive, ph_problem, crpa
    gc.collect()

    energies = system.energies[:, system.active_band_index]
    original_xi = energies - system.chemical_potential
    target = float(np.mean(expit(-system.beta * original_xi)))
    normal_mu = normal_mu_at_filling(energies, target, temperature)
    xi = energies - normal_mu
    problem = PairingProblem(system.coords, system.weight, 1 / temperature, target)
    options = dict(nonlinear_kwargs or {})
    seed = _saved_gap_seed(source / '03_superconductivity' / 'superconductivity.npz',
                           system.coords, gamma_metadata['source_signature'], target, gamma_metadata)
    if seed is not None:
        options.setdefault('initial_delta', seed)
    options.setdefault('progress', progress)
    if progress:
        print(f'Pairing kBT={temperature:g} eV ({temperature / KB_EV_PER_K:.6g} K); '
              f'fixed active filling={target:.12g}')
        print(f'Normal mu={normal_mu:.12g} eV; shift from vertex-temperature mu='
              f'{1e3 * (normal_mu - system.chemical_potential):+.6g} meV')
    nonlinear = solve_nonlinear_gap(problem, pair.interaction, xi, **options)
    if not nonlinear.converged or not nonlinear.linearized.converged:
        raise RuntimeError(f'Lower-temperature solution did not converge: {nonlinear.status}; '
                           f'gap residual={nonlinear.gap_residual:.3e}, '
                           f'filling residual={nonlinear.filling_residual:.3e}')

    model = deepcopy(config['model'])
    model['temperature_eV'] = temperature
    diagnostics = dict(pair.diagnostics)
    diagnostics.update(
        approximation='fixed full Gamma at vertex_temperature_eV; finite-T gap and number equations',
        vertex_temperature_eV=vertex_temperature, pairing_temperature_eV=temperature,
        temperature_eV=temperature, beta_per_eV=1 / temperature,
        source_directory=str(source), source_config_sha256=config_digest(config),
        source_gamma_file_sha256=gamma_metadata['gamma_file_sha256'],
        normal_state='free active band; no HF', number_constraint='original active-band filling fixed',
        quadrature_weight=float(system.weight), active_band_index=int(system.active_band_index),
        symmetry_tolerance=float(options.get('symmetry_tolerance', 1e-5)), hermitian_part_used=True,
        gap_tolerance=float(options.get('gap_tolerance', 1e-8)),
        filling_tolerance=float(options.get('filling_tolerance', 1e-11)),
        original_normal_mu_eV=float(system.chemical_potential),
        normal_mu_temperature_shift_eV=normal_mu - float(system.chemical_potential),
        source_active_filling=target, model_config=model,
        vertex_model_config=deepcopy(config['model']), hf_used=False,
        gap_seed='saved converged gap' if seed is not None and 'initial_delta' in options else 'linearized mode',
        fixed_vertex_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    result = SuperconductivityResult(problem, xi, normal_mu, nonlinear.linearized,
                                    nonlinear, diagnostics)
    minus = inversion_indices(problem)
    mean_xi = .5 * (nonlinear.centered_xi + nonlinear.centered_xi[minus])
    mismatch = .5 * (nonlinear.centered_xi - nonlinear.centered_xi[minus])
    energy = np.sqrt(mean_xi**2 + np.abs(nonlinear.delta)**2)
    min_excitation = float(np.min(np.minimum(np.abs(mismatch + energy), np.abs(mismatch - energy))))
    odd_defect = float(np.linalg.norm(nonlinear.delta + nonlinear.delta[minus]) /
                       max(np.linalg.norm(nonlinear.delta), np.finfo(float).tiny))
    destination = (Path(output_directory).expanduser().resolve() if output_directory is not None
                   else source / '04_fixed_vertex_temperature' / f'T{temperature:.12g}_eV')
    if destination == source or destination == source / '03_superconductivity':
        raise ValueError('Choose a separate destination to preserve the original superconductivity outputs.')
    summary = {
        'approximation': diagnostics['approximation'], 'exact_zero_temperature': False,
        'vertex_temperature_eV': vertex_temperature, 'pairing_temperature_eV': temperature,
        'pairing_temperature_K': temperature / KB_EV_PER_K,
        'source_directory': str(source), 'source_config_sha256': config_digest(config),
        'source_signature': gamma_metadata['source_signature'],
        'vertex_construction': gamma_metadata['construction'],
        'source_gamma_file_sha256': gamma_metadata['gamma_file_sha256'],
        'lambda_max': float(nonlinear.linearized.leading_eigenvalue),
        'classification': nonlinear.linearized.classification,
        'eigen_residual': float(nonlinear.linearized.residual),
        'status': nonlinear.status, 'converged': bool(nonlinear.converged),
        'is_superconducting': bool(nonlinear.is_superconducting),
        'max_gap_meV': 1e3 * nonlinear.max_gap,
        'active_filling': target, 'actual_filling': float(nonlinear.filling),
        'normal_mu_eV': normal_mu,
        'normal_mu_temperature_shift_meV': 1e3 * (normal_mu - system.chemical_potential),
        'superconducting_mu_shift_meV': 1e3 * nonlinear.mu_shift,
        'superconducting_mu_eV': normal_mu + nonlinear.mu_shift,
        'gap_residual': float(nonlinear.gap_residual),
        'filling_residual': float(nonlinear.filling_residual),
        'odd_gap_defect': odd_defect, 'iterations': int(nonlinear.iterations),
        'mesh_sampled_min_bdg_excitation_meV': 1e3 * min_excitation,
        'mesh_sampled_min_bdg_excitation_over_kBT': min_excitation / temperature,
    }
    destination.mkdir(parents=True, exist_ok=True)
    preserve_replaced_pairing_outputs(destination, gamma_metadata)
    save_superconductivity(destination / 'superconductivity.npz', result)
    write_json(destination / 'summary.json', summary)
    return system, result, summary, destination


def plot_order_parameter(system, result):
    """Actual nonlinear order parameter in meV, with one global plotting phase fixed."""
    import matplotlib.pyplot as plt
    if result.nonlinear is None or not result.nonlinear.converged:
        raise ValueError('A converged nonlinear result is required.')
    delta = result.nonlinear.delta.copy()
    if np.any(np.abs(delta) > 0):
        delta *= np.exp(-1j * np.angle(delta[np.argmax(np.abs(delta))]))
    delta *= 1e3
    limit = max(float(np.max(np.abs(delta.real))), float(np.max(np.abs(delta.imag))), 1e-15)
    max_amplitude = max(float(np.max(np.abs(delta))), 1e-15)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, values, title, cmap, vmin, vmax in (
        (axes[0], delta.real, r'Re $\Delta(k)$', 'RdBu_r', -limit, limit),
        (axes[1], delta.imag, r'Im $\Delta(k)$', 'RdBu_r', -limit, limit),
        (axes[2], np.abs(delta), r'$|\Delta(k)|$', 'viridis', 0, max_amplitude),
    ):
        artist = ax.scatter(system.points[:, 0], system.points[:, 1], c=values, cmap=cmap,
                            vmin=vmin, vmax=vmax, s=max(12, 4000 / len(values)))
        ax.set(title=title, xlabel=r'$k_x a$', ylabel=r'$k_y a$', aspect='equal')
        fig.colorbar(artist, ax=ax, label='meV')
    tv = result.diagnostics['vertex_temperature_eV']
    temperature = 1 / result.problem.beta
    fig.suptitle(f'Nonlinear order parameter: kBT = {temperature:g} eV ({temperature / KB_EV_PER_K:.3g} K)\n'
                 f'Fixed Gamma from kBT = {tv:g} eV; max |Delta| = {result.nonlinear.max_gap * 1e3:.6g} meV')
    return fig
