"""Source replacement, nonlinear identities and safe checkpoint tests."""
from dataclasses import replace
import json

import numpy as np
import pytest

from parquet_superconductivity.crpa import compute_constrained_rpa
from parquet_superconductivity.parquet import (build_screened_problem, solve_screened_parquet,
    reconstruct_full_gamma, load_full_gamma, load_checkpoint, save_checkpoint)
from free_fermion_gamma.model import ABCAParams, FreeFermionConfig, build_free_fermion_system
from static_parquet_memory.gamma_ph_only import evaluate_gamma_ph
from static_parquet_memory.solver import build_problem


@pytest.fixture(scope='module')
def physical():
    system = build_free_fermion_system(FreeFermionConfig(
        radius=1, spacing=.05, temperature_eV=.005,
        dielectric_constant=15., target_particles_per_momentum=4.3,
        band_params=ABCAParams(delta_d=.13)))
    crpa = compute_constrained_rpa(system)
    return system, crpa, build_screened_problem(system, crpa, maximum_memory_mb=256)


@pytest.fixture(scope='module')
def solved(physical):
    result, info = solve_screened_parquet(physical[2], target_scale=.05, starting_scale=.025,
        stages=[.025, .05], tolerance=1e-10, maximum_memory_mb=256,
        krylov_restart=8, krylov_maximum_iterations=6)
    assert result.converged and info['target_reached']
    return result


def test_crpa_changes_equation_source_and_has_exact_antisymmetry(physical):
    system, crpa, problem = physical
    old = build_problem(system, maximum_memory_mb=256, memory_profile='gamma-ph')
    assert np.linalg.norm(problem.bare_flat - old.bare_flat) > 1e-5
    np.testing.assert_allclose(problem.bare_flat[problem.crossing_permutation],
                               -problem.bare_flat, atol=1e-10)
    # A scalar orbital contraction checks exchange orientation and both screened legs.
    iq = problem._sector_indices[(1, 0)]
    row, col = 0, len(problem.domain_coords[iq]) - 1
    a = system.active_band_index
    q, p = problem.domain_coords[iq][[row, col]]
    left = system.spinors[problem.left_rows[iq], :, a]
    right = system.spinors[problem.right_rows[iq], :, a]
    expected = (crpa.interaction((-1, 0)) * np.vdot(left[row], right[row])
                * np.vdot(right[col], left[col])
                - crpa.interaction(p - q) * np.vdot(left[row], left[col])
                * np.vdot(right[col], right[row]))
    np.testing.assert_allclose(problem.bare[iq][row, col], expected, rtol=2e-14)
    assert problem.source_metadata['hf_used'] is False


def test_newton_root_and_full_crossing_bse_identity(solved, tmp_path):
    problem = solved.problem
    ev = evaluate_gamma_ph(problem, solved.kernel, interaction_scale=.05,
                           want_jacobian=False, memory_mode='streamed')
    assert ev.relative_residual < 1e-10
    archive = reconstruct_full_gamma(solved, tmp_path, target_scale=.05)
    gamma = archive.blocks
    # Phi is defined by its own BSE. Gamma_explicit - K equals Phi - R,
    # so substituting Gamma - K would hide the finite-root residual.
    phi = tuple(np.linalg.solve(np.eye(len(k)) - k * b[None, :],
                               (k * b[None, :]) @ k)
                for k, b in zip(solved.kernel, problem.bubble))
    cross = problem.crossing(phi)
    for index, (g, k, b, source, crossed) in enumerate(zip(
            gamma, solved.kernel, problem.bubble, problem.bare, cross)):
        response = np.eye(len(k)) - k * b[None, :]
        residual = k - .05 * source + crossed
        old_bse_gamma = np.linalg.solve(response, k)
        np.testing.assert_allclose(g - k - (k * b[None, :]) @ g,
                                   -response @ residual, rtol=1e-3, atol=2e-12)
        np.testing.assert_allclose(g - old_bse_gamma, -residual,
                                   rtol=1e-3, atol=2e-12)
        np.testing.assert_allclose(g, .05 * source - crossed + phi[index],
                                   rtol=2e-13, atol=2e-13)
    assert archive.metadata['crossing_defect'] < 1e-12
    restored = load_full_gamma(problem, tmp_path, target_scale=.05)
    np.testing.assert_array_equal(restored.flat, archive.flat)
    with pytest.raises(ValueError, match='target'):
        load_full_gamma(problem, tmp_path, target_scale=1.)
    writable = np.load(restored.path, mmap_mode='r+')
    writable[0] += .1
    writable.flush()
    writable._mmap.close()
    with pytest.raises(ValueError, match='checksum'):
        load_full_gamma(problem, tmp_path, target_scale=.05)
    restored.flat._mmap.close()
    archive.flat._mmap.close()


def test_resume_validates_actual_equation_and_source(solved, tmp_path):
    save_checkpoint(solved, tmp_path, target_scale=.05, tolerance=1e-10)
    restored = load_checkpoint(solved.problem, tmp_path, tolerance=1e-10, target_scale=.05)
    np.testing.assert_array_equal(solved.problem.pack(restored.kernel),
                                  solved.problem.pack(solved.kernel))
    resumed, info = solve_screened_parquet(solved.problem, target_scale=.05,
        checkpoint_directory=tmp_path, tolerance=1e-10, maximum_memory_mb=256)
    assert info['target_reached'] and not info['attempts']
    metadata = json.loads((tmp_path / 'checkpoint.json').read_text())
    flat = np.load(tmp_path / metadata['kernel_file'], mmap_mode='r+')
    flat[0] += .1
    flat.flush()
    flat._mmap.close()
    with pytest.raises(ValueError, match='not a converged solution'):
        load_checkpoint(solved.problem, tmp_path, tolerance=1e-10, target_scale=.05)
    metadata['source_signature'] = 'wrong source'
    (tmp_path / 'checkpoint.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='signature differs'):
        load_checkpoint(solved.problem, tmp_path, target_scale=.05)


def test_failed_or_subtarget_stage_cannot_feed_pairing(solved, tmp_path):
    failed = replace(solved, converged=False, status='maximum_iterations')
    with pytest.raises(ValueError, match='only converged'):
        save_checkpoint(failed, tmp_path, target_scale=1.)
    with pytest.raises(ValueError, match='failed stage'):
        reconstruct_full_gamma(failed, tmp_path)
    with pytest.raises(ValueError, match='target coupling'):
        reconstruct_full_gamma(solved, tmp_path, target_scale=1.)
