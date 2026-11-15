"""Independent routing, normalization, parity and superconductivity checks."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.special import expit

from parquet_superconductivity.pairing import (
    PairingProblem, extract_cooper_vertex, inversion_indices,
    linearized_gap_diagnostic, solve_nonlinear_gap, solve_superconductivity,
    save_superconductivity, cooper_susceptibility,
    nonlinear_cooper_susceptibility, bdg_electron_occupation,
)
from free_fermion_gamma.model import FreeFermionConfig, build_free_fermion_system
from static_parquet_memory.solver import build_problem


@pytest.fixture(scope="module")
def source_problem():
    system = build_free_fermion_system(FreeFermionConfig(
        radius=2, spacing=0.03, temperature_eV=0.005,
        target_particles_per_momentum=4.3, dielectric_constant=15.,
    ))
    return build_problem(system)


def test_tree_level_routing_and_half_factor_match_raw_rpa(source_problem):
    source = source_problem
    vertex = extract_cooper_vertex(source, source.bare)
    system = source.system
    coords = system.coords
    positions = {tuple(c): i for i, c in enumerate(coords)}
    minus = np.asarray([positions[tuple(-c)] for c in coords])
    spinors = system.spinors[:, :, system.active_band_index]
    # Independently construct the RPA notebook's raw-direct Cooper matrix.
    direct = (system.interaction(coords[None, :, :] - coords[:, None, :])
              * (spinors.conj() @ spinors.T)
              * (spinors[minus].conj() @ spinors[minus].T))
    odd_direct = 0.25 * (direct - direct[minus, :] - direct[:, minus]
                        + direct[np.ix_(minus, minus)])
    np.testing.assert_allclose(vertex.interaction, odd_direct, rtol=3e-13, atol=2e-12)
    np.testing.assert_allclose(vertex.raw_gamma, 2 * odd_direct, rtol=3e-13, atol=4e-12)
    assert max(v for k, v in vertex.diagnostics.items() if k.endswith("relative_error")) < 1e-12
    streamed = extract_cooper_vertex(source, lambda i: source.bare[i])
    packed = extract_cooper_vertex(source, source.bare_flat)
    np.testing.assert_array_equal(streamed.raw_gamma, vertex.raw_gamma)
    np.testing.assert_array_equal(packed.raw_gamma, vertex.raw_gamma)


def test_bad_crossing_is_rejected_before_projection(source_problem):
    broken = [block.copy() for block in source_problem.bare]
    zero = next(i for i, q in enumerate(source_problem.transfers) if np.all(q == 0))
    broken[zero][0, -1] += 1e5
    with pytest.raises(ValueError, match="symmetry tolerance"):
        extract_cooper_vertex(source_problem, broken)


def toy(weight=0.01, beta=1000.):
    problem = PairingProblem(np.asarray([[-1, 0], [0, 0], [1, 0]]), weight, beta, 0.5)
    mode = np.asarray([1., 0., -1.], dtype=complex) / np.sqrt(2.)
    interaction = -np.outer(mode, mode.conj())
    return problem, interaction


def test_odd_space_and_single_quadrature_factor():
    problem, interaction = toy()
    result = linearized_gap_diagnostic(problem, interaction, np.zeros(3))
    np.testing.assert_allclose(result.leading_eigenvalue, problem.weight * problem.beta / 4, rtol=1e-14)
    np.testing.assert_allclose(result.gap_mode, -result.gap_mode[inversion_indices(problem)], atol=1e-14)
    assert result.gap_mode[1] == 0
    scaled = linearized_gap_diagnostic(replace(problem, weight=2 * problem.weight), interaction, np.zeros(3))
    np.testing.assert_allclose(scaled.leading_eigenvalue, 2 * result.leading_eigenvalue)
    assert result.is_unstable and result.residual < 1e-13


def test_asymmetric_dispersion_susceptibility_and_bdg():
    left = np.asarray([-.03, .012, .01, .02])
    right = np.asarray([.01, -.012, -.03, -.02])
    beta = 80.
    chi = cooper_susceptibility(left, right, beta=beta)
    regular = np.abs(left + right) > 1e-12
    expected = (1 - expit(-beta * left[regular]) - expit(-beta * right[regular])) / (left + right)[regular]
    np.testing.assert_allclose(chi[regular], expected, rtol=1e-14)
    f = expit(-beta * left[~regular])
    np.testing.assert_allclose(chi[~regular], beta * f * (1-f), rtol=1e-14)
    np.testing.assert_allclose(nonlinear_cooper_susceptibility(left, right, np.zeros(4), beta=beta), chi, rtol=1e-14)
    np.testing.assert_allclose(bdg_electron_occupation(left, right, np.zeros(4), beta=beta), expit(-beta * left), atol=2e-16)
    # Independent diagonalization of each asymmetric 2x2 BdG Hamiltonian.
    delta = np.asarray([.008+.003j, .007, -.008-.003j, -.007])
    occupations = []
    for a, b, d in zip(left, right, delta):
        values, vectors = np.linalg.eigh([[a, d], [np.conj(d), -b]])
        occupations.append(np.sum(abs(vectors[0])**2 * expit(-beta * values)))
    np.testing.assert_allclose(bdg_electron_occupation(left, right, delta, beta=beta), occupations, atol=2e-16)


def test_normal_gate_and_manufactured_attractive_gap():
    problem, interaction = toy()
    stable = solve_nonlinear_gap(problem, -interaction, np.zeros(3))
    assert stable.status == "normal-state-stable" and stable.converged
    assert stable.iterations == 0 and np.all(stable.delta == 0)
    paired = solve_nonlinear_gap(problem, interaction, np.zeros(3), mixing=.5,
                                initial_gap=1e-4, gap_tolerance=1e-9)
    assert paired.converged and paired.is_superconducting
    assert paired.gap_residual < 1e-9 and abs(paired.filling_residual) < 1e-11
    assert paired.max_gap > 1e-3
    np.testing.assert_allclose(paired.delta, -paired.delta[::-1], atol=1e-15)
    # Separable model's independently reduced scalar gap equation.
    gap = paired.max_gap
    np.testing.assert_allclose(problem.weight * np.tanh(problem.beta * gap / 2) / (2 * gap), 1., rtol=2e-9)


def test_nonhermitian_vertex_is_not_silently_repaired():
    problem, interaction = toy()
    interaction[0, 2] += .1j
    with pytest.raises(ValueError, match="Hermiticity defect"):
        linearized_gap_diagnostic(problem, interaction, np.zeros(3))


def test_full_space_reference_kernel_agreement():
    # Explicit full-space P_- construction reproduces the RPA notebook's
    # kernel independently of the reduced-basis implementation.
    coords = np.asarray([(x, y) for y in range(-1, 2) for x in range(-1, 2)])
    xi = .001 * coords[:, 0] + .0003 * coords[:, 1] + .002 * np.sum(coords**2, axis=1)
    target = float(np.mean(expit(-100 * xi)))
    problem = PairingProblem(coords, .003, 100., target)
    rng = np.random.default_rng(49)
    raw = rng.normal(size=(len(coords), len(coords))) + 1j * rng.normal(size=(len(coords), len(coords)))
    interaction = raw + raw.conj().T
    got = linearized_gap_diagnostic(problem, interaction, xi)
    minus = inversion_indices(problem)
    projector = 0.5 * (np.eye(len(coords)) - np.eye(len(coords))[minus])
    chi = cooper_susceptibility(xi, xi[minus], beta=problem.beta)
    projected = projector @ interaction @ projector
    kernel = -problem.weight * np.sqrt(chi[:, None]) * projected * np.sqrt(chi[None, :])
    values, vectors = np.linalg.eigh(kernel)
    np.testing.assert_allclose(got.leading_eigenvalue, values[-1], rtol=1e-14)
    expected = -problem.weight * projected @ (np.sqrt(chi) * vectors[:, -1])
    expected /= np.max(np.abs(expected))
    expected *= np.conj(expected[np.argmax(abs(expected))]) / abs(expected[np.argmax(abs(expected))])
    np.testing.assert_allclose(got.gap_mode, expected, atol=1e-13)


def test_free_wrapper_and_saved_metadata(source_problem, tmp_path):
    source = source_problem
    vertex = extract_cooper_vertex(source, source.bare)
    result = solve_superconductivity(source.system, vertex, run_nonlinear=False)
    active_xi = source.system.energies[:, source.system.active_band_index] - source.system.chemical_potential
    np.testing.assert_array_equal(result.xi, active_xi)
    assert result.target_filling == pytest.approx(float(np.mean(expit(-source.system.beta * active_xi))))
    path = save_superconductivity(tmp_path / "pairing.npz", result)
    with np.load(path) as saved:
        metadata = json.loads(str(saved['metadata_json']))
        assert metadata['normal_state'] == "free bands; no HF"
        assert metadata['nonlinear_requested'] is False
        assert metadata['antisymmetrized_vertex_factor'] == .5
        np.testing.assert_array_equal(saved['pair_coords'], source.system.coords)
