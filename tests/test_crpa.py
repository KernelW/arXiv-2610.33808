"""Physics and cache checks for the free-band constrained RPA stage."""

from dataclasses import replace

import numpy as np
import pytest

from parquet_superconductivity.crpa import (
    compute_constrained_rpa,
    constrained_pairs,
    load_constrained_rpa,
    save_constrained_rpa,
)
from free_fermion_gamma.model import (
    ABCAParams,
    FreeFermionConfig,
    build_free_fermion_system,
    fermi,
)
from free_fermion_gamma.rpa import compute_band_pair_rpa


@pytest.fixture(scope="module")
def system():
    # Thermal broadening keeps the tiny test mesh useful while testing actual
    # ABCA states and finite density rather than synthetic identity spinors.
    return build_free_fermion_system(FreeFermionConfig(
        radius=2, spacing=0.03, temperature_eV=0.005,
        target_particles_per_momentum=4.3,
        band_params=ABCAParams(delta_d=0.13),
    ))


@pytest.fixture(scope="module")
def result(system):
    return compute_constrained_rpa(system)


def test_exactly_one_pair_is_excluded():
    for active in (0, 4, 7):
        pairs = {tuple(pair) for pair in constrained_pairs(active)}
        assert len(pairs) == 63
        assert (active, active) not in pairs
        assert pairs == {(m, n) for n, m in pairs}
        assert sum(n == active or m == active for n, m in pairs) == 14


def test_full_and_active_match_independent_reference(system, result):
    full = compute_band_pair_rpa(system, screening="full-8band")
    active = compute_band_pair_rpa(system, screening="active-only")
    np.testing.assert_allclose(result.polarization_full, full.polarization.real, rtol=3e-14, atol=1e-17)
    np.testing.assert_allclose(result.polarization_active, active.polarization.real, rtol=3e-14, atol=1e-17)
    np.testing.assert_allclose(
        result.polarization_rest,
        full.polarization.real - active.polarization.real,
        rtol=3e-13, atol=1e-17,
    )


def test_manual_finite_q_sum_retains_mixed_pairs(system, result):
    # Explicit scalar loops independent of the vectorized overlap contractions
    # and reference polarization implementation.
    q = (1, 0)
    index = {tuple(coord): row for row, coord in enumerate(system.coords)}
    active = system.active_band_index
    total = 0.0
    remote_remote = 0.0
    for k, coord in enumerate(system.coords):
        final = index.get(tuple(coord + q))
        if final is None:
            continue
        for n in range(8):
            for m in range(8):
                if (n, m) == (active, active):
                    continue
                ei = system.energies[k, n]
                ef = system.energies[final, m]
                fi = float(fermi(ei, system.chemical_potential, system.config.temperature_eV))
                ff = float(fermi(ef, system.chemical_potential, system.config.temperature_eV))
                if abs(ef - ei) <= 1e-12:
                    mid = float(fermi((ef + ei) / 2, system.chemical_potential, system.config.temperature_eV))
                    line = -mid * (1 - mid) / system.config.temperature_eV
                else:
                    line = (ff - fi) / (ef - ei)
                overlap = np.vdot(system.spinors[k, :, n], system.spinors[final, :, m])
                value = system.weight * abs(overlap) ** 2 * line
                total += value
                if n != active and m != active:
                    remote_remote += value
    iq = np.flatnonzero(np.all(result.transfer_coords == q, axis=1))[0]
    np.testing.assert_allclose(result.polarization_rest[iq], total, rtol=1e-13, atol=1e-17)
    assert abs(total - remote_remote) > 1e-8


def test_sequential_screening_recovers_full_rpa(result):
    constrained = result.screened_interaction
    sequential = constrained / (1.0 - constrained * result.polarization_active)
    full = result.bare_interaction / (1.0 - result.bare_interaction * result.polarization_full)
    np.testing.assert_allclose(sequential, full, rtol=4e-15, atol=1e-13)


def test_symmetry_static_q0_and_screening_bounds(system, result):
    np.testing.assert_array_equal(
        result.interaction.lookup(-result.transfer_coords).real, result.screened_interaction,
    )
    assert np.all(result.polarization_rest <= 0)
    assert np.all(result.screened_interaction > 0)
    assert np.all(result.screened_interaction <= result.bare_interaction)
    zero = np.flatnonzero(np.all(result.transfer_coords == 0, axis=1))[0]
    occupation = fermi(system.energies, system.chemical_potential, system.config.temperature_eV)
    derivative = -occupation * (1 - occupation) / system.config.temperature_eV
    remote = [n for n in range(8) if n != system.active_band_index]
    np.testing.assert_allclose(result.polarization_rest[zero], system.weight * derivative[:, remote].sum(), rtol=1e-14)
    assert result.metadata["hf_used"] is False
    assert result.metadata["screening_species_count"] == 1
    assert result.metadata["spin_valley_degeneracy_factor"] == 1
    assert len(result.metadata["model_source_sha256"]) == 64


def test_cache_roundtrip_and_mismatched_input(system, result, tmp_path):
    path = save_constrained_rpa(result, tmp_path / "crpa")
    loaded = load_constrained_rpa(path, system)
    np.testing.assert_array_equal(loaded.screened_interaction, result.screened_interaction)
    assert loaded.metadata == result.metadata
    different = build_free_fermion_system(replace(system.config, temperature_eV=0.006))
    with pytest.raises(ValueError, match="provenance"):
        load_constrained_rpa(path, different)


def test_radius_zero_exact_thermal_limit():
    small = build_free_fermion_system(FreeFermionConfig(
        radius=0, spacing=0.03, temperature_eV=0.005, target_particles_per_momentum=4.3,
    ))
    result = compute_constrained_rpa(small)
    assert result.transfer_coords.shape == (1, 2)
    np.testing.assert_allclose(
        result.polarization_full,
        result.polarization_active + result.polarization_rest,
        rtol=1e-14,
    )
