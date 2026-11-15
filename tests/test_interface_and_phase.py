"""Portable controls and phase-analysis checks independent of the reference data."""
import json
from pathlib import Path
import numpy as np
import pytest
from parquet_superconductivity.config import default_config, point_config
from parquet_superconductivity.phase_winding import orbital_reference_gap, analyze_closed_curve
from parquet_superconductivity.run import main


def test_reference_controls_and_unit_conversion():
    expected = json.loads((Path(__file__).parents[1] / "configs/reference_D110_n6p375.json").read_text())
    assert default_config() == expected
    c = point_config(125, 7, mesh_radius=4)
    assert c["model"]["band_params"]["delta_d"] == .125
    assert c["model"]["carrier_density_cm2"] == 7e11
    assert c["model"]["spacing"] == .03
    with pytest.raises(ValueError):
        point_config(110, -1)
    with pytest.raises(ValueError):
        point_config(110, 6, mesh_radius=2.5)


def test_cli_preview_does_not_create_results(tmp_path, capsys):
    destination = tmp_path / "not_created"
    assert main(["--delta-d-mev", "125", "--density", "7", "--mesh-radius", "2",
                 "--preview", "--output", str(destination)]) == 0
    assert not destination.exists()
    output = capsys.readouterr().out
    assert '"delta_d_meV": 125.0' in output
    assert '"density_1e11_cm2": 7.0' in output


def test_orbital_reference_removes_bloch_rephasing():
    rng = np.random.default_rng(12)
    minus = np.array([2, 1, 0])
    u = rng.normal(size=(3, 2)) + 1j*rng.normal(size=(3, 2))
    delta = np.array([1+2j, 0, -1-2j])
    chi = rng.normal(size=3)
    a, _ = orbital_reference_gap(delta, u, minus)
    transformed = delta*np.exp(-1j*(chi+chi[minus]))
    b, _ = orbital_reference_gap(transformed, u*np.exp(1j*chi[:, None]), minus)
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_known_f_wave_winding_and_node_rejection():
    theta = np.linspace(0, 2*np.pi, 241)
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    delta = np.exp(-3j*theta)
    assert analyze_closed_curve(xy, delta, 1e-3)["winding"] == -3
    assert analyze_closed_curve(xy[::-1], delta[::-1], 1e-3)["winding"] == -3
    delta[50] = 0
    assert analyze_closed_curve(xy, delta, 1e-3)["winding"] is None
