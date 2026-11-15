"""Phase of a saved nonlinear gap in a nonvanishing orbital reference gauge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import config_digest, make_system
from .crpa import load_constrained_rpa
from .parquet import FULL_GAMMA_CONSTRUCTION
from .pairing import _validate_production_gamma_metadata


def orbital_reference_gap(delta, spinors, minus, orbital=0, minimum_overlap=1e-8):
    """Fix u_orbital(k) positive real; phase equals u_a(k)u_a(-k)Delta(k)."""
    anchor = np.asarray(spinors)[:, orbital]
    if not np.all(np.isfinite(anchor)) or np.min(np.abs(anchor)) <= minimum_overlap:
        raise ValueError('The chosen orbital vanishes: a single reference gauge is not valid.')
    phase = anchor / np.abs(anchor)
    return np.asarray(delta, complex) * phase * phase[minus], float(np.min(np.abs(anchor)))


def analyze_closed_curve(points, values, amplitude_floor):
    """CCW winding and exact zero proximity for a piecewise-linear complex curve.

    Points and complex values must include the repeated endpoint. This determines
    the winding of this interpolation, not convergence of the underlying mesh.
    """
    points = np.asarray(points, float)
    values = np.asarray(values, complex)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
            or values.shape != (len(points),) or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(values))):
        raise ValueError('A finite closed curve with one complex value per vertex is required.')
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    if (np.linalg.norm(points[-1] - points[0]) > 1e-9 * scale
            or abs(values[-1] - values[0]) > 1e-9 * max(float(np.abs(values).max()), 1e-30)):
        raise ValueError('Open contours cannot be assigned a winding.')
    points, values = points.copy(), values.copy()
    points[-1], values[-1] = points[0], values[0]
    area = .5 * np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
    if abs(area) <= np.finfo(float).eps * scale**2:
        raise ValueError('Degenerate closed contour.')
    if area < 0:
        points, values = points[::-1].copy(), values[::-1].copy()
    # Start at the rightmost contour vertex, then traverse counterclockwise.
    start = int(np.argmax(points[:-1, 0]))
    points = np.roll(points[:-1], -start, axis=0)
    values = np.roll(values[:-1], -start)
    points = np.vstack((points, points[0]))
    values = np.r_[values, values[0]]
    dz = np.diff(values)
    denominator = np.abs(dz)**2
    t = np.divide(-np.real(values[:-1].conj() * dz), denominator,
                  out=np.zeros_like(denominator), where=denominator > 0)
    minimum = float(np.min(np.abs(values[:-1] + np.clip(t, 0, 1) * dz)))
    increments = np.angle(values[1:] * values[:-1].conj())
    turns = np.r_[0., np.cumsum(increments)] / (2 * np.pi)
    raw_winding = float(turns[-1])
    max_step = float(np.max(np.abs(increments)))
    reliable = bool(minimum > amplitude_floor and max_step < np.pi / 2
                    and abs(raw_winding - round(raw_winding)) < 1e-6)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.r_[0., np.cumsum(lengths)]
    return {
        'points': points, 'delta': values, 'arc_fraction': arclength / arclength[-1],
        'phase_turns': turns, 'area': abs(float(area)),
        'winding': int(round(raw_winding)) if reliable else None,
        'winding_sum': raw_winding, 'resolved_in_linear_interpolation': reliable,
        'min_gap_meV': 1e3 * minimum, 'max_phase_step_rad': max_step,
        'contour_vertices': len(points) - 1,
    }


def saved_phase_winding(gap_archive, *, relative_node_threshold=1e-3):
    """Load saved low-T gap, validate its Bloch gauge, and analyze normal FS loops."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if not 0 < relative_node_threshold < 1:
        raise ValueError('relative_node_threshold must lie between zero and one.')
    gap_archive = Path(gap_archive).expanduser().resolve()
    with np.load(gap_archive, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json'].item()))
        coords, xi, delta = (saved[key].copy() for key in ('pair_coords', 'xi', 'delta'))
    if not metadata.get('converged') or not metadata.get('is_superconducting'):
        raise ValueError('A converged nonzero nonlinear superconducting gap is required.')
    if metadata.get('vertex_construction') != FULL_GAMMA_CONSTRUCTION:
        raise ValueError('This gap uses a legacy vertex construction. Recompute superconductivity '
                         'with Gamma=Lambda_irr-C Phi+Phi before plotting phase winding.')
    _validate_production_gamma_metadata(metadata.get('source_gamma_metadata', {}))
    source = Path(metadata['source_directory'])
    config = json.loads((source / 'config.json').read_text())
    if config_digest(config) != metadata['source_config_sha256']:
        raise ValueError('The saved gap source configuration has changed.')
    system = make_system(config)
    crpa = load_constrained_rpa(source / '01_screening' / 'constrained_rpa.npz', system)
    gamma_metadata = json.loads((source / '02_vertex' / 'full_gamma.json').read_text())
    _validate_production_gamma_metadata(gamma_metadata)
    if (gamma_metadata['source_signature'] != metadata['source_signature']
            or gamma_metadata['gamma_file_sha256'] != metadata['source_gamma_file_sha256']
            or gamma_metadata['gamma_file_sha256']
                != metadata['source_gamma_metadata'].get('gamma_file_sha256')
            or gamma_metadata['source_metadata']['crpa_metadata']['system_fingerprint_sha256']
                != crpa.metadata['system_fingerprint_sha256']):
        raise ValueError('The reconstructed Bloch gauge does not match the saved gap source.')
    if not np.array_equal(coords, system.coords):
        raise ValueError('Saved gap coordinates do not match the source band basis.')
    if (xi.shape != (len(coords),) or delta.shape != xi.shape
            or not np.all(np.isfinite(xi)) or not np.all(np.isfinite(delta))):
        raise ValueError('Invalid saved energy or gap arrays.')
    expected_xi = system.energies[:, system.active_band_index] - metadata['normal_mu_eV']
    if not np.allclose(xi, expected_xi, rtol=0, atol=1e-13):
        raise ValueError('Saved normal energies disagree with the source band and chemical potential.')
    index = {tuple(c): i for i, c in enumerate(coords)}
    minus = np.array([index[tuple(-c)] for c in coords])
    delta_ref, minimum_overlap = orbital_reference_gap(
        delta, system.spinors[:, :, system.active_band_index], minus)
    # Only an overall phase is chosen here; this does not change any winding.
    delta_ref *= np.exp(-1j * np.angle(delta_ref[np.argmax(np.abs(delta_ref))]))
    amplitude_floor = relative_node_threshold * float(np.max(np.abs(delta_ref)))
    tri = mtri.Triangulation(system.points[:, 0], system.points[:, 1])
    real = mtri.LinearTriInterpolator(tri, delta_ref.real)
    imag = mtri.LinearTriInterpolator(tri, delta_ref.imag)
    fig, ax = plt.subplots()
    try:
        contour = ax.tricontour(tri, xi, levels=[0.])
        segments = [np.asarray(s).copy() for s in contour.allsegs[0] if len(s) > 1]
    finally:
        plt.close(fig)
    loops, skipped = [], []
    for segment in segments:
        scale = max(float(np.ptp(segment, axis=0).max()), 1e-12)
        if len(segment) < 4 or np.linalg.norm(segment[-1] - segment[0]) > 1e-9 * scale:
            skipped.append({'reason': 'open or degenerate normal-state Fermi contour',
                            'vertices': len(segment)})
            continue
        z = real(segment[:, 0], segment[:, 1]) + 1j * imag(segment[:, 0], segment[:, 1])
        if np.any(np.ma.getmaskarray(z)):
            raise ValueError('A Fermi contour lies outside the gap interpolation domain.')
        loops.append(analyze_closed_curve(segment, np.asarray(z), amplitude_floor))
    loops.sort(key=lambda loop: loop['area'], reverse=True)
    for i, loop in enumerate(loops):
        loop['label'] = f'FS {i + 1}'
    if not loops:
        raise ValueError('No closed normal-state Fermi contours were found on this momentum domain.')
    scalar_keys = ('label', 'winding', 'winding_sum', 'resolved_in_linear_interpolation',
                   'min_gap_meV', 'max_phase_step_rad', 'contour_vertices', 'area')
    report = {
        'gap_archive': str(gap_archive),
        'gap_archive_sha256': hashlib.sha256(gap_archive.read_bytes()).hexdigest(),
        'vertex_construction': gamma_metadata['construction'],
        'source_gamma_file_sha256': gamma_metadata['gamma_file_sha256'],
        'pairing_temperature_eV': metadata['temperature_eV'],
        'vertex_temperature_eV': metadata['vertex_temperature_eV'],
        'gauge': 'A1-positive band gauge; phase of the A1-A1 orbital pair component',
        'minimum_A1_overlap': minimum_overlap,
        'contours': 'normal-state xi(k)=0 at the saved pairing temperature and normal chemical potential',
        'orientation': 'counterclockwise for every loop, including inner loops',
        'interpolation': 'piecewise linear complex gap on the original triangular mesh',
        'relative_node_threshold': relative_node_threshold,
        'node_threshold_meV': 1e3 * amplitude_floor,
        'interpretation': 'Phase winding in the stated gauge; not a Chern-number calculation. Mesh convergence untested.',
        'skipped_contours': skipped,
        'loops': [{key: loop[key] for key in scalar_keys} for loop in loops],
    }
    return system.points.copy(), delta_ref, loops, report


def plot_saved_phase_winding(gap_archive, *, output_directory=None, relative_node_threshold=1e-3):
    """Return figure/report and save a PNG, JSON diagnostics, and NPZ curve data."""
    import matplotlib.pyplot as plt

    points, delta, loops, report = saved_phase_winding(
        gap_archive, relative_node_threshold=relative_node_threshold)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = ['#10a6a6', '#e78521', '#7755bb', '#33994a']
    visible = np.abs(delta) > report['node_threshold_meV'] / 1e3
    artist = axes[0].scatter(points[visible, 0], points[visible, 1], c=np.angle(delta[visible]),
        s=27, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0].scatter(points[~visible, 0], points[~visible, 1], c='gray', marker='x', s=24)
    cb = fig.colorbar(artist, ax=axes[0], ticks=[-np.pi, 0, np.pi], shrink=.8)
    cb.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    cb.set_label(r'$\arg\Delta_{\mathrm{A1+}}$')
    axes[0].set(xlabel=r'$k_x a$', ylabel=r'$k_y a$', aspect='equal', title='Phase on the calculated mesh')
    arrays = {'points': points, 'delta_A1_reference_eV': delta}
    for i, loop in enumerate(loops):
        color = colors[i % len(colors)]
        p, z = loop['points'], loop['delta']
        axes[0].plot(p[:, 0], p[:, 1], color=color, lw=2, label=loop['label'])
        axes[0].plot(p[0, 0], p[0, 1], 'o', color=color, ms=4)
        # Densify each original segment for a faithful complex-interpolation plot.
        t = np.linspace(0, 1, 12, endpoint=False)
        dense_z = (z[:-1, None] + np.diff(z)[:, None] * t).ravel()
        s = loop['arc_fraction']
        dense_s = (s[:-1, None] + np.diff(s)[:, None] * t).ravel()
        dense_z, dense_s = np.r_[dense_z, z[-1]], np.r_[dense_s, 1.]
        phase_turns = (np.unwrap(np.angle(dense_z)) - np.angle(z[0])) / (2 * np.pi)
        label = f"{loop['label']}: w = {loop['winding']}" if loop['winding'] is not None else f"{loop['label']}: unresolved"
        axes[1].plot(dense_s, phase_turns, color=color, lw=2, label=label)
        axes[2].plot(dense_s, 1e3 * np.abs(dense_z), color=color, lw=2,
                     label=f"{loop['label']}: min {loop['min_gap_meV']:.3g} meV")
        for key in ('points', 'delta', 'arc_fraction', 'phase_turns'):
            arrays[f'loop_{i + 1}_{key}'] = loop[key]
    axes[0].legend(loc='lower left', fontsize=9)
    axes[1].set(title='Unwrapped phase along each loop', ylabel=r'$[\phi(s)-\phi(0)]/(2\pi)$')
    axes[2].set(title='Gap magnitude along each loop', ylabel=r'$|\Delta_{\mathrm{A1+}}|$ (meV)')
    axes[2].axhline(report['node_threshold_meV'], color='gray', ls=':', lw=1, label='Node flag threshold')
    for ax in axes[1:]:
        ax.set(xlabel='Counterclockwise arc length / loop length', xlim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle(f"Nonlinear gap phase | kBT = {report['pairing_temperature_eV']:g} eV | A1-positive reference gauge\n"
                 'Normal-state Fermi contours; winding of the finite-grid interpolation (not a Chern number)', fontsize=12)
    destination = (Path(output_directory) if output_directory is not None else Path(gap_archive).parent)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / 'phase_winding.png', dpi=180)
    (destination / 'phase_winding.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(destination / 'phase_winding.npz', metadata_json=json.dumps(report), **arrays)
    return fig, report
