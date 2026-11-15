r"""Static band-pair RPA screening for one free isospin block.

The density bubble can retain every ordered transition ``(n,m)`` among the
eight free ABCA bands, or the same active-band pair space used to evaluate
Gamma.  On the open hexagonal mesh,

.. math::

 \Pi(Q)=w\sum_{k,n,m}
 |\langle u_n(k)|u_m(k+Q)\rangle|^2
 \frac{f(E_m(k+Q))-f(E_n(k))}{E_m(k+Q)-E_n(k)},

and ``W_RPA(Q)=V0(Q)/(1-V0(Q) Pi(Q))``.  A validated table can be selected
independently for the Hartree and/or Fock leg of
:class:`free_fermion_gamma.gamma.GammaSector`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from .model import (
    FreeFermionSystem,
    centered_hex_coords,
    fermi,
    hex_norm,
)
from .gamma import (
    active_associated_pairs,
    active_only_pairs,
    all_band_pairs,
)


Array = np.ndarray
RPAScreening = Literal["full-8band", "active-only", "active-associated"]
SCHEMA = "free_fermion_one_isospin_band_pair_rpa_v2"
LEGACY_FULL_BAND_SCHEMA = "free_fermion_one_isospin_full_8band_rpa_v1"


def lindhard_divided_difference(
    final_energy: Array,
    initial_energy: Array,
    *,
    mu: float,
    temperature: float,
    degeneracy_tolerance: float = 1.0e-12,
) -> Array:
    """Return ``[f(Ef)-f(Ei)]/[Ef-Ei]`` with the exact static limit."""

    final, initial = np.broadcast_arrays(
        np.asarray(final_energy, dtype=float),
        np.asarray(initial_energy, dtype=float),
    )
    tolerance = float(degeneracy_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be finite and non-negative")
    denominator = final - initial
    numerator = fermi(final, mu, temperature) - fermi(initial, mu, temperature)
    regular = np.abs(denominator) > tolerance
    result = np.empty_like(denominator)
    np.divide(numerator, denominator, out=result, where=regular)
    midpoint_occupation = fermi(0.5 * (final + initial), mu, temperature)
    derivative = -midpoint_occupation * (1.0 - midpoint_occupation) / temperature
    result[~regular] = derivative[~regular]
    return result


@dataclass(frozen=True)
class HexInteractionTable:
    """A complete lookup table on the canonical ``H_radius`` transfer mesh."""

    radius: int
    coords: Array
    values: Array

    def __post_init__(self) -> None:
        radius = int(self.radius)
        coords = np.asarray(self.coords, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.complex128)
        expected_coords = centered_hex_coords(radius)
        if radius < 0 or not np.array_equal(coords, expected_coords):
            raise ValueError("interaction coordinates must be the canonical full hexagon")
        if values.shape != (len(coords),) or not np.all(np.isfinite(values)):
            raise ValueError("interaction values have an invalid shape or content")
        box = np.full((2 * radius + 1, 2 * radius + 1), -1, dtype=np.int64)
        box[coords[:, 0] + radius, coords[:, 1] + radius] = np.arange(len(coords))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "coords", np.ascontiguousarray(coords))
        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(self, "_index_box", box)

    def lookup(self, transfer_coords: Array) -> Array:
        raw = np.asarray(transfer_coords)
        if raw.shape[-1:] != (2,):
            raise ValueError("transfer coordinates must end in an axis of size 2")
        if raw.dtype.kind not in "iu":
            rounded = np.rint(raw)
            if not np.all(np.isfinite(raw)) or not np.array_equal(raw, rounded):
                raise ValueError("transfer coordinates must be integers")
        coords = np.asarray(raw, dtype=np.int64)
        if np.any(hex_norm(coords) > self.radius):
            raise ValueError("requested transfer lies outside the RPA table")
        indices = self._index_box[
            coords[..., 0] + self.radius,
            coords[..., 1] + self.radius,
        ]
        if np.any(indices < 0):
            raise RuntimeError("RPA transfer lookup table is incomplete")
        return self.values[indices]

    def __call__(self, transfer_coords: Array) -> Array:
        return self.lookup(transfer_coords)

    @property
    def inversion_conjugacy_defect(self) -> float:
        return float(np.max(np.abs(self.values - self.lookup(-self.coords).conj())))


@dataclass(frozen=True)
class FullBandRPAResult:
    """Validated one-isospin static RPA result for a selected band-pair set.

    The historical class name is retained for compatibility.  A full-eight-
    band result has all 64 ordered pairs; matched Gamma approximations have
    either one or 15 pairs.
    """

    system: FreeFermionSystem
    screening_pairs: Array
    transfer_coords: Array
    polarization_raw: Array
    polarization: Array
    intraband_polarization_raw: Array
    intraband_polarization: Array
    bare_interaction: Array
    denominator: Array
    screened_interaction: Array
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        pairs = np.asarray(self.screening_pairs, dtype=np.int64)
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or np.any((pairs < 0) | (pairs >= 8))
            or len({tuple(pair) for pair in pairs}) != len(pairs)
        ):
            raise ValueError("screening_pairs must be unique ordered pairs in 0..7")
        pair_set = {tuple(pair) for pair in pairs}
        if pair_set != {(m, n) for n, m in pair_set}:
            raise ValueError("screening_pairs must be closed under pair reversal")
        coords = np.asarray(self.transfer_coords, dtype=np.int64)
        expected = centered_hex_coords(2 * self.system.radius)
        if not np.array_equal(coords, expected):
            raise ValueError("RPA transfers must be the complete canonical H_(2M)")
        ntransfer = len(coords)
        arrays: dict[str, Array] = {}
        for name in (
            "polarization_raw",
            "polarization",
            "intraband_polarization_raw",
            "intraband_polarization",
            "bare_interaction",
            "denominator",
            "screened_interaction",
        ):
            value = np.asarray(getattr(self, name), dtype=np.complex128)
            if value.shape != (ntransfer,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must have finite shape (Ntransfer,)")
            arrays[name] = np.ascontiguousarray(value)
            object.__setattr__(self, name, arrays[name])
        opposite = _hex_indices(-coords, 2 * self.system.radius)
        expected_pi = 0.5 * (
            arrays["polarization_raw"]
            + arrays["polarization_raw"][opposite].conj()
        )
        expected_intra = 0.5 * (
            arrays["intraband_polarization_raw"]
            + arrays["intraband_polarization_raw"][opposite].conj()
        )
        if not np.allclose(arrays["polarization"], expected_pi, rtol=2e-13, atol=2e-13):
            raise ValueError("symmetrized selected-pair polarization is inconsistent")
        if not np.allclose(
            arrays["intraband_polarization"], expected_intra, rtol=2e-13, atol=2e-13
        ):
            raise ValueError("symmetrized intraband polarization is inconsistent")
        expected_bare = self.system.interaction(coords)
        if not np.allclose(arrays["bare_interaction"], expected_bare, rtol=2e-13, atol=2e-13):
            raise ValueError("RPA bare interaction differs from the free system")
        expected_denominator = 1.0 - expected_bare * arrays["polarization"]
        if not np.allclose(arrays["denominator"], expected_denominator, rtol=2e-13, atol=2e-13):
            raise ValueError("RPA denominator is inconsistent")
        expected_screened = expected_bare / expected_denominator
        if not np.allclose(
            arrays["screened_interaction"], expected_screened, rtol=2e-13, atol=2e-13
        ):
            raise ValueError("screened interaction is inconsistent")
        if np.max(np.abs(arrays["screened_interaction"].imag)) > 1.0e-9:
            raise ValueError("static RPA interaction is unexpectedly complex")
        table = HexInteractionTable(
            2 * self.system.radius, coords.copy(), arrays["screened_interaction"].copy()
        )
        if table.inversion_conjugacy_defect > 1.0e-9:
            raise ValueError("screened interaction violates inversion reciprocity")
        metadata = json.loads(json.dumps(dict(self.metadata), sort_keys=True))
        if (
            metadata.get("schema") not in (SCHEMA, LEGACY_FULL_BAND_SCHEMA)
            or metadata.get("hf_used") is not False
        ):
            raise ValueError("RPA metadata does not identify the free eight-band scheme")
        if metadata.get("screening_band_count") != 8:
            raise ValueError("RPA metadata does not identify the eight-band model")
        expected_pair_count = int(metadata.get("screening_pair_count", 64))
        if expected_pair_count != len(pairs):
            raise ValueError("RPA metadata screening-pair count is inconsistent")
        metadata_pairs = metadata.get("screening_pairs")
        if metadata.get("schema") == SCHEMA:
            if metadata_pairs is None:
                raise ValueError("selected-pair RPA metadata must record screening_pairs")
            if not np.array_equal(np.asarray(metadata_pairs, dtype=np.int64), pairs):
                raise ValueError("RPA metadata screening pairs are inconsistent")
            scheme = metadata.get("screening_scheme")
            expected_pairs = screening_pairs_for_scheme(self.system, scheme)
            if not np.array_equal(pairs, expected_pairs):
                raise ValueError("RPA screening scheme and pair array are inconsistent")
        elif not np.array_equal(pairs, all_band_pairs()):
            raise ValueError("legacy RPA schema is valid only for all 64 pairs")
        object.__setattr__(self, "screening_pairs", np.ascontiguousarray(pairs))
        object.__setattr__(self, "transfer_coords", np.ascontiguousarray(coords))
        object.__setattr__(self, "interaction", table)
        object.__setattr__(self, "metadata", metadata)

    @property
    def interband_polarization(self) -> Array:
        return self.polarization - self.intraband_polarization

    @property
    def minimum_abs_denominator(self) -> float:
        return float(np.min(np.abs(self.denominator)))

    @property
    def screening_scheme(self) -> str:
        """Canonical scheme name, including for legacy 64-pair caches."""

        return str(self.metadata.get("screening_scheme", "full-8band"))


def _hex_indices(coords: Array, radius: int) -> Array:
    canonical = centered_hex_coords(radius)
    box = np.full((2 * radius + 1, 2 * radius + 1), -1, dtype=np.int64)
    box[canonical[:, 0] + radius, canonical[:, 1] + radius] = np.arange(len(canonical))
    value = np.asarray(coords, dtype=np.int64)
    if np.any(hex_norm(value) > radius):
        raise ValueError("coordinates lie outside the requested hexagon")
    return box[value[..., 0] + radius, value[..., 1] + radius]


def _canonical_config(system: FreeFermionSystem) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(system.config), sort_keys=True))


def screening_pairs_for_scheme(
    system: FreeFermionSystem,
    screening: RPAScreening,
) -> Array:
    """Return the ordered free-band transitions retained in an RPA bubble."""

    if not isinstance(system, FreeFermionSystem):
        raise TypeError("system must be a FreeFermionSystem")
    if screening == "full-8band":
        pairs = all_band_pairs()
    elif screening == "active-only":
        pairs = active_only_pairs(system.active_band_index)
    elif screening == "active-associated":
        pairs = active_associated_pairs(system.active_band_index)
    else:
        raise ValueError(
            "screening must be 'full-8band', 'active-only', or "
            "'active-associated'"
        )
    # All public schemes are closed under n <-> m.  This makes the selected
    # density response reciprocal before the explicit Q/-Q symmetrization.
    pair_set = {tuple(pair) for pair in pairs}
    if pair_set != {(m, n) for n, m in pair_set}:
        raise RuntimeError("RPA screening pair set must be closed under reversal")
    return pairs


def compute_band_pair_rpa(
    system: FreeFermionSystem,
    *,
    screening: RPAScreening = "full-8band",
    degeneracy_tolerance: float = 1.0e-12,
    denominator_tolerance: float = 1.0e-12,
    symmetrize: bool = True,
    progress: bool = False,
) -> FullBandRPAResult:
    """Compute ``W_RPA`` from a selected ordered band-transition space."""

    if not isinstance(system, FreeFermionSystem):
        raise TypeError("system must be a FreeFermionSystem")
    if not symmetrize:
        raise ValueError(
            "symmetrize=False is not supported: Gamma requires an exactly "
            "reciprocal RPA interaction; the unsymmetrized bubble is retained "
            "separately as polarization_raw"
        )
    pairs = screening_pairs_for_scheme(system, screening)
    initial_bands = pairs[:, 0]
    final_bands = pairs[:, 1]
    diagonal_pairs = initial_bands == final_bands
    denominator_tolerance = float(denominator_tolerance)
    if not np.isfinite(denominator_tolerance) or denominator_tolerance <= 0.0:
        raise ValueError("denominator_tolerance must be finite and positive")
    transfers = centered_hex_coords(2 * system.radius)
    ntransfer = len(transfers)
    pi_raw = np.empty(ntransfer, dtype=np.complex128)
    pi_intra_raw = np.empty(ntransfer, dtype=np.complex128)
    coords = system.coords
    radius = system.radius
    index_box = np.full((2 * radius + 1, 2 * radius + 1), -1, dtype=np.int64)
    index_box[coords[:, 0] + radius, coords[:, 1] + radius] = np.arange(len(coords))
    started = time.time()
    interval = max(1, ntransfer // 20)

    for iq, transfer in enumerate(transfers):
        if not np.any(transfer):
            occupation = fermi(
                system.energies,
                system.chemical_potential,
                system.config.temperature_eV,
            )
            derivative = -occupation * (1.0 - occupation) / system.config.temperature_eV
            value = system.weight * np.sum(derivative[:, initial_bands[diagonal_pairs]])
            pi_raw[iq] = value
            pi_intra_raw[iq] = value
        else:
            shifted_coords = coords + transfer
            valid = hex_norm(shifted_coords) <= radius
            initial_rows = np.flatnonzero(valid)
            selected = shifted_coords[valid]
            final_rows = index_box[
                selected[:, 0] + radius,
                selected[:, 1] + radius,
            ]
            initial_energy = system.energies[initial_rows]
            final_energy = system.energies[final_rows]
            initial_spinor = system.spinors[initial_rows]
            final_spinor = system.spinors[final_rows]
            initial_selected = np.take(initial_spinor, initial_bands, axis=2)
            final_selected = np.take(final_spinor, final_bands, axis=2)
            form_factor = np.einsum(
                "kab,kab->kb",
                initial_selected.conj(),
                final_selected,
                optimize=True,
            )
            line = lindhard_divided_difference(
                np.take(final_energy, final_bands, axis=1),
                np.take(initial_energy, initial_bands, axis=1),
                mu=system.chemical_potential,
                temperature=system.config.temperature_eV,
                degeneracy_tolerance=degeneracy_tolerance,
            )
            contribution = np.abs(form_factor) ** 2 * line
            pi_raw[iq] = system.weight * np.sum(contribution)
            pi_intra_raw[iq] = system.weight * np.sum(contribution[:, diagonal_pairs])
        if progress and (iq == 0 or (iq + 1) % interval == 0 or iq + 1 == ntransfer):
            elapsed = time.time() - started
            remaining = (ntransfer - iq - 1) * elapsed / (iq + 1)
            print(
                f"  {screening} RPA {iq + 1:,}/{ntransfer:,}; "
                f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s",
                flush=True,
            )

    opposite = _hex_indices(-transfers, 2 * radius)
    raw_symmetry_error = float(np.max(np.abs(pi_raw - pi_raw[opposite].conj())))
    polarization = 0.5 * (pi_raw + pi_raw[opposite].conj())
    intraband = 0.5 * (pi_intra_raw + pi_intra_raw[opposite].conj())
    positive_max = float(np.max(polarization.real, initial=0.0))
    if positive_max > 1.0e-10:
        raise ValueError(
            f"static {screening} polarization must be non-positive; "
            f"max={positive_max:.3e}"
        )
    bare = np.asarray(system.interaction(transfers), dtype=np.complex128)
    denominator = 1.0 - bare * polarization
    minimum_denominator = float(np.min(np.abs(denominator)))
    if minimum_denominator < denominator_tolerance:
        raise np.linalg.LinAlgError(
            "RPA denominator is numerically singular; "
            f"minimum magnitude={minimum_denominator:.3e}"
        )
    screened = bare / denominator
    metadata = {
        "schema": SCHEMA,
        "calculation": "static one-isospin selected-band-pair free-fermion RPA",
        "hf_used": False,
        "isospin": system.config.isospin,
        "screening_species_count": 1,
        "screening_band_count": 8,
        "screening_scheme": screening,
        "screening_pair_count": len(pairs),
        "screening_pairs": pairs.tolist(),
        "ordered_band_transition_count": len(pairs),
        "chemical_potential_eV": system.chemical_potential,
        "temperature_eV": system.config.temperature_eV,
        "normalization": "open-hex integral weight abs(det(basis))/(2*pi)^2",
        "quadrature_weight": system.weight,
        "bubble_routing": "k,k+Q on the H_M intersection domain",
        "rpa_formula": "W_RPA(Q)=V0(Q)/(1-V0(Q)*Pi_selected_pairs(Q))",
        "gamma_usage": (
            "selectable screened input for Hartree V_H(-Q) and/or "
            "Fock V_F(p-q)"
        ),
        "symmetrized_q_and_minus_q": bool(symmetrize),
        "raw_polarization_inversion_error": raw_symmetry_error,
        "minimum_abs_rpa_denominator": minimum_denominator,
        "elapsed_seconds": time.time() - started,
        "config": _canonical_config(system),
    }
    return FullBandRPAResult(
        system=system,
        screening_pairs=pairs,
        transfer_coords=transfers,
        polarization_raw=pi_raw,
        polarization=polarization,
        intraband_polarization_raw=pi_intra_raw,
        intraband_polarization=intraband,
        bare_interaction=bare,
        denominator=denominator,
        screened_interaction=screened,
        metadata=metadata,
    )


def compute_full_band_rpa(
    system: FreeFermionSystem,
    *,
    degeneracy_tolerance: float = 1.0e-12,
    denominator_tolerance: float = 1.0e-12,
    symmetrize: bool = True,
    progress: bool = False,
) -> FullBandRPAResult:
    """Compatibility wrapper for all 64 transitions of one isospin."""

    return compute_band_pair_rpa(
        system,
        screening="full-8band",
        degeneracy_tolerance=degeneracy_tolerance,
        denominator_tolerance=denominator_tolerance,
        symmetrize=symmetrize,
        progress=progress,
    )


def compute_rpa_for_gamma_approximation(
    system: FreeFermionSystem,
    approximation: Literal["active-only", "active-associated"],
    **kwargs: Any,
) -> FullBandRPAResult:
    """Compute an RPA bubble in the same pair space used internally by Gamma."""

    if approximation not in ("active-only", "active-associated"):
        raise ValueError("approximation must be 'active-only' or 'active-associated'")
    return compute_band_pair_rpa(system, screening=approximation, **kwargs)


def save_full_band_rpa(result: FullBandRPAResult, path: str | Path) -> Path:
    """Atomically save the screening table and provenance in compressed NPZ."""

    if not isinstance(result, FullBandRPAResult):
        raise TypeError("result must be a FullBandRPAResult")
    output = Path(path).expanduser().resolve()
    if not output.suffix:
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(result.metadata, sort_keys=True)),
            screening_pairs=result.screening_pairs,
            transfer_coords=result.transfer_coords,
            polarization_raw=result.polarization_raw,
            polarization=result.polarization,
            intraband_polarization_raw=result.intraband_polarization_raw,
            intraband_polarization=result.intraband_polarization,
            bare_interaction=result.bare_interaction,
            denominator=result.denominator,
            screened_interaction=result.screened_interaction,
        )
    temporary.replace(output)
    return output


def load_full_band_rpa(
    path: str | Path,
    system: FreeFermionSystem,
) -> FullBandRPAResult:
    """Load an RPA table and require exact agreement with the current free input."""

    source = Path(path).expanduser().resolve()
    required = {
        "metadata_json",
        "transfer_coords",
        "polarization_raw",
        "polarization",
        "intraband_polarization_raw",
        "intraband_polarization",
        "bare_interaction",
        "denominator",
        "screened_interaction",
    }
    with np.load(source, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise KeyError(f"RPA archive is missing {sorted(missing)}")
        metadata = json.loads(str(saved["metadata_json"].item()))
        arrays = {name: np.asarray(saved[name]) for name in required if name != "metadata_json"}
        if "screening_pairs" in saved.files:
            screening_pairs = np.asarray(saved["screening_pairs"], dtype=np.int64)
        elif metadata.get("schema") == LEGACY_FULL_BAND_SCHEMA:
            screening_pairs = all_band_pairs()
        else:
            raise KeyError("selected-pair RPA archive is missing screening_pairs")
    current_config = _canonical_config(system)
    cached_config = metadata.get("config")
    if cached_config != current_config:
        cached_dict = cached_config if isinstance(cached_config, dict) else {}
        differences = []
        for key in sorted(set(cached_dict) | set(current_config)):
            if cached_dict.get(key) != current_config.get(key):
                differences.append(
                    f"{key}: cached={cached_dict.get(key)!r}, "
                    f"current={current_config.get(key)!r}"
                )
        detail = "; ".join(differences) or "configuration metadata differ"
        raise ValueError(
            "RPA archive was built from a different free-system configuration "
            f"({detail}). Recompute it or use a cache path for the current parameters."
        )
    if not np.isclose(
        float(metadata.get("chemical_potential_eV", np.nan)),
        system.chemical_potential,
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise ValueError("RPA archive chemical potential differs from the free system")
    return FullBandRPAResult(
        system=system,
        screening_pairs=screening_pairs,
        metadata=metadata,
        **arrays,
    )


__all__ = [
    "FullBandRPAResult",
    "HexInteractionTable",
    "RPAScreening",
    "compute_band_pair_rpa",
    "compute_full_band_rpa",
    "compute_rpa_for_gamma_approximation",
    "lindhard_divided_difference",
    "load_full_band_rpa",
    "save_full_band_rpa",
    "screening_pairs_for_scheme",
]
