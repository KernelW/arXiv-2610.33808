r"""Constrained static RPA for the free eight-band ABCA model.

The active band is excluded only when *both* ends of a density bubble are
active. Thus Pi_rest = Pi_full - Pi_active retains 63 ordered transitions,
including the fourteen mixed active/remote transitions. Its direct sum avoids
cancellation in Pi_full - Pi_active. The convention is Pi <= 0 and
V_RPA^c = V0 / (1 - V0 Pi_rest), for a single isospin on the open hexagon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np



from free_fermion_gamma.model import (  # noqa: E402
    FreeFermionSystem,
    centered_hex_coords,
    fermi,
    hex_norm,
)
from free_fermion_gamma.rpa import (  # noqa: E402
    HexInteractionTable,
    lindhard_divided_difference,
)
import free_fermion_gamma.model as _reference_model  # noqa: E402


SCHEMA = "free_abca_one_isospin_constrained_rpa_v1"


def _canonical_config(system: FreeFermionSystem) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(system.config), sort_keys=True))


def _system_fingerprint(system: FreeFermionSystem) -> str:
    """Fingerprint the exact free eigensystem and quadrature used by a cache."""

    digest = hashlib.sha256()
    digest.update(json.dumps(_canonical_config(system), sort_keys=True).encode())
    for value in (
        system.coords, system.basis, system.energies, system.spinors,
        np.asarray([system.chemical_potential, system.weight]),
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _model_source_fingerprint() -> str:
    return hashlib.sha256(Path(_reference_model.__file__).read_bytes()).hexdigest()


def _opposite_indices(coords: np.ndarray, radius: int) -> np.ndarray:
    index_box = np.full((2 * radius + 1, 2 * radius + 1), -1, dtype=np.int64)
    index_box[coords[:, 0] + radius, coords[:, 1] + radius] = np.arange(len(coords))
    return index_box[-coords[:, 0] + radius, -coords[:, 1] + radius]


def constrained_pairs(active_band_index: int) -> np.ndarray:
    """All ordered eight-band pairs except (active, active), in row order."""

    active = int(active_band_index)
    if active != active_band_index or not 0 <= active < 8:
        raise ValueError("active_band_index must be an integer in 0..7")
    return np.asarray(
        [(n, m) for n in range(8) for m in range(8) if (n, m) != (active, active)],
        dtype=np.int64,
    )


@dataclass(frozen=True)
class ConstrainedRPAResult:
    """Constrained interaction and separately retained polarization checks."""

    system: FreeFermionSystem
    transfer_coords: np.ndarray
    polarization_full: np.ndarray
    polarization_active: np.ndarray
    polarization_rest: np.ndarray
    bare_interaction: np.ndarray
    screened_interaction: np.ndarray
    metadata: dict[str, Any]
    interaction: HexInteractionTable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.system, FreeFermionSystem):
            raise TypeError("system must be a FreeFermionSystem")
        coords = np.asarray(self.transfer_coords, dtype=np.int64)
        expected = centered_hex_coords(2 * self.system.radius)
        if not np.array_equal(coords, expected):
            raise ValueError("cRPA transfers must be the canonical complete H_(2M)")
        object.__setattr__(self, "transfer_coords", np.ascontiguousarray(coords))
        for name in (
            "polarization_full", "polarization_active", "polarization_rest",
            "bare_interaction", "screened_interaction",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (len(coords),) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape (Ntransfer,)")
            object.__setattr__(self, name, np.ascontiguousarray(value))
        for name in ("polarization_full", "polarization_active", "polarization_rest"):
            if np.max(getattr(self, name), initial=0.0) > 1.0e-12:
                raise ValueError(f"{name} violates the non-positive static convention")
        if not np.allclose(
            self.polarization_full,
            self.polarization_active + self.polarization_rest,
            rtol=2.0e-13, atol=1.0e-16,
        ):
            raise ValueError("Pi_full differs from Pi_active + Pi_rest")
        bare = self.system.interaction(coords)
        if not np.allclose(self.bare_interaction, bare, rtol=2.0e-13, atol=1.0e-13):
            raise ValueError("bare interaction differs from the free system")
        denominator = 1.0 - bare * self.polarization_rest
        if np.min(denominator) <= 0:
            raise ValueError("invalid constrained RPA denominator")
        if not np.allclose(
            self.screened_interaction, bare / denominator, rtol=2.0e-13, atol=1.0e-13,
        ):
            raise ValueError("V_RPA^c is inconsistent with the constrained bubble")
        table = HexInteractionTable(2 * self.system.radius, coords, self.screened_interaction)
        if table.inversion_conjugacy_defect > 1.0e-10:
            raise ValueError("V_RPA^c violates inversion symmetry")
        metadata = json.loads(json.dumps(self.metadata, sort_keys=True))
        active = self.system.active_band_index
        if (
            metadata.get("schema") != SCHEMA
            or metadata.get("hf_used") is not False
            or metadata.get("screening_species_count") != 1
            or metadata.get("screening_band_count") != 8
            or metadata.get("screening_pair_count") != 63
            or metadata.get("excluded_pairs") != [[active, active]]
            or metadata.get("screening_pairs") != constrained_pairs(active).tolist()
            or metadata.get("config") != _canonical_config(self.system)
            or metadata.get("system_fingerprint_sha256") != _system_fingerprint(self.system)
            or metadata.get("model_source_sha256") != _model_source_fingerprint()
        ):
            raise ValueError("cRPA provenance does not match the exact free-system input")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "interaction", table)

    @property
    def transfers(self) -> np.ndarray:
        """Alias for integer transfer coordinates."""

        return self.transfer_coords

    @property
    def denominator(self) -> np.ndarray:
        return 1.0 - self.bare_interaction * self.polarization_rest

    @property
    def screening_pairs(self) -> np.ndarray:
        return constrained_pairs(self.system.active_band_index)


def compute_constrained_rpa(
    system: FreeFermionSystem,
    progress: bool = False,
    *,
    degeneracy_tolerance: float = 1.0e-12,
) -> ConstrainedRPAResult:
    r"""Evaluate Pi_rest and V_RPA^c, without a Hartree--Fock calculation.

    For each transfer Q only k for which both k and k+Q belong to H_M is
    included. The transfer table spans H_(2M), with no folding or periodic
    wrapping. The static divided difference uses f'(E) at degeneracy, and
    q=0 uses exact orthogonality between different bands.
    """

    if not isinstance(system, FreeFermionSystem):
        raise TypeError("system must be a FreeFermionSystem")
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0:
        raise ValueError("degeneracy_tolerance must be finite and non-negative")
    started = time.monotonic()
    radius = system.radius
    transfers = centered_hex_coords(2 * radius)
    raw = np.empty((3, len(transfers)), dtype=float)
    active = system.active_band_index
    rest_mask = np.ones((8, 8), dtype=bool)
    rest_mask[active, active] = False
    index_box = np.full((2 * radius + 1, 2 * radius + 1), -1, dtype=np.int64)
    index_box[system.coords[:, 0] + radius, system.coords[:, 1] + radius] = np.arange(
        len(system.coords)
    )
    interval = max(1, len(transfers) // 20)
    for iq, transfer in enumerate(transfers):
        if np.all(transfer == 0):
            occupation = fermi(
                system.energies, system.chemical_potential, system.config.temperature_eV,
            )
            derivative = -occupation * (1.0 - occupation) / system.config.temperature_eV
            raw[0, iq] = system.weight * derivative.sum()
            raw[1, iq] = system.weight * derivative[:, active].sum()
            raw[2, iq] = system.weight * np.delete(derivative, active, axis=1).sum()
        else:
            shifted = system.coords + transfer
            valid = hex_norm(shifted) <= radius
            initial_rows = np.flatnonzero(valid)
            final_coords = shifted[valid]
            final_rows = index_box[final_coords[:, 0] + radius, final_coords[:, 1] + radius]
            overlap = np.einsum(
                "kan,kam->knm",
                system.spinors[initial_rows].conj(), system.spinors[final_rows],
                optimize=True,
            )
            line = lindhard_divided_difference(
                system.energies[final_rows, None, :],
                system.energies[initial_rows, :, None],
                mu=system.chemical_potential,
                temperature=system.config.temperature_eV,
                degeneracy_tolerance=degeneracy_tolerance,
            )
            contribution = np.abs(overlap) ** 2 * line
            raw[0, iq] = system.weight * contribution.sum()
            raw[1, iq] = system.weight * contribution[:, active, active].sum()
            raw[2, iq] = system.weight * contribution[:, rest_mask].sum()
        if progress and (iq == 0 or (iq + 1) % interval == 0 or iq + 1 == len(transfers)):
            elapsed = time.monotonic() - started
            remaining = elapsed * (len(transfers) - iq - 1) / (iq + 1)
            print(
                f"  cRPA {iq + 1:,}/{len(transfers):,}; "
                f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s", flush=True,
            )

    opposite = _opposite_indices(transfers, 2 * radius)
    symmetry_defects = np.max(np.abs(raw - raw[:, opposite]), axis=1)
    full, active_pi, rest = 0.5 * (raw + raw[:, opposite])
    bare = np.asarray(system.interaction(transfers), dtype=float)
    screened = bare / (1.0 - bare * rest)
    metadata = {
        "schema": SCHEMA,
        "calculation": "one-isospin constrained RPA from the free eight-band ABCA system",
        "hf_used": False,
        "isospin": system.config.isospin,
        "screening_species_count": 1,
        "spin_valley_degeneracy_factor": 1,
        "screening_band_count": 8,
        "screening_pair_count": 63,
        "screening_pairs": constrained_pairs(active).tolist(),
        "excluded_pairs": [[active, active]],
        "mixed_active_remote_transitions_retained": True,
        "active_band_index": active,
        "chemical_potential_eV": system.chemical_potential,
        "temperature_eV": system.config.temperature_eV,
        "normalization": "open-hex integral weight abs(det(basis))/(2*pi)^2",
        "quadrature_weight": system.weight,
        "bubble_routing": "k,k+Q on the H_M intersection domain",
        "polarization_convention": "Pi=[f(E_final)-f(E_initial)]/(E_final-E_initial) <= 0",
        "rpa_formula": "V_RPA^c(Q)=V0(Q)/(1-V0(Q)*Pi_rest(Q))",
        "rest_evaluation": "direct sum of 63 pairs; full minus active retained as diagnostic",
        "symmetrized_q_and_minus_q": True,
        "raw_polarization_inversion_error": dict(zip(
            ("full", "active", "rest"), symmetry_defects.tolist(),
        )),
        "decomposition_max_abs_error": float(np.max(np.abs(full - active_pi - rest))),
        "degeneracy_tolerance_eV": float(degeneracy_tolerance),
        "elapsed_seconds": time.monotonic() - started,
        "config": _canonical_config(system),
        "system_fingerprint_sha256": _system_fingerprint(system),
        "model_source_path": str(Path(_reference_model.__file__).resolve()),
        "model_source_sha256": _model_source_fingerprint(),
    }
    return ConstrainedRPAResult(
        system, transfers, full, active_pi, rest, bare, screened, metadata,
    )


def save_constrained_rpa(result: ConstrainedRPAResult, path: str | Path) -> Path:
    """Atomically save a complete cRPA table and exact input provenance."""

    if not isinstance(result, ConstrainedRPAResult):
        raise TypeError("result must be a ConstrainedRPAResult")
    output = Path(path).expanduser().resolve()
    if not output.suffix:
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(result.metadata, sort_keys=True)),
            **{name: getattr(result, name) for name in (
                "transfer_coords", "polarization_full", "polarization_active",
                "polarization_rest", "bare_interaction", "screened_interaction",
            )},
        )
    temporary.replace(output)
    return output


def load_constrained_rpa(path: str | Path, system: FreeFermionSystem) -> ConstrainedRPAResult:
    """Read a cache only if its exact free-system input still agrees."""

    required = {
        "metadata_json", "transfer_coords", "polarization_full", "polarization_active",
        "polarization_rest", "bare_interaction", "screened_interaction",
    }
    with np.load(Path(path).expanduser(), allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"cRPA archive is missing {sorted(missing)}")
        metadata = json.loads(str(saved["metadata_json"].item()))
        arrays = {name: saved[name] for name in required if name != "metadata_json"}
    return ConstrainedRPAResult(system=system, metadata=metadata, **arrays)


__all__ = [
    "ConstrainedRPAResult", "compute_constrained_rpa", "constrained_pairs",
    "save_constrained_rpa", "load_constrained_rpa",
]
