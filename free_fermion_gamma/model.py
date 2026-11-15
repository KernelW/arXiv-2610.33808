"""Free ABCA eigensystem and the open triangular-hexagonal momentum mesh.

This module extracts only the non-Hartree--Fock ingredients from
``charge order/finite_Q_HF_free_bare.ipynb``.  Momenta are dimensionless
``k*a`` and energies are in eV.  The orbital order is
``(A1,B1,A2,B2,A3,B3,A4,B4)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import brentq


Array = np.ndarray
ISOSPIN_LABELS = ("K up", "K down", "K' up", "K' down")
ISOSPIN_VALLEY = {"K up": +1, "K down": +1, "K' up": -1, "K' down": -1}
ORBITAL_LABELS = ("A1", "B1", "A2", "B2", "A3", "B3", "A4", "B4")


def hex_norm(coords: Array) -> Array:
    """Integer-coordinate radius for the triangular hexagonal mesh."""

    value = np.asarray(coords, dtype=np.int64)
    if value.shape[-1:] != (2,):
        raise ValueError("hexagonal coordinates must end in an axis of size 2")
    i, j = value[..., 0], value[..., 1]
    return np.maximum.reduce((np.abs(i), np.abs(j), np.abs(i - j)))


def centered_hex_coords(radius: int) -> Array:
    """Match ``build_hexagonal_mesh`` in the source notebook exactly."""

    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return np.asarray(
        [
            (i, j)
            for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1)
            if max(abs(i), abs(j), abs(i - j)) <= radius
        ],
        dtype=np.int64,
    )


def triangular_basis(spacing: float) -> Array:
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    return spacing * np.asarray(
        [[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float
    )


@dataclass(frozen=True)
class ABCAParams:
    """Eight-orbital free Hamiltonian parameters, in eV."""

    gamma0: float = 3.1
    gamma1: float = 0.38
    gamma2: float = -0.015
    gamma3: float = -0.29
    gamma4: float = -0.141
    dd: float = -0.0105
    delta_d: float = 0.150


def abca_hamiltonians(kpoints: Array, params: ABCAParams, *, valley: int) -> Array:
    """Return the free ABCA Hamiltonian for one valley."""

    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 2:
        raise ValueError("kpoints must have shape (N,2)")
    if valley not in (-1, +1):
        raise ValueError("valley must be +1 or -1")
    pi = valley * kpoints[:, 0] + 1j * kpoints[:, 1]
    pd = pi.conj()
    prefactor = np.sqrt(3.0) / 2.0
    v0 = prefactor * params.gamma0
    v3 = prefactor * params.gamma3
    v4 = prefactor * params.gamma4
    h = np.zeros((len(kpoints), 8, 8), dtype=np.complex128)
    d = params.delta_d
    h[:, np.arange(8), np.arange(8)] = (
        d / 2.0 + params.dd,
        d / 2.0,
        d / 6.0,
        d / 6.0,
        -d / 6.0,
        -d / 6.0,
        -d / 2.0,
        -d / 2.0 + params.dd,
    )

    def put(i: int, j: int, value: Array | float) -> None:
        h[:, i, j] = value
        h[:, j, i] = np.conj(value)

    put(0, 1, v0 * pd)
    put(0, 2, v4 * pd)
    put(0, 3, v3 * pi)
    put(0, 5, params.gamma2 / 2.0)
    put(1, 2, params.gamma1)
    put(1, 3, v4 * pd)
    put(2, 3, v0 * pd)
    put(2, 4, v4 * pd)
    put(2, 5, v3 * pi)
    put(2, 7, params.gamma2 / 2.0)
    put(3, 4, params.gamma1)
    put(3, 5, v4 * pd)
    put(4, 5, v0 * pd)
    put(4, 6, v4 * pd)
    put(4, 7, v3 * pi)
    put(5, 6, params.gamma1)
    put(5, 7, v4 * pd)
    put(6, 7, v0 * pd)
    return h


def fermi(energy: Array, mu: float, temperature: float) -> Array:
    argument = np.clip((np.asarray(energy) - float(mu)) / float(temperature), -700, 700)
    return 1.0 / (np.exp(argument) + 1.0)


@dataclass(frozen=True)
class FreeFermionConfig:
    """Physical and numerical inputs for one fixed isospin block."""

    radius: int = 30
    spacing: float = 0.004
    isospin: str = "K up"
    active_band_index: int = 4
    carrier_density_cm2: float = 7.0e11
    lattice_constant_nm: float = 0.246
    temperature_eV: float = 1.0e-5
    dielectric_constant: float = 5.0
    gate_distance_nm: float = 20.0
    target_particles_per_momentum: float | None = None
    band_params: ABCAParams = field(default_factory=ABCAParams)

    def __post_init__(self) -> None:
        if int(self.radius) != self.radius or self.radius < 0:
            raise ValueError("radius must be a non-negative integer")
        if not np.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("spacing must be finite and positive")
        if self.isospin not in ISOSPIN_LABELS:
            raise ValueError(f"isospin must be one of {ISOSPIN_LABELS}")
        if not 0 <= int(self.active_band_index) < 8:
            raise ValueError("active_band_index must lie in 0..7")
        for value, name in (
            (self.lattice_constant_nm, "lattice_constant_nm"),
            (self.temperature_eV, "temperature_eV"),
            (self.dielectric_constant, "dielectric_constant"),
            (self.gate_distance_nm, "gate_distance_nm"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.carrier_density_cm2) or self.carrier_density_cm2 < 0:
            raise ValueError("carrier_density_cm2 must be finite and non-negative")
        if self.target_particles_per_momentum is not None:
            target = float(self.target_particles_per_momentum)
            if not 0.0 < target < 8.0:
                raise ValueError("target_particles_per_momentum must lie in (0,8)")


@dataclass(frozen=True)
class FreeFermionSystem:
    """Free dispersion/eigenstates and bare gated Coulomb interaction."""

    config: FreeFermionConfig
    coords: Array
    points: Array
    basis: Array
    weight: float
    hamiltonians: Array
    energies: Array
    spinors: Array
    chemical_potential: float
    target_particles: float

    def __post_init__(self) -> None:
        n = len(self.coords)
        if self.coords.shape != (n, 2) or self.points.shape != (n, 2):
            raise ValueError("invalid momentum mesh")
        if self.hamiltonians.shape != (n, 8, 8):
            raise ValueError("hamiltonians must have shape (N,8,8)")
        if self.energies.shape != (n, 8) or self.spinors.shape != (n, 8, 8):
            raise ValueError("invalid eigensystem shapes")
        if not np.all(np.isfinite(self.energies)) or not np.all(np.isfinite(self.spinors)):
            raise ValueError("free eigensystem contains non-finite values")
        gram = np.einsum("kan,kam->knm", self.spinors.conj(), self.spinors, optimize=True)
        if np.max(np.abs(gram - np.eye(8))) > 1.0e-9:
            raise ValueError("free eigenstates are not unitary")

    @property
    def radius(self) -> int:
        return int(self.config.radius)

    @property
    def beta(self) -> float:
        return 1.0 / self.config.temperature_eV

    @property
    def active_band_index(self) -> int:
        return int(self.config.active_band_index)

    @property
    def valley(self) -> int:
        return ISOSPIN_VALLEY[self.config.isospin]

    @property
    def filling(self) -> float:
        return float(np.mean(np.sum(fermi(
            self.energies, self.chemical_potential, self.config.temperature_eV
        ), axis=1)))

    def interaction(self, transfer_coords: Array) -> Array:
        """Bare dual-gated Coulomb ``V(q)`` including its exact q=0 limit."""

        coords = np.asarray(transfer_coords)
        if coords.shape[-1:] != (2,):
            raise ValueError("transfer coordinates must end in an axis of size 2")
        qpoints = coords @ self.basis
        qnorm = np.linalg.norm(qpoints, axis=-1)
        e2_over_epsilon = (
            1.44 / self.config.dielectric_constant / self.config.lattice_constant_nm
        )
        gate = self.config.gate_distance_nm / self.config.lattice_constant_nm
        ratio = np.empty_like(qnorm, dtype=float)
        np.divide(np.tanh(gate * qnorm), qnorm, out=ratio, where=qnorm > 0.0)
        ratio = np.where(qnorm > 0.0, ratio, gate)
        return 2.0 * np.pi * e2_over_epsilon * ratio

    def domain(self, transfer: Iterable[int]) -> tuple[Array, Array]:
        """Return rows for ``p`` and ``p-Q`` on the open mesh."""

        q = np.asarray(tuple(transfer))
        if q.shape != (2,) or not np.array_equal(q, np.asarray(q, dtype=np.int64)):
            raise ValueError("transfer must contain two integers")
        q = np.asarray(q, dtype=np.int64)
        if hex_norm(q) > 2 * self.radius:
            raise ValueError("transfer lies outside the H_(2M) difference mesh")
        lookup = {tuple(coord): row for row, coord in enumerate(self.coords)}
        source = np.asarray(
            [row for row, coord in enumerate(self.coords) if tuple(coord - q) in lookup],
            dtype=np.int64,
        )
        shifted = np.asarray([lookup[tuple(self.coords[row] - q)] for row in source])
        return source, shifted


def _chemical_potential(energies: Array, target: float, temperature: float) -> float:
    flat = np.asarray(energies, dtype=float)
    pad = max(float(np.ptp(flat)), 20.0 * temperature, 1.0)
    number = lambda mu: float(np.mean(np.sum(fermi(flat, mu, temperature), axis=1)) - target)
    return float(
        brentq(number, float(flat.min() - pad), float(flat.max() + pad), xtol=1.0e-13)
    )


def build_free_fermion_system(
    config: FreeFermionConfig | None = None,
) -> FreeFermionSystem:
    """Construct the free eigensystem for the requested isospin; no HF is run."""

    config = config or FreeFermionConfig()
    coords = centered_hex_coords(config.radius)
    basis = triangular_basis(config.spacing)
    points = coords @ basis
    weight = float(abs(np.linalg.det(basis)) / (2.0 * np.pi) ** 2)
    h0 = abca_hamiltonians(points, config.band_params, valley=ISOSPIN_VALLEY[config.isospin])
    energies, spinors = np.linalg.eigh(h0)
    if config.target_particles_per_momentum is None:
        active_particles = (
            config.carrier_density_cm2
            * (config.lattice_constant_nm * 1.0e-7) ** 2
            / (len(coords) * weight)
        )
        target = float(config.active_band_index + active_particles)
    else:
        target = float(config.target_particles_per_momentum)
    if not 0.0 < target < 8.0:
        raise ValueError(
            f"computed target filling {target:.8g} lies outside the eight-band block"
        )
    mu = _chemical_potential(energies, target, config.temperature_eV)
    return FreeFermionSystem(
        config=config,
        coords=coords,
        points=points,
        basis=basis,
        weight=weight,
        hamiltonians=h0,
        energies=energies,
        spinors=spinors,
        chemical_potential=mu,
        target_particles=target,
    )


__all__ = [
    "ABCAParams",
    "FreeFermionConfig",
    "FreeFermionSystem",
    "ISOSPIN_LABELS",
    "ISOSPIN_VALLEY",
    "ORBITAL_LABELS",
    "abca_hamiltonians",
    "build_free_fermion_system",
    "centered_hex_coords",
    "fermi",
    "hex_norm",
    "triangular_basis",
]
