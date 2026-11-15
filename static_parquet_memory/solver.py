"""Dense, static two-particle-hole-channel parquet approximation.

Only one active band and one isospin are retained. All four external momenta
are on the same open hexagonal mesh. For each transfer Q, rows and columns
are q,p in D_Q = {k: k and k-Q are on the mesh}; the four legs are
(q-Q,q;p-Q,p). The crossing operation is

    C[X]_Q(q,p) = X_(p-q)(p-Q,p).

The irreducible input is the antisymmetrized *bare* interaction. With
B_Q = w L_Q <= 0, the equations implemented here are

    Gamma_Q = (1 - K_Q B_Q)^(-1) K_Q,
    Phi_Q = Gamma_Q - K_Q,
    K_Q = g U_Q - C[Phi]_Q.

B is diagonal and its multiplication scales matrix columns. This is an
explicit static approximation: the particle-particle channel and frequency
dependence are omitted. The calculation is not a self-consistent HF solve.
Dense all-transfer tensors make this implementation a small-mesh reference;
the memory guard deliberately rejects a production-sized uncompressed run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve

from free_fermion_gamma.gamma import particle_hole_line
from free_fermion_gamma.model import (
    FreeFermionSystem,
    centered_hex_coords,
    hex_norm,
)

Array = np.ndarray
Blocks = tuple[Array, ...]
_CROSSING_CHUNK = 1 << 18


def _finite_chunks(values: Array, chunk_size: int = _CROSSING_CHUNK) -> bool:
    """Avoid an entry-sized boolean temporary for a large packed tensor."""
    flat = values.reshape(-1) if values.flags.c_contiguous else values
    if flat.ndim != 1:
        return bool(np.all(np.isfinite(flat)))
    return all(np.all(np.isfinite(flat[start:start + chunk_size]))
               for start in range(0, len(flat), chunk_size))


def _norm(blocks: Sequence[Array]) -> float:
    return float(np.sqrt(sum(float(np.vdot(x, x).real) for x in blocks)))


def _difference_norm(left: Sequence[Array], right: Sequence[Array]) -> float:
    return float(np.sqrt(sum(
        float(np.vdot(difference, difference).real)
        for a, b in zip(left, right)
        for difference in (a - b,)
    )))


def _relative(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def _coord(value: Iterable[int], name: str) -> tuple[int, int]:
    raw = np.asarray(tuple(value))
    if raw.shape != (2,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain two finite integers")
    converted = raw.astype(np.int64)
    if not np.array_equal(raw, converted):
        raise ValueError(f"{name} must contain two integers")
    return int(converted[0]), int(converted[1])


@dataclass
class StaticParquetProblem:
    """All-Q active-band tensors and their exact finite-mesh crossing map.

    ``bare[i]`` is U_Q(q,p), ``bubble[i]`` is the diagonal w L_Q(q), and
    ``offsets`` packs each dense block in C row-major order. ``crossing``
    performs a permutation, without conjugation or an additional sign.
    The minus sign from crossing appears in the equation for K.
    """

    system: FreeFermionSystem
    transfers: Array
    domain_coords: tuple[Array, ...]
    bare: Blocks
    bubble: tuple[Array, ...]
    crossing_permutation: Array
    offsets: Array
    memory_diagnostics: dict[str, int | float | str]
    right_rows: tuple[Array, ...]
    left_rows: tuple[Array, ...]
    _sector_indices: dict[tuple[int, int], int] = field(repr=False)
    _position_maps: tuple[dict[tuple[int, int], int], ...] = field(repr=False)
    _bare_flat: Array = field(repr=False)

    @property
    def transfer_coords(self) -> Array:
        """Alias for transfers, matching the existing RPA naming convention."""
        return self.transfers

    @property
    def total_entries(self) -> int:
        return int(self.offsets[-1])

    @property
    def bare_flat(self) -> Array:
        """Shared row-major bare storage; ``bare`` contains views of this array."""
        return self._bare_flat

    @property
    def sector_sizes(self) -> Array:
        return np.asarray([len(coords) for coords in self.domain_coords], dtype=np.int64)

    def _validate_blocks(self, blocks: Sequence[Array], name: str = "blocks") -> Blocks:
        values = tuple(blocks)
        if len(values) != len(self.domain_coords):
            raise ValueError(f"{name} must contain {len(self.domain_coords)} transfer blocks")
        checked = []
        for index, (value, coords) in enumerate(zip(values, self.domain_coords)):
            matrix = np.asarray(value, dtype=np.complex128)
            expected = (len(coords), len(coords))
            if matrix.shape != expected:
                raise ValueError(
                    f"{name} block at Q={tuple(self.transfers[index])} has shape "
                    f"{matrix.shape}; expected {expected}"
                )
            if not _finite_chunks(matrix):
                raise ValueError(f"{name} contains non-finite values at Q={tuple(self.transfers[index])}")
            checked.append(matrix)
        return tuple(checked)

    def pack(self, blocks: Sequence[Array], *, copy: bool = True) -> Array:
        """Pack blocks, optionally reusing an already shared contiguous buffer.

        The default preserves the independent-copy behavior. ``copy=False``
        returns a view when the blocks already span one packed complex array;
        independent or noncontiguous matrices still need a new packed array.
        """
        checked = self._validate_blocks(blocks)
        if not copy:
            owner = checked[0]
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if (owner.dtype == np.complex128 and owner.flags.c_contiguous
                    and owner.size == self.total_entries):
                flat = owner.reshape(-1)
                address = flat.__array_interface__["data"][0]
                if all(matrix.flags.c_contiguous and
                       matrix.__array_interface__["data"][0] == address + int(self.offsets[i]) * 16
                       for i, matrix in enumerate(checked)):
                    return flat
        flat = np.empty(self.total_entries, dtype=np.complex128)
        self._pack_checked_into(checked, flat)
        return flat

    def _pack_checked_into(self, checked: Blocks, output: Array) -> None:
        for i, matrix in enumerate(checked):
            destination = output[int(self.offsets[i]):int(self.offsets[i + 1])]
            if (np.shares_memory(matrix, output)
                    and (not matrix.flags.c_contiguous or not output.flags.c_contiguous
                         or matrix.__array_interface__["data"][0] != destination.__array_interface__["data"][0])):
                raise ValueError("pack_into does not support overlapping reordered input blocks")
        for i, matrix in enumerate(checked):
            destination = output[int(self.offsets[i]):int(self.offsets[i + 1])]
            destination[:] = matrix.reshape(-1)

    def pack_into(self, blocks: Sequence[Array], output: Array) -> Array:
        """Copy blocks into caller-owned packed storage without concatenation."""
        flat = np.asarray(output)
        if (flat.shape != (self.total_entries,) or flat.dtype != np.complex128
                or not flat.flags.writeable):
            raise ValueError("output must be a writable packed complex128 array")
        self._pack_checked_into(self._validate_blocks(blocks), flat)
        return flat

    def unpack(self, packed: Array) -> Blocks:
        """Return block views of a finite packed complex vector."""
        flat = np.asarray(packed, dtype=np.complex128)
        if flat.shape != (self.total_entries,) or not _finite_chunks(flat):
            raise ValueError(f"packed values must be finite with shape ({self.total_entries},)")
        return tuple(
            flat[int(self.offsets[i]):int(self.offsets[i + 1])].reshape(len(coords), len(coords))
            for i, coords in enumerate(self.domain_coords)
        )

    def crossing(self, blocks: Sequence[Array]) -> Blocks:
        """Apply C; crossing twice returns the original four-leg tensor."""
        output = np.empty(self.total_entries, dtype=np.complex128)
        self.crossing_into(self.pack(blocks, copy=False), output, check_finite=False)
        return self.unpack(output)

    def crossing_destinations(self, sector: int) -> Array:
        """Source-block destinations in C, as a compact permutation view.

        Because C is an involution, this slice supports both scattering one
        computed source block into a crossed output and gathering its adjoint.
        """
        if not isinstance(sector, (int, np.integer)) or not 0 <= sector < len(self.domain_coords):
            raise ValueError("sector must be a valid integer transfer index")
        return self.crossing_permutation[int(self.offsets[sector]):int(self.offsets[sector + 1])]

    def crossing_into(
        self, packed: Array, output: Array, *, add: bool = False,
        scale: complex = 1.0, chunk_size: int = _CROSSING_CHUNK,
        check_finite: bool = True,
    ) -> Array:
        """Write ``scale*C[packed]`` into output, optionally adding it.

        Gather temporaries are bounded by ``chunk_size``; input and output must
        not overlap. This avoids constructing another full crossed tensor.
        """
        source, destination = np.asarray(packed), np.asarray(output)
        if (source.shape != (self.total_entries,) or not np.issubdtype(source.dtype, np.number)
                or destination.shape != source.shape or destination.dtype != np.complex128
                or not destination.flags.writeable):
            raise ValueError("crossing_into requires packed numeric input and writable complex128 output")
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")
        if np.shares_memory(source, destination):
            raise ValueError("crossing_into input and output must not overlap")
        if check_finite and not _finite_chunks(source, int(chunk_size)):
            raise ValueError("packed crossing input must be finite")
        for start in range(0, self.total_entries, int(chunk_size)):
            stop = min(start + int(chunk_size), self.total_entries)
            values = source[self.crossing_permutation[start:stop]]
            if scale != 1.0:
                values = scale * values
            if add:
                destination[start:stop] += values
            else:
                destination[start:stop] = values
        return destination

    def solve_one_channel(self, kernel: Sequence[Array]) -> Blocks:
        """Solve each static BSE with the supplied channel-irreducible K."""
        checked = self._validate_blocks(kernel, "kernel")
        vertices = []
        for index, (matrix, bubble) in enumerate(zip(checked, self.bubble)):
            response = np.eye(len(matrix), dtype=np.complex128) - matrix * bubble[None, :]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", LinAlgWarning)
                    vertex = solve(
                        response, matrix, assume_a="gen", check_finite=False,
                        overwrite_a=True, overwrite_b=False,
                    )
            except (np.linalg.LinAlgError, LinAlgWarning) as error:
                transfer = tuple(int(x) for x in self.transfers[index])
                raise np.linalg.LinAlgError(
                    f"Static BSE is singular or ill-conditioned at Q={transfer}; "
                    "reduce the interaction scale or inspect a possible response pole."
                ) from error
            if not np.all(np.isfinite(vertex)):
                raise np.linalg.LinAlgError(
                    f"Static BSE produced non-finite values at Q={tuple(self.transfers[index])}"
                )
            vertices.append(vertex)
        return tuple(vertices)

    def vertex_at(
        self, blocks: Sequence[Array], transfer: Iterable[int],
        q: Iterable[int], p: Iterable[int],
    ) -> complex:
        """Look up a four-point amplitude from bare, kernel, or gamma blocks."""
        key = _coord(transfer, "transfer")
        try:
            index = self._sector_indices[key]
        except KeyError as error:
            raise ValueError(f"transfer {key} is outside the difference mesh") from error
        qkey, pkey = _coord(q, "q"), _coord(p, "p")
        positions = self._position_maps[index]
        if qkey not in positions or pkey not in positions:
            raise ValueError(f"q and p must both lie in D_Q for Q={key}")
        if len(blocks) != len(self.domain_coords):
            raise ValueError("blocks must contain one matrix per transfer")
        matrix = np.asarray(blocks[index])
        size = len(positions)
        if matrix.shape != (size, size):
            raise ValueError(f"block at Q={key} must have shape {(size, size)}")
        return complex(matrix[positions[qkey], positions[pkey]])


@dataclass(frozen=True)
class IterationDiagnostics:
    iteration: int
    fixedpoint_residual: float
    bse_residual: float
    crossing_defect: float
    hermiticity_defect: float


@dataclass(frozen=True)
class StaticParquetResult:
    problem: StaticParquetProblem
    gamma: Blocks
    kernel: Blocks
    phi: Blocks
    history: tuple[IterationDiagnostics, ...]
    converged: bool
    interaction_scale: float

    @property
    def irreducible(self) -> Blocks:
        return self.kernel

    @property
    def final_diagnostics(self) -> IterationDiagnostics:
        return self.history[-1]

    def vertex_at(self, transfer: Iterable[int], q: Iterable[int], p: Iterable[int]) -> complex:
        return self.problem.vertex_at(self.gamma, transfer, q, p)


def _problem_memory_plan(
    system: FreeFermionSystem, memory_profile: str,
) -> tuple[Array, Array, dict[str, int | float | str]]:
    if not isinstance(system, FreeFermionSystem):
        raise TypeError("system must be a FreeFermionSystem")
    if memory_profile not in ("legacy", "gamma-ph"):
        raise ValueError("memory_profile must be 'legacy' or 'gamma-ph'")
    radius = system.radius
    transfers = centered_hex_coords(2 * radius)
    sizes = np.asarray([
        np.count_nonzero(hex_norm(system.coords - transfer) <= radius)
        for transfer in transfers
    ], dtype=np.int64)
    entries = sum(int(size) ** 2 for size in sizes)
    if entries > np.iinfo(np.intp).max:
        raise MemoryError("the packed vertex exceeds this platform's addressable array size")
    domain_entries = sum(int(size) for size in sizes)
    largest = int(np.max(sizes))
    side = 2 * radius + 1
    crossing_dtype = np.dtype(np.uint32 if entries - 1 <= np.iinfo(np.uint32).max else np.uint64)
    position_dtype = np.dtype(np.int32 if max(len(transfers), largest) <= np.iinfo(np.int32).max else np.int64)
    vertex_bytes = entries * np.dtype(np.complex128).itemsize
    crossing_bytes = entries * crossing_dtype.itemsize
    # Exact persistent array payload, plus a conservative allowance for Python
    # coordinate maps, tuples, and ndarray headers. The eigensystem is shared.
    system_bytes = sum(getattr(system, name).nbytes for name in (
        "coords", "points", "basis", "hamiltonians", "energies", "spinors",
    ))
    array_bytes = (vertex_bytes + crossing_bytes + 40 * domain_entries
                   + transfers.nbytes + (len(transfers) + 1) * 8 + system_bytes)
    metadata_bytes = 256 * domain_entries + 1024 * len(transfers)
    persistent_bytes = array_bytes + metadata_bytes
    position_bytes = len(transfers) * side * side * position_dtype.itemsize
    transfer_position_bytes = (4 * radius + 1) ** 2 * position_dtype.itemsize
    chunk_bytes = min(entries, _CROSSING_CHUNK) * 64
    construction_scratch = (position_bytes + transfer_position_bytes
                            + 256 * largest * largest + 256 * len(system.coords)
                            + chunk_bytes)
    construction_peak = persistent_bytes + construction_scratch
    # Keep the existing twelve-tensor reservation for the older Gamma/Phi
    # iteration. The Gamma^ph solver adds its own explicit working-set plan.
    legacy_peak = persistent_bytes + 11 * vertex_bytes + 64 * largest * largest
    peak_bytes = max(construction_peak, legacy_peak) if memory_profile == "legacy" else construction_peak
    diagnostics: dict[str, int | float | str] = {
        "number_of_momenta": len(system.coords),
        "number_of_sectors": len(transfers),
        "total_vertex_entries": entries,
        "domain_entry_count": domain_entries,
        "largest_sector_size": largest,
        "vertex_bytes": vertex_bytes,
        "crossing_bytes": crossing_bytes,
        "crossing_index_dtype": crossing_dtype.name,
        "position_index_dtype": position_dtype.name,
        "persistent_array_bytes": array_bytes,
        "persistent_metadata_bytes": metadata_bytes,
        "persistent_bytes": persistent_bytes,
        "construction_scratch_bytes": construction_scratch,
        "construction_peak_bytes": construction_peak,
        "construction_peak_mb": construction_peak / 1.0e6,
        "legacy_solver_peak_bytes": legacy_peak,
        "memory_profile": memory_profile,
        "estimated_peak_bytes": peak_bytes,
        "estimated_peak_mb": peak_bytes / 1.0e6,
    }
    return transfers, sizes, diagnostics


def estimate_problem_memory(
    system: FreeFermionSystem, *, memory_profile: str = "legacy",
) -> dict[str, int | float | str]:
    """Plan exact tensor sizes before allocating bare or crossing storage.

    ``gamma-ph`` accounts for construction and solver-neutral persistent data.
    Its solver must separately add its working vectors and sector scratch.
    ``legacy`` also reserves twelve complex tensors for the older iteration.
    Estimates include conservative Python metadata/scratch allowances; they
    describe a planning bound, not an operating-system memory measurement.
    """
    return _problem_memory_plan(system, memory_profile)[2]


def build_problem(
    system: FreeFermionSystem, *, maximum_memory_mb: float = 512.0,
    memory_profile: str = "legacy",
) -> StaticParquetProblem:
    """Construct exact all-Q data using shared bare and compact crossing arrays.

    The guard uses the selected planning profile before allocating either
    packed tensor. MB denotes 1,000,000 bytes. Finite-domain and Pauli checks
    are chunked, so they never allocate another full crossing or bare tensor.
    """
    budget = float(maximum_memory_mb)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("maximum_memory_mb must be finite and positive")
    transfers, sizes, diagnostics = _problem_memory_plan(system, memory_profile)
    diagnostics["maximum_memory_mb"] = budget
    entries = int(diagnostics["total_vertex_entries"])
    peak_bytes = int(diagnostics["estimated_peak_bytes"])
    if peak_bytes > budget * 1.0e6:
        raise MemoryError(
            f"Dense static parquet ({memory_profile}) needs an estimated {peak_bytes / 1.0e6:.1f} MB "
            f"for {entries:,} active-band vertex entries; the limit is {budget:g} MB. "
            "Use a smaller momentum radius or an explicitly larger memory limit."
        )

    radius, side = system.radius, 2 * system.radius + 1
    offsets = np.asarray([0, *np.cumsum([int(size) ** 2 for size in sizes])], dtype=np.int64)
    bare_flat = np.empty(entries, dtype=np.complex128)
    row_lookup = {tuple(int(x) for x in coord): row for row, coord in enumerate(system.coords)}
    domain_coords, right_rows, left_rows, bare, bubble, position_maps = [], [], [], [], [], []
    active = system.active_band_index
    position_dtype = np.dtype(diagnostics["position_index_dtype"])
    positions = np.full((len(transfers), side, side), -1, dtype=position_dtype)
    transfer_positions = np.full((4 * radius + 1, 4 * radius + 1), -1, dtype=position_dtype)
    transfer_positions[transfers[:, 0] + 2 * radius, transfers[:, 1] + 2 * radius] = np.arange(len(transfers))
    for index, transfer in enumerate(transfers):
        rows = np.flatnonzero(hex_norm(system.coords - transfer) <= radius)
        coords = system.coords[rows].copy()
        shifted = np.asarray([row_lookup[tuple(int(x) for x in coord - transfer)] for coord in coords])
        left = system.spinors[shifted, :, active]
        right = system.spinors[rows, :, active]
        density = np.sum(left.conj() * right, axis=1)
        delta = coords[None, :, :] - coords[:, None, :]
        matrix = bare_flat[int(offsets[index]):int(offsets[index + 1])].reshape(len(coords), len(coords))
        matrix[:] = (
            complex(system.interaction(-transfer)) * np.outer(density, density.conj())
            - system.interaction(delta) * (left.conj() @ left.T) * (right @ right.conj().T)
        )
        line = particle_hole_line(
            system.energies[shifted, active], system.energies[rows, active],
            mu=system.chemical_potential, temperature=system.config.temperature_eV,
        )
        if not np.all(np.isfinite(line)) or np.max(line, initial=0.0) > 1.0e-11:
            raise ValueError(f"static Lindhard line must be finite and non-positive at Q={tuple(transfer)}")
        domain_coords.append(coords)
        right_rows.append(rows)
        left_rows.append(shifted)
        bare.append(matrix)
        bubble.append(system.weight * np.minimum(line, 0.0))
        position_maps.append({tuple(int(x) for x in coord): row for row, coord in enumerate(coords)})
        positions[index, coords[:, 0] + radius, coords[:, 1] + radius] = np.arange(len(coords))

    permutation = np.empty(entries, dtype=np.dtype(diagnostics["crossing_index_dtype"]))
    for index, (transfer, coords) in enumerate(zip(transfers, domain_coords)):
        delta = coords[None, :, :] - coords[:, None, :]
        sectors = transfer_positions[delta[..., 0] + 2 * radius, delta[..., 1] + 2 * radius]
        crossed_q = coords - transfer
        qpos = positions[sectors, crossed_q[None, :, 0] + radius, crossed_q[None, :, 1] + radius]
        ppos = positions[sectors, coords[None, :, 0] + radius, coords[None, :, 1] + radius]
        if np.any(sectors < 0) or np.any(qpos < 0) or np.any(ppos < 0):
            raise ValueError("the open momentum domain is not closed under crossing")
        permutation[int(offsets[index]):int(offsets[index + 1])] = (
            offsets[sectors] + qpos * sizes[sectors] + ppos
        ).reshape(-1)
    del positions, transfer_positions, row_lookup
    for start in range(0, entries, _CROSSING_CHUNK):
        stop = min(start + _CROSSING_CHUNK, entries)
        if not np.array_equal(permutation[permutation[start:stop]],
                              np.arange(start, stop, dtype=permutation.dtype)):
            raise ValueError("constructed crossing map is not an involution")
    problem = StaticParquetProblem(
        system=system, transfers=transfers, domain_coords=tuple(domain_coords),
        bare=tuple(bare), bubble=tuple(bubble), crossing_permutation=permutation,
        offsets=offsets, memory_diagnostics=diagnostics, right_rows=tuple(right_rows),
        left_rows=tuple(left_rows),
        _sector_indices={tuple(int(x) for x in coord): i for i, coord in enumerate(transfers)},
        _position_maps=tuple(position_maps),
        _bare_flat=bare_flat,
    )
    squared_defect, squared_norm = 0.0, 0.0
    for start in range(0, entries, _CROSSING_CHUNK):
        stop = min(start + _CROSSING_CHUNK, entries)
        value = bare_flat[start:stop]
        difference = value + bare_flat[permutation[start:stop]]
        squared_defect += float(np.vdot(difference, difference).real)
        squared_norm += float(np.vdot(value, value).real)
    relative_defect = _relative(float(np.sqrt(squared_defect)), float(np.sqrt(squared_norm)))
    diagnostics["bare_crossing_defect"] = relative_defect
    diagnostics["crossing_involution_defect"] = 0.0
    if relative_defect > 1.0e-10:
        raise ValueError(f"bare HF kernel violates crossing: relative defect {relative_defect:.3e}")
    return problem


def solve_parquet(
    problem: StaticParquetProblem, *, interaction_scale: float = 0.01,
    mixing: float = 0.3, tolerance: float = 1.0e-9,
    maximum_iterations: int = 300, initial_kernel: Sequence[Array] | None = None,
    progress: Callable[[IterationDiagnostics], None] | None = None,
) -> StaticParquetResult:
    """Iterate the two static ph channels with an unmixed residual check.

    Each history record describes the returned/current K and its newly solved
    Gamma. Exhausting the iteration limit returns a coherent result with
    ``converged=False``; it never marks a small damped step as convergence and
    never antisymmetrizes the vertex after solving it. Singular sector solves
    raise an informative ``numpy.linalg.LinAlgError``.
    """
    if not isinstance(problem, StaticParquetProblem):
        raise TypeError("problem must be a StaticParquetProblem")
    scale, fraction, threshold = float(interaction_scale), float(mixing), float(tolerance)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("interaction_scale must be finite and non-negative")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("mixing must be finite with 0 < mixing <= 1")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be a positive integer")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    source = tuple(scale * matrix for matrix in problem.bare)
    kernel = tuple(matrix.copy() for matrix in (
        source if initial_kernel is None else problem._validate_blocks(initial_kernel, "initial_kernel")
    ))
    source_norm = _norm(source)
    history = []
    converged = False
    for iteration in range(1, int(maximum_iterations) + 1):
        gamma = problem.solve_one_channel(kernel)
        phi = tuple(vertex - irreducible for vertex, irreducible in zip(gamma, kernel))
        crossed_phi = problem.crossing(phi)
        target = tuple(bare - crossed for bare, crossed in zip(source, crossed_phi))
        denominator = source_norm if source_norm > 0.0 else (_norm(kernel) or 1.0)
        fixedpoint = _difference_norm(target, kernel) / denominator
        gamma_norm = _norm(gamma)
        bse = _norm(tuple(
            vertex - irreducible - (irreducible * line[None, :]) @ vertex
            for vertex, irreducible, line in zip(gamma, kernel, problem.bubble)
        ))
        crossed_gamma = problem.crossing(gamma)
        crossing_error = _norm(tuple(a + b for a, b in zip(gamma, crossed_gamma)))
        hermiticity_error = _norm(tuple(matrix - matrix.conj().T for matrix in gamma))
        record = IterationDiagnostics(
            iteration=iteration, fixedpoint_residual=float(fixedpoint),
            bse_residual=_relative(bse, gamma_norm),
            crossing_defect=_relative(crossing_error, gamma_norm),
            hermiticity_defect=_relative(hermiticity_error, gamma_norm),
        )
        history.append(record)
        if progress is not None:
            progress(record)
        if record.fixedpoint_residual <= threshold:
            converged = True
            break
        if iteration < maximum_iterations:
            kernel = tuple((1.0 - fraction) * current + fraction * desired
                           for current, desired in zip(kernel, target))
    return StaticParquetResult(
        problem=problem, gamma=gamma, kernel=kernel, phi=phi,
        history=tuple(history), converged=converged, interaction_scale=scale,
    )


__all__ = [
    "StaticParquetProblem", "StaticParquetResult", "IterationDiagnostics",
    "build_problem", "estimate_problem_memory", "solve_parquet",
]
