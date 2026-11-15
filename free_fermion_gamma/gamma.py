r"""Matrix-free solution of ``Gamma = U + K Gamma`` at fixed transfer Q.

For composite index ``x=(q,u,v)``, the projected antisymmetrized interaction is

.. math::

 U^Q_{uv,ij}(q,p) = V_H(-Q)d_{uv}(q)d_{ij}(p)^*
 -V_F(p-q)\langle u_{q-Q}|i_{p-Q}\rangle\langle j_p|v_q\rangle,

where ``d_uv(q)=<u(q-Q)|v(q)>``.  The discretized response equation is

``(I - w U L) Gamma = U``.

The Hartree and Fock interactions are independently replaceable; both default
to the bare gated Coulomb interaction.  ``L`` is the static divided difference
and is non-positive.  The pole kernel
used for stability is the Hermitian matrix
``C=-w sqrt(-L) U sqrt(-L)``; a response pole occurs at eigenvalue one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
from scipy import fft
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigsh, gmres

from .model import FreeFermionSystem, fermi, hex_norm


Array = np.ndarray
Approximation = Literal["active-only", "active-associated", "full-8band"]
InteractionLookup = Callable[[Array], Array]


def active_only_pairs(active_band: int) -> Array:
    active_band = int(active_band)
    if not 0 <= active_band < 8:
        raise ValueError("active_band must lie in 0..7")
    return np.asarray([(active_band, active_band)], dtype=np.int64)


def active_associated_pairs(active_band: int) -> Array:
    """Ordered pairs ``(n,m)`` for which ``n=a`` or ``m=a`` (15 total)."""

    active_band = int(active_band)
    if not 0 <= active_band < 8:
        raise ValueError("active_band must lie in 0..7")
    pairs = [(active_band, m) for m in range(8)]
    pairs.extend((n, active_band) for n in range(8) if n != active_band)
    return np.asarray(pairs, dtype=np.int64)


def all_band_pairs() -> Array:
    return np.asarray([(n, m) for n in range(8) for m in range(8)], dtype=np.int64)


def _validate_pairs(pairs: Array) -> Array:
    value = np.asarray(pairs, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2 or np.any((value < 0) | (value >= 8)):
        raise ValueError("band pairs must have shape (N,2) with entries in 0..7")
    if len({tuple(pair) for pair in value}) != len(value):
        raise ValueError("band pairs must be unique")
    return value


def particle_hole_line(
    left_energy: Array,
    right_energy: Array,
    *,
    mu: float,
    temperature: float,
    degeneracy_tolerance: float = 1.0e-12,
) -> Array:
    """Stable finite-temperature divided difference, including ``f'`` limit."""

    left = np.asarray(left_energy, dtype=float) - float(mu)
    right = np.asarray(right_energy, dtype=float) - float(mu)
    if left.shape != right.shape:
        raise ValueError("left and right energies must have the same shape")
    difference = left - right
    left_f = fermi(left, 0.0, temperature)
    right_f = fermi(right, 0.0, temperature)
    result = np.empty_like(difference)
    regular = np.abs(difference) > float(degeneracy_tolerance)
    np.divide(left_f - right_f, difference, out=result, where=regular)
    midpoint_f = fermi(0.5 * (left + right), 0.0, temperature)
    result[~regular] = -midpoint_f[~regular] * (1.0 - midpoint_f[~regular]) / temperature
    return result


class _OpenHexConvolver:
    """Zero-padded linear convolution on the integer box containing H_M."""

    def __init__(
        self,
        system: FreeFermionSystem,
        interaction: InteractionLookup,
    ) -> None:
        self.radius = system.radius
        self.side = 2 * self.radius + 1
        delta = np.arange(-2 * self.radius, 2 * self.radius + 1, dtype=np.int64)
        i, j = np.meshgrid(delta, delta, indexing="ij")
        coords = np.stack((i, j), axis=-1)
        # Only H_(2M) can occur as the difference of two points in H_M.
        # Keeping the rest of the embedding box at zero lets tabulated
        # interactions reject out-of-domain lookups without affecting the
        # exact open-mesh convolution.
        valid = hex_norm(coords) <= 2 * self.radius
        kernel = np.zeros(coords.shape[:-1], dtype=np.complex128)
        forward = np.asarray(interaction(coords[valid]), dtype=np.complex128)
        reverse = np.asarray(interaction(-coords[valid]), dtype=np.complex128)
        scale = max(1.0, float(np.max(np.abs(forward), initial=0.0)))
        if np.max(np.abs(forward - reverse.conj()), initial=0.0) > 1.0e-9 * scale:
            raise ValueError(
                "exchange_interaction must satisfy V(P)=conj(V(-P)) on H_(2M)"
            )
        # Linear convolution produces sum_p h(q-p)x(p).  The kernel therefore
        # stores V_F(-(q-p)) so the projected action uses the stated V_F(p-q).
        kernel[valid] = reverse
        if not np.all(np.isfinite(kernel)):
            raise ValueError("exchange interaction contains non-finite values")
        self.kernel_side = len(delta)
        full_side = self.side + self.kernel_side - 1
        self.fft_shape = (fft.next_fast_len(full_side), fft.next_fast_len(full_side))
        self.kernel_fft = fft.fftn(kernel, s=self.fft_shape)
        self.crop_start = (self.kernel_side - 1) // 2

    def apply(self, values: Array) -> Array:
        values = np.asarray(values, dtype=np.complex128)
        if values.shape[:2] != (self.side, self.side):
            raise ValueError("convolution input has the wrong integer-box shape")
        spectrum = fft.fftn(values, s=self.fft_shape, axes=(0, 1))
        tail = (None,) * (values.ndim - 2)
        convolved = fft.ifftn(
            spectrum * self.kernel_fft[(slice(None), slice(None)) + tail],
            axes=(0, 1),
        )
        start = self.crop_start
        stop = start + self.side
        return convolved[start:stop, start:stop]


@dataclass(frozen=True)
class SolverDiagnostics:
    source_coord: tuple[int, int]
    source_pair: tuple[int, int]
    method: str
    info: int
    iterations: int
    relative_residual: float


@dataclass(frozen=True)
class GammaColumns:
    """Selected columns of the reconstructed vertex.

    ``values`` is indexed as ``[output_momentum, output_pair, source]``.
    """

    values: Array
    output_coords: Array
    output_pairs: Array
    source_coords: Array
    source_pairs: Array
    diagnostics: tuple[SolverDiagnostics, ...]


class GammaSector:
    """One fixed-Q free-fermion Gamma problem in either requested approximation."""

    def __init__(
        self,
        system: FreeFermionSystem,
        transfer: Iterable[int],
        approximation: Approximation,
        *,
        hartree_interaction: InteractionLookup | None = None,
        exchange_interaction: InteractionLookup | None = None,
        loop_tolerance: float = 1.0e-14,
        degeneracy_tolerance: float = 1.0e-12,
    ) -> None:
        self.system = system
        self.transfer = np.asarray(tuple(transfer), dtype=np.int64)
        if self.transfer.shape != (2,):
            raise ValueError("transfer must contain two integers")
        if approximation == "active-only":
            pairs = active_only_pairs(system.active_band_index)
        elif approximation == "active-associated":
            pairs = active_associated_pairs(system.active_band_index)
        elif approximation == "full-8band":
            pairs = all_band_pairs()
        else:
            raise ValueError(
                "approximation must be 'active-only', 'active-associated', "
                "or 'full-8band'"
            )
        self.approximation = approximation
        self.pairs = pairs
        self.right_rows, self.left_rows = system.domain(self.transfer)
        self.domain_coords = system.coords[self.right_rows]
        self.left_spinors = system.spinors[self.left_rows]
        self.right_spinors = system.spinors[self.right_rows]
        self.left_energies = system.energies[self.left_rows]
        self.right_energies = system.energies[self.right_rows]
        self._position = {tuple(coord): i for i, coord in enumerate(self.domain_coords)}
        self._density_overlaps = np.einsum(
            "kan,kam->knm", self.left_spinors.conj(), self.right_spinors, optimize=True
        )
        n, m = self.pairs.T
        self.loop = particle_hole_line(
            self.left_energies[:, n],
            self.right_energies[:, m],
            mu=system.chemical_potential,
            temperature=system.config.temperature_eV,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        if not np.all(np.isfinite(self.loop)):
            raise ValueError("particle-hole line contains non-finite values")
        positive = float(np.max(self.loop, initial=0.0))
        if positive > max(1.0e-11, 10.0 * degeneracy_tolerance):
            raise ValueError(f"static particle-hole line must be non-positive; max={positive}")
        self.loop = np.minimum(self.loop, 0.0)
        self.loop_tolerance = float(loop_tolerance)
        if not np.isfinite(self.loop_tolerance) or self.loop_tolerance < 0.0:
            raise ValueError("loop_tolerance must be finite and non-negative")
        flat_loop = self.loop.reshape(-1)
        self.support = np.flatnonzero(-flat_loop > self.loop_tolerance)
        self.sqrt_response = np.sqrt(-flat_loop[self.support])
        # Hartree and Fock may be tuned independently.  Omitting either lookup
        # preserves the original bare gated-Coulomb interaction for that term.
        self.hartree_interaction = (
            system.interaction if hartree_interaction is None else hartree_interaction
        )
        self.exchange_interaction = (
            system.interaction if exchange_interaction is None else exchange_interaction
        )
        for name, interaction in (
            ("hartree_interaction", self.hartree_interaction),
            ("exchange_interaction", self.exchange_interaction),
        ):
            probe = np.asarray(interaction(np.zeros((1, 2), dtype=np.int64)))
            if probe.shape != (1,) or not np.all(np.isfinite(probe)):
                raise ValueError(f"{name} must return one finite value per transfer")
        self._convolver = _OpenHexConvolver(system, self.exchange_interaction)
        self._bare_v_minus_q = complex(system.interaction(-self.transfer))
        self._hartree_minus_q = complex(self.hartree_interaction(-self.transfer))
        if not np.isfinite(self._hartree_minus_q):
            raise ValueError("hartree_interaction(-Q) must be finite")
        if abs(self._hartree_minus_q.imag) > 1.0e-9 * max(
            1.0, abs(self._hartree_minus_q)
        ):
            raise ValueError("hartree_interaction(-Q) must be real for a Hermitian kernel")
        self._hartree_minus_q = complex(self._hartree_minus_q.real)
        self._pole_operator = LinearOperator(
            (len(self.support), len(self.support)),
            matvec=self._pole_matvec,
            rmatvec=self._pole_matvec,
            dtype=np.dtype(np.complex128),
        )

    @property
    def domain_size(self) -> int:
        return len(self.domain_coords)

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def composite_size(self) -> int:
        return self.domain_size * self.pair_count

    @property
    def support_size(self) -> int:
        return len(self.support)

    @property
    def pole_operator(self) -> LinearOperator:
        return self._pole_operator

    @property
    def bare_hartree_strength(self) -> complex:
        """The bare reference value ``V0(-Q)``, whether or not it is selected."""

        return self._bare_v_minus_q

    @property
    def hartree_strength(self) -> complex:
        """The actual direct factor ``V_H(-Q)`` used by this sector."""

        return self._hartree_minus_q

    @property
    def response_operator(self) -> LinearOperator:
        return LinearOperator(
            self._pole_operator.shape,
            matvec=lambda x: np.asarray(x) - self._pole_matvec(x),
            rmatvec=lambda x: np.asarray(x) - self._pole_matvec(x),
            dtype=np.dtype(np.complex128),
        )

    def position(self, coord: Iterable[int]) -> int:
        key = tuple(int(x) for x in coord)
        try:
            return self._position[key]
        except KeyError as error:
            raise ValueError(f"momentum {key} is not in D_Q") from error

    def _atomic_field(self, values: Array, input_pairs: Array) -> Array:
        input_pairs = _validate_pairs(input_pairs)
        x = np.asarray(values, dtype=np.complex128)
        if x.shape != (self.domain_size, len(input_pairs)):
            raise ValueError("band-pair field has the wrong shape")
        band = np.zeros((self.domain_size, 8, 8), dtype=np.complex128)
        band[:, input_pairs[:, 0], input_pairs[:, 1]] = x
        return np.einsum(
            "kan,knm,kbm->kab",
            self.left_spinors,
            band,
            self.right_spinors.conj(),
            optimize=True,
        )

    def apply_interaction(
        self,
        values: Array,
        *,
        input_pairs: Array | None = None,
        output_pairs: Array | None = None,
    ) -> Array:
        """Apply projected ``U=Hartree-Fock`` without constructing it densely."""

        input_pairs = self.pairs if input_pairs is None else _validate_pairs(input_pairs)
        output_pairs = self.pairs if output_pairs is None else _validate_pairs(output_pairs)
        x = np.asarray(values, dtype=np.complex128).reshape(
            self.domain_size, len(input_pairs)
        )
        atomic = self._atomic_field(x, input_pairs)
        radius = self.system.radius
        side = 2 * radius + 1
        box = np.zeros((side, side, 8, 8), dtype=np.complex128)
        box[
            self.domain_coords[:, 0] + radius,
            self.domain_coords[:, 1] + radius,
        ] = atomic
        convolved_box = self._convolver.apply(box)
        convolved = convolved_box[
            self.domain_coords[:, 0] + radius,
            self.domain_coords[:, 1] + radius,
        ]
        projected = np.einsum(
            "kau,kab,kbv->kuv",
            self.left_spinors.conj(),
            convolved,
            self.right_spinors,
            optimize=True,
        )
        exchange = projected[:, output_pairs[:, 0], output_pairs[:, 1]]
        d_in = self._density_overlaps[:, input_pairs[:, 0], input_pairs[:, 1]]
        d_out = self._density_overlaps[:, output_pairs[:, 0], output_pairs[:, 1]]
        hartree = self._hartree_minus_q * d_out * np.vdot(d_in, x)
        return hartree - exchange

    def interaction_column(
        self,
        source_position: int,
        source_pair: Sequence[int],
        *,
        output_pairs: Array | None = None,
    ) -> Array:
        """Construct ``U[:,source]`` for an arbitrary external band pair."""

        output_pairs = self.pairs if output_pairs is None else _validate_pairs(output_pairs)
        source_position = int(source_position)
        if not 0 <= source_position < self.domain_size:
            raise ValueError("source_position is outside D_Q")
        source_pair_array = _validate_pairs(np.asarray(source_pair).reshape(1, 2))
        i, j = map(int, source_pair_array[0])
        u, v = output_pairs.T
        left_source = self.left_spinors[source_position, :, i]
        right_source = self.right_spinors[source_position, :, j]
        left_out = self.left_spinors[:, :, u]
        right_out = self.right_spinors[:, :, v]
        left_overlap = np.einsum("kob,o->kb", left_out.conj(), left_source, optimize=True)
        right_overlap = np.einsum(
            "o,kob->kb", right_source.conj(), right_out, optimize=True
        )
        delta = self.domain_coords[source_position] - self.domain_coords
        exchange = self.exchange_interaction(delta)[:, None] * left_overlap * right_overlap
        d_source = self._density_overlaps[source_position, i, j]
        d_out = self._density_overlaps[:, u, v]
        hartree = self._hartree_minus_q * d_out * np.conj(d_source)
        return hartree - exchange

    def _interaction_matvec(self, vector: Array) -> Array:
        return self.apply_interaction(np.asarray(vector).reshape(self.domain_size, -1)).reshape(-1)

    def _pole_matvec(self, vector: Array) -> Array:
        vector = np.asarray(vector, dtype=np.complex128)
        if vector.shape != (self.support_size,):
            raise ValueError("pole-kernel vector has the wrong shape")
        full = np.zeros(self.composite_size, dtype=np.complex128)
        full[self.support] = self.sqrt_response * vector
        acted = self._interaction_matvec(full)[self.support]
        return -self.system.weight * self.sqrt_response * acted

    @staticmethod
    def _dense_from_operator(operator: LinearOperator) -> Array:
        size = operator.shape[0]
        eye = np.eye(size, dtype=np.complex128)
        return np.column_stack([operator.matvec(eye[:, column]) for column in range(size)])

    def dense_pole_kernel(self, *, maximum_size: int = 4096) -> Array:
        if self.support_size > int(maximum_size):
            raise MemoryError(
                f"support dimension {self.support_size} exceeds dense guard {maximum_size}"
            )
        kernel = self._dense_from_operator(self.pole_operator)
        return 0.5 * (kernel + kernel.conj().T)

    def largest_eigenvalues(
        self,
        count: int = 4,
        *,
        dense_cutoff: int = 192,
        tolerance: float = 1.0e-9,
        maxiter: int | None = None,
        initial_vector: Array | None = None,
    ) -> Array:
        """Largest algebraic pole-kernel eigenvalues, sorted ascending."""

        size = self.support_size
        count = int(count)
        if count <= 0:
            raise ValueError("count must be positive")
        if initial_vector is not None:
            initial_vector = np.asarray(initial_vector, dtype=np.complex128)
            if initial_vector.shape != (size,) or not np.all(np.isfinite(initial_vector)):
                raise ValueError("initial_vector must be finite with shape (support_size,)")
            if size and np.linalg.norm(initial_vector) == 0.0:
                raise ValueError("initial_vector must be nonzero")
        if size == 0:
            return np.empty(0, dtype=float)
        if size <= dense_cutoff or size == 1:
            return np.linalg.eigvalsh(self.dense_pole_kernel(maximum_size=dense_cutoff))[
                -min(count, size) :
            ]
        k = min(max(1, int(count)), size - 1)
        values = eigsh(
            self.pole_operator,
            k=k,
            which="LA",
            return_eigenvectors=False,
            tol=tolerance,
            maxiter=maxiter,
            v0=initial_vector,
        )
        return np.sort(np.asarray(values, dtype=float))

    def solve_columns(
        self,
        source_coords: Array,
        source_pairs: Array,
        *,
        output_coords: Array | None = None,
        output_pairs: Array | None = None,
        dense_cutoff: int = 256,
        relative_tolerance: float = 1.0e-10,
        absolute_tolerance: float = 0.0,
        restart: int | None = 80,
        maxiter: int | None = None,
    ) -> GammaColumns:
        """Solve and reconstruct selected columns of the full external Gamma.

        Only internal loops are truncated.  ``source_pairs`` and
        ``output_pairs`` may contain any of the 64 external band pairs, as
        required by the note's reconstruction step.
        """

        source_coords = np.asarray(source_coords, dtype=np.int64)
        source_pairs = _validate_pairs(source_pairs)
        if source_coords.ndim != 2 or source_coords.shape[1] != 2:
            raise ValueError("source_coords must have shape (Nsource,2)")
        if len(source_coords) != len(source_pairs):
            raise ValueError("source_coords and source_pairs must have equal length")
        source_positions = np.asarray([self.position(coord) for coord in source_coords])
        output_pairs = self.pairs if output_pairs is None else _validate_pairs(output_pairs)
        if output_coords is None:
            output_positions = np.arange(self.domain_size)
            selected_output_coords = self.domain_coords.copy()
        else:
            selected_output_coords = np.asarray(output_coords, dtype=np.int64)
            if selected_output_coords.ndim != 2 or selected_output_coords.shape[1] != 2:
                raise ValueError("output_coords must have shape (Noutput,2)")
            output_positions = np.asarray(
                [self.position(coord) for coord in selected_output_coords], dtype=np.int64
            )
        values = np.empty(
            (len(output_positions), len(output_pairs), len(source_positions)),
            dtype=np.complex128,
        )
        diagnostics: list[SolverDiagnostics] = []

        dense_factor = None
        if 0 < self.support_size <= int(dense_cutoff):
            response = np.eye(self.support_size, dtype=np.complex128) - self.dense_pole_kernel(
                maximum_size=dense_cutoff
            )
            dense_factor = lu_factor(response, check_finite=False)

        for column, (position, pair) in enumerate(zip(source_positions, source_pairs)):
            source_internal = self.interaction_column(position, pair)
            if self.support_size:
                rhs = self.sqrt_response * source_internal.reshape(-1)[self.support]
                if np.linalg.norm(rhs) == 0.0:
                    transformed = np.zeros_like(rhs)
                    info, iterations, method = 0, 0, "zero-rhs"
                elif dense_factor is not None:
                    transformed = lu_solve(dense_factor, rhs, check_finite=False)
                    info, iterations, method = 0, 1, "dense-lu"
                else:
                    iterations = 0

                    def count_iteration(_value: float) -> None:
                        nonlocal iterations
                        iterations += 1

                    transformed, info = gmres(
                        self.response_operator,
                        rhs,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                        restart=restart,
                        maxiter=maxiter,
                        callback=count_iteration,
                        callback_type="pr_norm",
                    )
                    method = "gmres"
                    if info != 0:
                        raise np.linalg.LinAlgError(
                            f"Gamma solve failed for source {tuple(source_coords[column])}, "
                            f"pair {tuple(pair)}; GMRES info={info}"
                        )
                residual_vector = self.response_operator.matvec(transformed) - rhs
                residual = float(
                    np.linalg.norm(residual_vector) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
                )
                gamma_support = transformed / self.sqrt_response
                response_weighted = np.zeros(self.composite_size, dtype=np.complex128)
                response_weighted[self.support] = (
                    self.sqrt_response**2 * gamma_support
                )
                correction = self.apply_interaction(
                    response_weighted.reshape(self.domain_size, self.pair_count),
                    output_pairs=output_pairs,
                )
            else:
                info, iterations, method, residual = 0, 0, "empty-loop", 0.0
                correction = np.zeros(
                    (self.domain_size, len(output_pairs)), dtype=np.complex128
                )
            source_output = self.interaction_column(
                position, pair, output_pairs=output_pairs
            )
            reconstructed = source_output - self.system.weight * correction
            values[:, :, column] = reconstructed[output_positions]
            diagnostics.append(
                SolverDiagnostics(
                    source_coord=tuple(int(x) for x in source_coords[column]),
                    source_pair=tuple(int(x) for x in pair),
                    method=method,
                    info=int(info),
                    iterations=int(iterations),
                    relative_residual=residual,
                )
            )
        return GammaColumns(
            values=values,
            output_coords=selected_output_coords,
            output_pairs=output_pairs.copy(),
            source_coords=source_coords.copy(),
            source_pairs=source_pairs.copy(),
            diagnostics=tuple(diagnostics),
        )


__all__ = [
    "GammaColumns",
    "GammaSector",
    "SolverDiagnostics",
    "active_associated_pairs",
    "active_only_pairs",
    "all_band_pairs",
    "particle_hole_line",
]
