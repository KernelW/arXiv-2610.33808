"""Solve the static channel-irreducible Gamma^ph as the only unknown.

R(K)=K-gU+C Phi(K), with (1-KB) Phi=(KB)K. The source U is
antisymmetrized bare interaction. The default seed screens *both* HF legs
with W_g=gV/(1-gV Pi_full8); screening changes the seed, never the equation.
No full Gamma tensor is returned or saved by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Sequence
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres, lgmres

from free_fermion_gamma.rpa import HexInteractionTable, compute_band_pair_rpa
from .solver import Blocks, StaticParquetProblem, _norm, _relative


_AUTO_STREAMED_ENTRIES = 100_000


def _memory_mode(value: str, entries: int) -> str:
    if value not in ("auto", "cached", "streamed"):
        raise ValueError("memory_mode must be 'auto', 'cached', or 'streamed'")
    return ("streamed" if entries > _AUTO_STREAMED_ENTRIES else "cached") if value == "auto" else value


def _views(problem: StaticParquetProblem, flat: np.ndarray) -> Blocks:
    """Views of an internally owned, already validated complex buffer."""
    return tuple(flat[int(problem.offsets[i]):int(problem.offsets[i + 1])].reshape(len(coords), len(coords))
                 for i, coords in enumerate(problem.domain_coords))


def _shared_flat(problem: StaticParquetProblem, blocks: Sequence[np.ndarray]) -> np.ndarray:
    # The domain implementation recognizes a tuple of views of one buffer.
    return problem.pack(blocks, copy=False)


def _destinations(problem: StaticParquetProblem, sector: int) -> np.ndarray:
    return problem.crossing_destinations(sector)


def estimate_gamma_ph_memory(
    problem_or_diagnostics: StaticParquetProblem | Mapping, *,
    krylov_restart: int = 8, memory_mode: str = "auto", retained_seed_count: int = 1,
    preserve_physical_symmetry: bool = False,
    krylov_method: str = "gmres", krylov_outer_k: int = 3,
) -> dict:
    """Conservative live-array planning estimate, not measured process RSS.

    Accepts preconstruction diagnostics so a caller can guard the entire run
    before allocating the problem. The model counts the GMRES basis, solver
    work vectors, cached derivative data when requested, a caller-retained
    continuation seed, and sector scratch. Python/BLAS allocator retention,
    imported libraries, and unrelated caller arrays are outside this bound.
    """
    for name, value in (("krylov_restart", krylov_restart), ("retained_seed_count", retained_seed_count)):
        minimum = 1 if name == "krylov_restart" else 0
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if krylov_method not in ("gmres", "lgmres"):
        raise ValueError("krylov_method must be 'gmres' or 'lgmres'")
    if (not isinstance(krylov_outer_k, (int, np.integer))
            or isinstance(krylov_outer_k, (bool, np.bool_)) or krylov_outer_k < 0):
        raise ValueError("krylov_outer_k must be a non-negative integer")
    diagnostics = problem_or_diagnostics if isinstance(problem_or_diagnostics, Mapping) else problem_or_diagnostics.memory_diagnostics
    entries = int(diagnostics["total_vertex_entries"])
    selected = _memory_mode(memory_mode, entries)
    vertex = int(diagnostics.get("vertex_bytes", 16 * entries))
    persistent = int(diagnostics.get("persistent_bytes", vertex + int(diagnostics.get("crossing_bytes", 8 * entries))))
    construction = int(diagnostics.get("construction_peak_bytes", persistent + 4 * vertex))
    # Basis restart+1; K, residual, rhs, x, Arnoldi residual and its last
    # operator output. During r=b-matvec(x), SciPy also keeps the previous r
    # and operator output alive while constructing the new output/residual.
    # A caller can independently retain its continuation seed.
    vector_count = int(krylov_restart) + 9 + int(retained_seed_count)
    restart = int(krylov_restart)
    krylov_small_space = 8 * restart * (restart + 1) + 24 * restart + 8
    if krylov_method == "lgmres":
        # One SciPy outer cycle per call releases V/Z before the next cycle.
        # With no right preconditioner Z aliases V/recycled directions. Count
        # V (restart+outer_k+1), retained directions and append transient, old
        # and new x during x0 copying, rhs/residual/dx/output work buffers.
        # No cached A*v is stored. Ordinary multi-cycle scipy.lgmres can keep
        # the previous V/Z alive while constructing the next basis; our
        # single-cycle wrapper is essential to this memory bound.
        vector_count = restart + 2 * int(krylov_outer_k) + 12 + int(retained_seed_count)
        width = restart + int(krylov_outer_k) + 2
        krylov_small_space = 64 * width * width + 64 * width
    derivative_bytes = 0 if selected == "streamed" else 2 * vertex
    # Cached mode also retains the prior K/R for the inverse-secant fallback.
    secant_bytes = 0 if selected == "streamed" else 2 * vertex
    largest = int(diagnostics["largest_sector_size"])
    sector_scratch = 12 * 16 * largest * largest + 64 * largest
    # The optional joint H/P projection owns only domain-size reindex maps and
    # two sector matrices. The usual K plus retained-seed count already covers
    # copying an immutable supplied seed before projecting it.
    symmetry_bytes = (2 * 16 * largest * largest
                      + 16 * int(diagnostics.get("domain_entry_count", largest))
                      + 256 * int(diagnostics.get("number_of_sectors", 1))) if preserve_physical_symmetry else 0
    solver_peak = persistent + vector_count * vertex + derivative_bytes + secant_bytes + sector_scratch + krylov_small_space + symmetry_bytes
    peak = max(construction, solver_peak)
    return {
        "memory_mode_requested": memory_mode, "memory_mode": selected,
        "total_vertex_entries": entries, "vertex_bytes": vertex,
        "persistent_problem_bytes": persistent, "construction_peak_bytes": construction,
        "gmres_restart": int(krylov_restart), "working_vector_count": vector_count,
        "krylov_method": krylov_method, "krylov_outer_k": int(krylov_outer_k),
        "krylov_small_space_bytes": krylov_small_space,
        "cached_derivative_bytes": derivative_bytes, "secant_history_bytes": secant_bytes,
        "retained_seed_count": int(retained_seed_count), "sector_scratch_bytes": sector_scratch,
        "preserve_physical_symmetry": bool(preserve_physical_symmetry),
        "physical_symmetry_bytes": symmetry_bytes,
        "solver_peak_bytes": solver_peak, "estimated_peak_bytes": peak,
        "estimated_peak_mb": peak / 1.0e6,
        "estimate_kind": "conservative live-array planning estimate; excludes allocator retention and imported libraries",
    }


class _PhysicalSymmetryProjector:
    """Joint real orthogonal projection onto Hermiticity and pair interchange.

    H X_Q(q,p)=X_Q(p,q)* and P X_Q(q,p)=X_-Q(p-Q,q-Q).
    H and P commute. Crossing obeys HC=CHP, so H alone is insufficient.
    No full tensor permutation or projected residual is constructed here.
    """

    def __init__(self, problem: StaticParquetProblem):
        self.problem = problem
        sectors = problem._sector_indices
        positions = problem._position_maps
        self.opposite = tuple(sectors[tuple(-q)] for q in problem.transfers)
        self.shift = tuple(np.asarray([positions[opposite][tuple(q - transfer)] for q in coords], dtype=np.intp)
                           for transfer, coords, opposite in zip(problem.transfers, problem.domain_coords, self.opposite))
        for index, opposite in enumerate(self.opposite):
            if not np.array_equal(self.shift[opposite][self.shift[index]], np.arange(len(self.shift[index]))):
                raise ValueError("physical pair interchange must be an involution")
            if not np.allclose(problem.bubble[index], problem.bubble[opposite][self.shift[index]],
                               rtol=1.0e-12, atol=1.0e-15):
                raise ValueError("particle-hole bubbles violate physical pair interchange")
        hermiticity, interchange = self.defects(problem.bare)
        if max(hermiticity, interchange) > 1.0e-10:
            raise ValueError("bare source violates the requested physical H/P symmetry")

    def pair_block(self, blocks: Sequence[np.ndarray], index: int) -> np.ndarray:
        shift = self.shift[index]
        return blocks[self.opposite[index]][np.ix_(shift, shift)].T

    def defects(self, blocks: Sequence[np.ndarray]) -> tuple[float, float]:
        norm = _norm(blocks)
        hermitian_squared = interchange_squared = 0.0
        for index, matrix in enumerate(blocks):
            difference = matrix - matrix.conj().T
            hermitian_squared += float(np.vdot(difference, difference).real)
            difference = matrix - self.pair_block(blocks, index)
            interchange_squared += float(np.vdot(difference, difference).real)
        return (_relative(np.sqrt(hermitian_squared), norm),
                _relative(np.sqrt(interchange_squared), norm))

    def project_inplace(self, blocks: Sequence[np.ndarray]) -> None:
        for matrix in blocks:
            matrix += matrix.conj().T
            matrix *= 0.5
        for index, opposite in enumerate(self.opposite):
            if index > opposite:
                continue
            matrix = blocks[index]
            matrix += self.pair_block(blocks, index)
            matrix *= 0.5
            if opposite != index:
                blocks[opposite][:] = self.pair_block(blocks, opposite)


def project_physical_symmetry(problem: StaticParquetProblem, kernel: Sequence[np.ndarray]) -> Blocks:
    """Return an owned physical H/P projection; never mutate the supplied K.

    This restricts the unknown to its physical symmetry subspace. The nonlinear
    equations and their full residual remain unchanged.
    """
    checked = problem._validate_blocks(kernel, "kernel")
    projected = _views(problem, problem.pack(checked))
    _PhysicalSymmetryProjector(problem).project_inplace(projected)
    return projected


def _factor_response(matrix: np.ndarray, bubble: np.ndarray, transfer) -> tuple:
    response = np.array(matrix, dtype=np.complex128, order="F", copy=True)
    response *= -bubble[None, :]
    response[np.diag_indices(len(matrix))] += 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            return lu_factor(response, overwrite_a=True, check_finite=False)
    except (np.linalg.LinAlgError, LinAlgWarning) as error:
        raise np.linalg.LinAlgError(f"Gamma^ph trial BSE is singular at Q={tuple(transfer)}") from error


def _phi_block(matrix: np.ndarray, bubble: np.ndarray, factor: tuple) -> np.ndarray:
    weighted = np.array(matrix, order="F", copy=True)
    weighted *= bubble[:, None]
    phi = np.empty_like(weighted, order="F")
    np.matmul(matrix, weighted, out=phi)
    del weighted
    phi = lu_solve(factor, phi, overwrite_b=True, check_finite=False)
    if not np.all(np.isfinite(phi)):
        raise np.linalg.LinAlgError("Gamma^ph trial produced a non-finite reducible vertex")
    return phi


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or (not zero and number == 0.0):
        raise ValueError(f"{name} must be finite and {'non-negative' if zero else 'positive'}")
    return number


def _pole_data(problem: StaticParquetProblem, kernel: Blocks) -> tuple[float, float]:
    largest, gap = 0.0, float("inf")
    for matrix, bubble in zip(kernel, problem.bubble):
        if np.any(bubble > 1.0e-12) or not np.all(np.isfinite(bubble)):
            raise ValueError("weighted static particle-hole bubbles must be finite and non-positive")
        root = np.sqrt(np.maximum(-bubble, 0.0))
        pole = -root[:, None] * matrix * root[None, :]
        hermitian_error = np.linalg.norm(pole - pole.conj().T)
        if hermitian_error <= 1.0e-11 * max(1.0, np.linalg.norm(pole)):
            eigenvalues = np.linalg.eigvalsh(0.5 * (pole + pole.conj().T))
        else:
            eigenvalues = np.linalg.eigvals(pole)
        largest = max(largest, float(np.max(eigenvalues.real, initial=0.0)))
        gap = min(gap, float(np.min(np.abs(1.0 - eigenvalues))))
    return largest, gap


def _guard_seed(problem: StaticParquetProblem, seed: Blocks, ceiling: float) -> tuple[Blocks, dict]:
    if not np.isfinite(ceiling) or not 0.0 < ceiling < 1.0:
        raise ValueError("pole_ceiling must lie strictly between zero and one")
    largest, gap = _pole_data(problem, seed)
    factor = min(1.0, ceiling / largest) if largest > 0.0 else 1.0
    # The caller owns this newly constructed seed, so scaling needs no second
    # whole vertex. Supplied continuation kernels never pass through this guard.
    if factor != 1.0:
        for block in seed:
            block *= factor
    limited = seed
    after, after_gap = _pole_data(problem, limited)
    return limited, {
        "raw_lambda_max": largest, "raw_pole_margin": gap,
        "screened_seed_scale": factor, "seed_scale": factor,
        "pole_ceiling": ceiling, "scaled_lambda_max": after,
        "initial_pole_margin": after_gap,
    }


def build_rpa_initial(
    problem: StaticParquetProblem, *, interaction_scale: float = 1.0,
    screening: str = "full-8band", pole_ceiling: float = 0.8,
) -> tuple[Blocks, dict]:
    """Construct a screened HF seed; apply a seed-only pole-margin guard."""
    scale = _positive(interaction_scale, "interaction_scale", zero=True)
    result = compute_band_pair_rpa(problem.system, screening=screening)
    denominator = 1.0 - scale * result.bare_interaction * result.polarization
    if not np.all(np.isfinite(denominator)) or np.min(np.abs(denominator)) < 1.0e-12:
        raise np.linalg.LinAlgError("the scaled RPA initial interaction has a singular denominator")
    interaction = HexInteractionTable(
        2 * problem.system.radius, result.transfer_coords,
        scale * result.bare_interaction / denominator,
    )
    seed = _views(problem, np.empty(problem.total_entries, dtype=np.complex128))
    active = problem.system.active_band_index
    for block, transfer, coords, rows, shifted in zip(
        seed, problem.transfers, problem.domain_coords, problem.right_rows, problem.left_rows,
    ):
        left = problem.system.spinors[shifted, :, active]
        right = problem.system.spinors[rows, :, active]
        density = np.sum(left.conj() * right, axis=1)
        delta = coords[None, :, :] - coords[:, None, :]
        block[:] = (
            complex(interaction(-transfer)) * np.outer(density, density.conj())
            - interaction(delta) * (left.conj() @ left.T) * (right @ right.conj().T)
        )
    limited, diagnostics = _guard_seed(problem, seed, pole_ceiling)
    diagnostics.update({
        "kind": "RPA-screened HF", "screening_scheme": result.screening_scheme,
        "screening_pair_count": len(result.screening_pairs),
        "interaction_scale": scale,
        "minimum_rpa_denominator": float(np.min(np.abs(denominator))),
        "hartree_interaction": "rpa", "fock_interaction": "rpa",
        "source_interaction": "antisymmetrized bare gU",
    })
    return limited, diagnostics


@dataclass
class GammaPhEvaluation:
    """Residual and transient derivative context; no full Gamma output."""
    problem: StaticParquetProblem = field(repr=False)
    residual: Blocks
    residual_norm: float
    relative_residual: float
    pole_margin: float
    residual_flat: np.ndarray = field(repr=False)
    memory_mode: str
    _kernel: Blocks = field(repr=False)
    _jacobian_enabled: bool = field(repr=False)
    _factors: tuple = field(repr=False)
    _right_factors: Blocks = field(repr=False)

    def jvp(self, direction: Sequence[np.ndarray]) -> Blocks:
        """Exact analytic derivative of R; complex amplitudes are preserved."""
        if not self._jacobian_enabled:
            raise ValueError("evaluation was constructed without a Jacobian context")
        delta = self.problem._validate_blocks(direction, "direction")
        out = self.problem.pack(delta)
        for index, (matrix, bubble, change) in enumerate(zip(self._kernel, self.problem.bubble, delta)):
            if self.memory_mode == "cached":
                product = np.array(change @ self._right_factors[index], order="F")
                dgamma = lu_solve(self._factors[index], product, overwrite_b=True, check_finite=False)
            else:
                # L H R through two local factorizations, without a full R
                # inverse or all-sector derivative cache. General complex K
                # is supported; no Hermitian shortcut changes the derivative.
                left = _factor_response(matrix, bubble, self.problem.transfers[index])
                right_transposed = _factor_response(matrix.T, bubble, self.problem.transfers[index])
                product = lu_solve(left, np.array(change, order="F", copy=True), overwrite_b=True, check_finite=False)
                dgamma = lu_solve(right_transposed, np.array(product.T, order="F"), overwrite_b=True, check_finite=False).T
                del left, right_transposed, product
            dgamma -= change
            out[_destinations(self.problem, index)] += dgamma.reshape(-1)
        return _views(self.problem, out)

    def adjoint_jvp(self, direction: Sequence[np.ndarray]) -> Blocks:
        """Frobenius adjoint of the analytic derivative for merit descent."""
        if not self._jacobian_enabled:
            raise ValueError("evaluation was constructed without a Jacobian context")
        delta = self.problem._validate_blocks(direction, "direction")
        packed_delta = _shared_flat(self.problem, delta)
        out = self.problem.pack(delta)
        for index, (matrix, bubble, change) in enumerate(zip(self._kernel, self.problem.bubble, delta)):
            cross = packed_delta[_destinations(self.problem, index)].reshape(matrix.shape)
            if self.memory_mode == "cached":
                product = np.array(cross @ self._right_factors[index].conj().T, order="F")
                acted = lu_solve(self._factors[index], product, trans=2, overwrite_b=True, check_finite=False)
            else:
                left = _factor_response(matrix, bubble, self.problem.transfers[index])
                right_transposed = _factor_response(matrix.T, bubble, self.problem.transfers[index])
                product = lu_solve(left, np.array(cross, order="F", copy=True), trans=2, overwrite_b=True, check_finite=False)
                acted = lu_solve(right_transposed, np.array(product.T, order="F"), trans=2, overwrite_b=True, check_finite=False).T
                del left, right_transposed, product
            acted -= cross
            start, stop = int(self.problem.offsets[index]), int(self.problem.offsets[index + 1])
            out[start:stop] += acted.reshape(-1)
        return _views(self.problem, out)


def evaluate_gamma_ph(
    problem: StaticParquetProblem, kernel: Sequence[np.ndarray], *,
    interaction_scale: float = 1.0, want_jacobian: bool = True,
    need_jacobian: bool | None = None,
    memory_mode: str = "auto",
) -> GammaPhEvaluation:
    """Evaluate R by directly solving Phi, without constructing full Gamma.

    ``need_jacobian`` is an alias for ``want_jacobian``. Singular or nearly
    singular BSE trial points raise LinAlgError and are rejected by the solver.
    """
    if need_jacobian is not None:
        want_jacobian = bool(need_jacobian)
    scale = _positive(interaction_scale, "interaction_scale", zero=True)
    selected = _memory_mode(memory_mode, problem.total_entries)
    checked = problem._validate_blocks(kernel, "kernel")
    _, gap = _pole_data(problem, checked)
    if not np.isfinite(gap) or gap < 1.0e-12:
        raise np.linalg.LinAlgError(f"Gamma^ph trial is at a BSE pole (margin={gap:.3e})")
    residual_flat = np.empty(problem.total_entries, dtype=np.complex128)
    residual = _views(problem, residual_flat)
    for out, matrix, bare in zip(residual, checked, problem.bare):
        out[:] = bare
        out *= -scale
        out += matrix
    factors, right_factors = [], []
    for index, (matrix, bubble) in enumerate(zip(checked, problem.bubble)):
        factor = _factor_response(matrix, bubble, problem.transfers[index])
        block = _phi_block(matrix, bubble, factor)
        residual_flat[_destinations(problem, index)] += block.reshape(-1)
        if want_jacobian and selected == "cached":
            factors.append(factor)
            # Phi is no longer needed after scatter: reuse its sector buffer.
            block += matrix
            block *= bubble[:, None]
            block[np.diag_indices(len(matrix))] += 1.0
            right_factors.append(block)
        else:
            del factor, block
    norm = float(np.linalg.norm(residual_flat))
    if not np.isfinite(norm):
        raise np.linalg.LinAlgError("Gamma^ph trial has a non-finite residual norm")
    source_norm = scale * _norm(problem.bare)
    normalization = source_norm if source_norm > 0.0 else (_norm(checked) or 1.0)
    return GammaPhEvaluation(
        problem=problem, residual=residual, residual_norm=norm,
        relative_residual=norm / normalization, pole_margin=gap,
        residual_flat=residual_flat, memory_mode=selected, _kernel=checked,
        _jacobian_enabled=bool(want_jacobian),
        _factors=tuple(factors), _right_factors=tuple(right_factors),
    )


@dataclass(frozen=True)
class GammaPhIteration:
    iteration: int
    relative_residual: float
    residual_norm: float
    pole_margin: float
    step: float
    krylov_info: int
    krylov_iterations: int
    direction: str
    hermiticity_defect: float = 0.0
    pair_interchange_defect: float | None = None
    krylov_method: str = "gmres"

    @property
    def fixedpoint_residual(self) -> float:
        return self.relative_residual


@dataclass(frozen=True)
class GammaPhResult:
    problem: StaticParquetProblem
    kernel: Blocks
    history: tuple[GammaPhIteration, ...]
    converged: bool
    status: str
    initial_diagnostics: dict
    interaction_scale: float

    @property
    def gamma_ph(self) -> Blocks:
        return self.kernel

    @property
    def final_diagnostics(self) -> GammaPhIteration:
        return self.history[-1]


def _enable_jacobian(evaluation: GammaPhEvaluation) -> GammaPhEvaluation:
    """Promote an accepted residual evaluation without a second flat residual."""
    if evaluation._jacobian_enabled:
        return evaluation
    if evaluation.memory_mode == "cached":
        factors, right_factors = [], []
        for transfer, matrix, bubble in zip(evaluation.problem.transfers, evaluation._kernel, evaluation.problem.bubble):
            factor = _factor_response(matrix, bubble, transfer)
            right = _phi_block(matrix, bubble, factor)
            right += matrix
            right *= bubble[:, None]
            right[np.diag_indices(len(matrix))] += 1.0
            factors.append(factor)
            right_factors.append(right)
        evaluation._factors = tuple(factors)
        evaluation._right_factors = tuple(right_factors)
    evaluation._jacobian_enabled = True
    return evaluation


def _line_search(
    evaluation: GammaPhEvaluation, kernel_flat: np.ndarray, direction: np.ndarray, *,
    interaction_scale: float, maximum_backtracks: int,
    minimum_step: float = 0.0,
    work_emit: Callable | None = None, iteration: int = 0,
    direction_name: str = "Newton--Krylov",
) -> tuple[np.ndarray, GammaPhEvaluation, float] | None:
    minimum_step = _positive(minimum_step, "minimum_step", zero=True)
    if minimum_step > 1.0:
        raise ValueError("minimum_step must lie in [0, 1]")
    started = perf_counter()
    def report(event: str, **fields) -> None:
        if work_emit is not None:
            work_emit(event, iteration=iteration, direction=direction_name, **fields)
    report("line_search_start")
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0.0:
        report("line_search_end", status="invalid_direction", trials=0,
               operation_elapsed_seconds=perf_counter() - started)
        return None
    derivative_blocks = evaluation.jvp(_views(evaluation.problem, direction))
    derivative = _shared_flat(evaluation.problem, derivative_blocks)
    slope = float(np.vdot(evaluation.residual_flat, derivative).real)
    # Reuse this exact Jd evaluation to check the full linear equation.
    # Sector-size temporaries avoid another all-Q work vector.
    linear_norm_squared = 0.0
    for acted, residual in zip(derivative_blocks, evaluation.residual):
        difference = acted + residual
        linear_norm_squared += float(np.vdot(difference, difference).real)
        del difference
    report("direction_check", merit_slope=slope,
           true_linear_relative_residual=float(np.sqrt(linear_norm_squared)
                                               / max(evaluation.residual_norm, np.finfo(float).tiny)),
           relative_correction=float(np.linalg.norm(direction)
                                     / max(np.linalg.norm(kernel_flat), evaluation.residual_norm,
                                           np.finfo(float).tiny)))
    del derivative, derivative_blocks
    if minimum_step == 0.0 and (not np.isfinite(slope) or slope >= 0.0):
        report("line_search_end", status="not_a_descent_direction", trials=0,
               operation_elapsed_seconds=perf_counter() - started)
        return None
    merit = evaluation.residual_norm ** 2 / 2.0 if minimum_step == 0.0 else None
    step = 1.0
    for trial_index in range(1, maximum_backtracks + 1):
        report("line_search_trial", trial=trial_index, step=float(step))
        trial_flat = np.empty_like(kernel_flat)
        np.multiply(direction, step, out=trial_flat)
        trial_flat += kernel_flat
        trial_kernel = _views(evaluation.problem, trial_flat)
        trial = None
        evaluation_started = perf_counter()
        report("evaluation_start", scope="trial", trial=trial_index, step=float(step))
        rejection_reason = "Armijo condition" if minimum_step == 0.0 else "nonlinear residual did not decrease"
        try:
            trial = evaluate_gamma_ph(
                evaluation.problem, trial_kernel, interaction_scale=interaction_scale,
                want_jacobian=False, memory_mode=evaluation.memory_mode,
            )
            trial_merit = trial.residual_norm ** 2 / 2.0 if minimum_step == 0.0 else None
        except (np.linalg.LinAlgError, ValueError) as error:
            trial_merit = float("inf")
            rejection_reason = type(error).__name__ + ": " + str(error)
        # Reporting stays outside the numerical rejection handler: exceptions
        # raised by a user callback must propagate instead of changing a step.
        if trial is None:
            report("evaluation_end", scope="trial", trial=trial_index, step=float(step),
                   status="rejected", reason=rejection_reason,
                   operation_elapsed_seconds=perf_counter() - evaluation_started)
        else:
            report("evaluation_end", scope="trial", trial=trial_index, step=float(step),
                   status="evaluated", relative_residual=float(trial.relative_residual),
                   pole_margin=float(trial.pole_margin),
                   operation_elapsed_seconds=perf_counter() - evaluation_started)
        valid_trial = (trial is not None and np.isfinite(trial.residual_norm)
                       and np.isfinite(trial.relative_residual))
        if minimum_step > 0.0:
            decreased = valid_trial and trial.relative_residual < evaluation.relative_residual
            accept_trial = valid_trial and (decreased or step <= minimum_step)
            acceptance_reason = "residual_decreased" if decreased else "minimum_step_reached"
        else:
            accept_trial = (valid_trial and np.isfinite(trial_merit)
                            and trial_merit <= merit + 1.0e-4 * step * slope)
            acceptance_reason = "Armijo condition"
        if accept_trial:
            report("line_search_accept", trial=trial_index, step=float(step),
                   reason=acceptance_reason, previous_relative_residual=float(evaluation.relative_residual),
                   relative_residual=float(trial.relative_residual), pole_margin=float(trial.pole_margin))
            report("line_search_end", status="accepted", trials=trial_index, step=float(step),
                   operation_elapsed_seconds=perf_counter() - started)
            return trial_flat, trial, step
        fields = {} if trial is None else {
            "relative_residual": float(trial.relative_residual), "pole_margin": float(trial.pole_margin),
        }
        report("line_search_reject", trial=trial_index, step=float(step), reason=rejection_reason, **fields)
        del trial, trial_kernel, trial_flat
        if minimum_step > 0.0 and step <= minimum_step:
            report("line_search_end", status="invalid_trial_at_minimum_step", trials=trial_index,
                   step=float(step), operation_elapsed_seconds=perf_counter() - started)
            return None
        step = max(minimum_step, step * 0.5)
    report("line_search_end", status="exhausted", trials=maximum_backtracks,
           operation_elapsed_seconds=perf_counter() - started)
    return None


def _lgmres_direction(operator, rhs, *, rtol, restart, maximum_cycles, outer_k,
                      recycled, report):
    """Use SciPy Arnoldi cycles with a bounded basis and a true residual check.

    Keep only a few normalized correction vectors between Newton solves.
    Their products with the old Jacobian must never be reused. Calling the
    public SciPy solver for one outer cycle at a time also releases its basis
    before a replacement basis is allocated. SciPy reports maxiter=1 after a
    completed cycle even when its last update solves the system, so certify
    convergence using the actual residual here.
    """
    recycled[:] = [(vector, None) for vector, _ in recycled[-outer_k:]] if outer_k else []
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros_like(rhs), 0, 0, 0.0
    inner_calls = matvec_calls = 0
    in_arnoldi = False
    cycle = 0

    class ProgressError(Exception):
        pass

    def matvec(direction):
        nonlocal inner_calls, matvec_calls
        result = operator.matvec(direction)
        matvec_calls += 1
        if in_arnoldi:
            inner_calls += 1
            try:
                report("krylov_matvec", cycle=cycle, inner_iteration=inner_calls,
                       matvec_calls=matvec_calls)
            except Exception as error:
                # SciPy catches LinAlgError raised within Arnoldi. Progress
                # consumer failures must propagate, including that type.
                raise ProgressError(error) from error
        return result

    def cycle_started(_x):
        nonlocal in_arnoldi
        in_arnoldi = True

    counted = LinearOperator(operator.shape, dtype=operator.dtype, matvec=matvec)
    correction = None
    relative_residual = float("inf")
    for cycle in range(1, maximum_cycles + 1):
        in_arnoldi = False
        try:
            correction, scipy_info = lgmres(
                counted, rhs, x0=correction, rtol=rtol, atol=0.0, maxiter=1,
                inner_m=restart, outer_k=outer_k, outer_v=recycled,
                store_outer_Av=False, callback=cycle_started,
            )
        except ProgressError as error:
            raise error.args[0] from None
        in_arnoldi = False
        # The operator owns this output; subtract in place to avoid another
        # all-Q vector solely for progress and stopping diagnostics.
        residual = counted.matvec(correction)
        residual -= rhs
        relative_residual = float(np.linalg.norm(residual) / rhs_norm)
        del residual
        report("krylov_cycle", cycle=cycle, inner_iterations=inner_calls,
               matvec_calls=matvec_calls, linear_relative_residual=relative_residual,
               residual_kind="Newton linear system", recycled_vectors=len(recycled))
        if np.isfinite(relative_residual) and relative_residual <= rtol:
            return correction, 0, inner_calls, relative_residual
        if not np.isfinite(relative_residual) or scipy_info < 0:
            return correction, cycle, inner_calls, relative_residual
    return correction, maximum_cycles, inner_calls, relative_residual


def solve_gamma_ph(
    problem: StaticParquetProblem, *, interaction_scale: float = 1.0,
    initial_kernel: Sequence[np.ndarray] | None = None, screening: str = "full-8band",
    tolerance: float = 1.0e-9, maximum_iterations: int = 100,
    krylov_restart: int = 8, krylov_maximum_iterations: int = 8,
    krylov_method: str = "gmres", krylov_outer_k: int = 3,
    maximum_memory_mb: float = 512.0, pole_ceiling: float = 0.8,
    maximum_backtracks: int = 20,
    minimum_line_search_step: float = 0.0,
    memory_mode: str = "auto",
    preserve_physical_symmetry: bool = False,
    stagnation_window: int = 5,
    stagnation_relative_improvement: float = 1.0e-3,
    stagnation_minimum_iterations: int = 5,
    progress: Callable[[GammaPhIteration], None] | None = None,
    work_progress: Callable[[dict], None] | None = None,
) -> GammaPhResult:
    """Damped analytic Newton--Krylov solve with residual-merit backtracking.

    The fallback directions are a one-vector inverse-secant update and the
    exact merit gradient. By default every accepted step decreases the
    *unmixed* residual merit. With ``minimum_line_search_step > 0``, accept
    any residual decrease, or a finite valid trial at that minimum step even
    if the residual increases. No smaller steps are tried. Set
    ``stagnation_window=0`` to continue through nonmonotone residual histories.
    Singular trial solves are rejected; no denominator is shifted.
    Failure returns the latest accepted K with an explicit nonconvergence status.
    A residual plateau returns ``stagnated`` so continuation can retry sooner.
    Tiny backtracked Newton steps with negligible progress try the fallback
    immediately; an unhelpful fallback stops the stage without another GMRES.
    A tiny backtracked Newton step with negligible progress tries the existing
    fallback directions before another Krylov solve; if none improves enough,
    the solver returns ``stagnated`` without promoting the poor trial.
    ``stagnation_window=0`` disables this check. Optional physical H/P symmetry
    preservation projects K and corrections, never the evaluated residual.
    ``work_progress`` receives scalar phase events and inner GMRES residuals;
    these inner residuals are not the nonlinear Gamma-ph residual.
    ``krylov_method='lgmres'`` recycles ``krylov_outer_k`` directions between
    restart cycles and Newton steps, recomputing their Jacobian products.
    It reports Jacobian actions during a cycle and true linear residuals at
    cycle boundaries. ``krylov_restart`` is SciPy's ``inner_m``; each cycle
    can additionally use the retained augmentation directions.
    """
    started = perf_counter()
    if work_progress is not None and not callable(work_progress):
        raise TypeError("work_progress must be callable")
    scale = _positive(interaction_scale, "interaction_scale", zero=True)
    threshold = _positive(tolerance, "tolerance")
    budget = _positive(maximum_memory_mb, "maximum_memory_mb")
    minimum_line_search_step = _positive(minimum_line_search_step, "minimum_line_search_step", zero=True)
    if minimum_line_search_step > 1.0:
        raise ValueError("minimum_line_search_step must lie in [0, 1]")
    for name, value in (
        ("maximum_iterations", maximum_iterations), ("krylov_restart", krylov_restart),
        ("krylov_maximum_iterations", krylov_maximum_iterations),
        ("maximum_backtracks", maximum_backtracks),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(preserve_physical_symmetry, (bool, np.bool_)):
        raise ValueError("preserve_physical_symmetry must be boolean")
    for name, value in (("stagnation_window", stagnation_window),
                        ("stagnation_minimum_iterations", stagnation_minimum_iterations)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    plateau_threshold = _positive(stagnation_relative_improvement, "stagnation_relative_improvement", zero=True)
    if plateau_threshold >= 1.0:
        raise ValueError("stagnation_relative_improvement must be less than one")
    memory = estimate_gamma_ph_memory(problem, krylov_restart=krylov_restart, memory_mode=memory_mode,
                                      preserve_physical_symmetry=preserve_physical_symmetry,
                                      krylov_method=krylov_method, krylov_outer_k=krylov_outer_k)
    selected = memory["memory_mode"]
    def emit(event: str, **fields) -> None:
        if work_progress is not None:
            work_progress({
                "event": event, "elapsed_seconds": perf_counter() - started,
                "interaction_scale": float(scale), "memory_mode": selected,
                "krylov_method": krylov_method, **fields,
            })
    estimate = int(memory["estimated_peak_bytes"])
    if estimate > budget * 1.0e6:
        raise MemoryError(
            f"Gamma^ph Newton--Krylov needs an estimated {estimate / 1.0e6:.1f} MB "
            f"including Krylov vectors; limit is {budget:g} MB. Reduce the mesh/restart."
        )
    projector = _PhysicalSymmetryProjector(problem) if preserve_physical_symmetry else None
    initialization_started = perf_counter()
    emit("initialization_start", iteration=0, seed_kind="RPA" if initial_kernel is None else "supplied")
    if initial_kernel is None:
        kernel, initial = build_rpa_initial(
            problem, interaction_scale=scale, screening=screening, pole_ceiling=pole_ceiling,
        )
    else:
        # Kernels are never mutated by the solver. Borrowing a shared supplied
        # buffer avoids a duplicate continuation seed; every trial owns a new K.
        supplied = problem._validate_blocks(initial_kernel, "initial_kernel")
        largest, gap = _pole_data(problem, supplied)
        kernel = supplied
        initial = {
            "kind": "supplied kernel", "source_interaction": "antisymmetrized bare gU",
            "raw_lambda_max": largest, "scaled_lambda_max": largest,
            "raw_pole_margin": gap, "initial_pole_margin": gap,
            "screened_seed_scale": 1.0, "seed_scale": 1.0,
            "pole_guard_applied": False,
        }
    initial["estimated_newton_peak_mb"] = estimate / 1.0e6
    initial["memory_diagnostics"] = memory
    initial["memory_mode"] = selected
    initial["krylov_method"] = krylov_method
    initial["krylov_outer_k"] = int(krylov_outer_k)
    kernel_flat = problem.pack(kernel) if projector is not None and initial_kernel is not None else _shared_flat(problem, kernel)
    kernel = _views(problem, kernel_flat)
    if projector is not None:
        initial_hermiticity, initial_interchange = projector.defects(kernel)
        projector.project_inplace(kernel)
        initial.update(initial_hermiticity_defect=initial_hermiticity,
                       initial_pair_interchange_defect=initial_interchange)
    initial["preserve_physical_symmetry"] = bool(preserve_physical_symmetry)
    initial["stagnation_window"] = int(stagnation_window)
    initial["stagnation_relative_improvement"] = plateau_threshold
    initial["stagnation_minimum_iterations"] = int(stagnation_minimum_iterations)
    initial["minimum_line_search_step"] = float(minimum_line_search_step)
    initial_pole_fields = {
        key: float(initial[key]) for key in (
            "raw_lambda_max", "scaled_lambda_max", "raw_pole_margin",
            "initial_pole_margin", "screened_seed_scale",
        ) if key in initial
    }
    emit("initialization_end", iteration=0, seed_kind="RPA" if initial_kernel is None else "supplied",
         operation_elapsed_seconds=perf_counter() - initialization_started, **initial_pole_fields)
    evaluation_started = perf_counter()
    emit("evaluation_start", iteration=0, scope="initial")
    current = evaluate_gamma_ph(problem, kernel, interaction_scale=scale, memory_mode=selected)
    emit("evaluation_end", iteration=0, scope="initial", status="evaluated",
         relative_residual=float(current.relative_residual), pole_margin=float(current.pole_margin),
         operation_elapsed_seconds=perf_counter() - evaluation_started)
    history, status, converged = [], "maximum_iterations", False
    previous_kernel = previous_residual = None
    recycled_directions = []
    last_step, last_info, last_count, last_direction = 0.0, 0, 0, "RPA seed" if initial_kernel is None else "supplied seed"
    for iteration in range(maximum_iterations + 1):
        # Sum sector errors without retaining an all-sector difference tensor.
        if projector is None:
            hermitian_norm = np.sqrt(sum(float(np.vdot(d, d).real)
                                          for block in kernel for d in (block - block.conj().T,)))
            hermiticity = _relative(float(hermitian_norm), float(np.linalg.norm(kernel_flat)))
            interchange = None
        else:
            hermiticity, interchange = projector.defects(kernel)
        record = GammaPhIteration(
            iteration, current.relative_residual, current.residual_norm,
            current.pole_margin, last_step, last_info, last_count, last_direction, hermiticity, interchange,
            krylov_method,
        )
        history.append(record)
        if progress is not None:
            progress(record)
        if current.relative_residual <= threshold:
            symmetry_tolerance = max(1.0e-7, 100.0 * threshold)
            converged = hermiticity <= symmetry_tolerance and (interchange is None or interchange <= symmetry_tolerance)
            status = "converged" if converged else "nonhermitian_root"
            break
        if iteration == maximum_iterations:
            break
        if (stagnation_window > 0 and iteration >= max(stagnation_window, stagnation_minimum_iterations)):
            earlier = history[-1 - stagnation_window].relative_residual
            improvement = (earlier - current.relative_residual) / max(earlier, np.finfo(float).tiny)
            if improvement <= plateau_threshold:
                status = "stagnated"
                emit("stagnation", iteration=iteration, window=int(stagnation_window),
                     relative_improvement=float(improvement),
                     relative_residual=float(current.relative_residual),
                     last_krylov_info=int(last_info), last_step=float(last_step))
                break
        packed_residual = current.residual_flat
        calls = 0
        last_inner_residual = None
        def count(value):
            nonlocal calls
            nonlocal last_inner_residual
            calls += 1
            last_inner_residual = float(value)
            emit("krylov_iteration", iteration=iteration + 1, inner_iteration=calls,
                 inner_relative_residual=last_inner_residual,
                 linear_relative_residual=last_inner_residual, residual_kind="Newton linear system")
        # Use real Krylov coefficients for the real/imaginary representation.
        # This solves the same complex equations and avoids complex mixing of
        # physically conjugate momentum sectors. The analytic JVP is unchanged.
        entries = problem.total_entries
        def apply_real(direction):
            # Interleaved real/imaginary views introduce no whole-vector copy.
            blocks = current.jvp(_views(problem, direction.view(np.complex128)))
            if projector is not None:
                # Solve within the joint physical subspace, suppressing only
                # roundoff components in this linear-system work buffer. The
                # actual nonlinear residual and analytic JVP stay unmodified.
                projector.project_inplace(blocks)
            return _shared_flat(problem, blocks).view(np.float64)
        operator = LinearOperator(
            (2 * entries, 2 * entries), dtype=np.float64,
            matvec=apply_real,
        )
        forcing = min(0.25, max(1.0e-4, 0.1 * np.sqrt(current.relative_residual)))
        krylov_started = perf_counter()
        emit("krylov_start", iteration=iteration + 1, inner_tolerance=float(forcing),
             restart=int(krylov_restart), maximum_restart_cycles=int(krylov_maximum_iterations),
             outer_k=int(krylov_outer_k) if krylov_method == "lgmres" else 0,
             recycled_vectors=len(recycled_directions),
             maximum_inner_iterations=int((krylov_restart + (krylov_outer_k if krylov_method == "lgmres" else 0))
                                          * krylov_maximum_iterations))
        rhs = -packed_residual.view(np.float64)
        if projector is not None:
            projector.project_inplace(_views(problem, rhs.view(np.complex128)))
        if krylov_method == "lgmres":
            real_newton, info, calls, last_inner_residual = _lgmres_direction(
                operator, rhs, rtol=forcing, restart=krylov_restart,
                maximum_cycles=krylov_maximum_iterations, outer_k=krylov_outer_k,
                recycled=recycled_directions,
                report=lambda name, **fields: emit(name, iteration=iteration + 1, **fields),
            )
        else:
            real_newton, info = gmres(
                operator, rhs, rtol=forcing, atol=0.0,
                restart=krylov_restart, maxiter=krylov_maximum_iterations,
                callback=count, callback_type="pr_norm",
            )
        emit("krylov_end", iteration=iteration + 1, info=int(info), inner_iterations=calls,
             inner_relative_residual=last_inner_residual,
             linear_relative_residual=last_inner_residual, residual_kind="Newton linear system",
             operation_elapsed_seconds=perf_counter() - krylov_started)
        del operator, rhs
        newton = real_newton.view(np.complex128)
        if projector is not None:
            projector.project_inplace(_views(problem, newton))
        newton_label = "Newton--LGMRES" if krylov_method == "lgmres" else "Newton--Krylov"
        accepted = _line_search(current, kernel_flat, newton,
                                interaction_scale=scale, maximum_backtracks=maximum_backtracks,
                                minimum_step=minimum_line_search_step,
                                work_emit=emit, iteration=iteration + 1, direction_name=newton_label)
        stalled_newton = None
        if accepted is not None and stagnation_window > 0 and minimum_line_search_step == 0.0:
            relative_improvement = (
                current.relative_residual - accepted[1].relative_residual
            ) / max(current.relative_residual, np.finfo(float).tiny)
            if (accepted[2] <= 1.0 / 32.0
                    and relative_improvement <= plateau_threshold
                    and accepted[1].relative_residual > threshold):
                stalled_newton = dict(step=float(accepted[2]),
                                      relative_improvement=float(relative_improvement),
                                      trial_relative_residual=float(accepted[1].relative_residual))
                emit("stalled_newton", iteration=iteration + 1,
                     reason="small_backtracked_step_with_negligible_residual_decrease",
                     relative_residual=float(current.relative_residual), **stalled_newton)
                # This trial has not been promoted to an accepted iterate.
                # Release its K/R buffers before constructing a fallback; the
                # current accepted state remains unchanged.
                accepted = None
        label = newton_label
        del newton, real_newton
        if accepted is None and selected == "cached" and previous_kernel is not None:
            step_difference = kernel_flat - previous_kernel
            residual_difference = packed_residual - previous_residual
            norm_squared = float(np.vdot(residual_difference, residual_difference).real)
            if norm_squared > np.finfo(float).tiny:
                coefficient = float(np.vdot(residual_difference, packed_residual).real) / norm_squared
                step_difference -= residual_difference
                step_difference *= -coefficient
                step_difference -= packed_residual
                if projector is not None:
                    projector.project_inplace(_views(problem, step_difference))
                accepted = _line_search(current, kernel_flat, step_difference,
                                        interaction_scale=scale, maximum_backtracks=maximum_backtracks,
                                        minimum_step=minimum_line_search_step,
                                        work_emit=emit, iteration=iteration + 1, direction_name="inverse secant")
                label = "inverse secant"
            del step_difference, residual_difference
        if accepted is not None and stalled_newton is not None:
            relative_improvement = (
                current.relative_residual - accepted[1].relative_residual
            ) / max(current.relative_residual, np.finfo(float).tiny)
            if relative_improvement <= plateau_threshold and accepted[1].relative_residual > threshold:
                emit("stall_recovery_reject", iteration=iteration + 1, direction=label,
                     relative_improvement=float(relative_improvement),
                     trial_relative_residual=float(accepted[1].relative_residual))
                accepted = None
        if accepted is None:
            # Build the merit-gradient fallback only after cheaper directions
            # fail. Streamed mode keeps no K/R inverse-secant history.
            gradient_started = perf_counter()
            emit("merit_gradient_start", iteration=iteration + 1, direction="merit gradient")
            gradient_blocks = current.adjoint_jvp(current.residual)
            gradient = _shared_flat(problem, gradient_blocks)
            if projector is not None:
                projector.project_inplace(gradient_blocks)
            gradient_norm = float(np.linalg.norm(gradient))
            emit("merit_gradient_end", iteration=iteration + 1, direction="merit gradient",
                 gradient_norm=gradient_norm,
                 operation_elapsed_seconds=perf_counter() - gradient_started)
            if gradient_norm > 0.0 and np.all(np.isfinite(gradient)):
                gradient *= -min(1.0, current.residual_norm / gradient_norm)
                accepted = _line_search(current, kernel_flat, gradient,
                                        interaction_scale=scale, maximum_backtracks=maximum_backtracks,
                                        minimum_step=minimum_line_search_step,
                                        work_emit=emit, iteration=iteration + 1, direction_name="merit gradient")
                label = "merit gradient"
            del gradient, gradient_blocks
        if accepted is not None and stalled_newton is not None:
            relative_improvement = (
                current.relative_residual - accepted[1].relative_residual
            ) / max(current.relative_residual, np.finfo(float).tiny)
            if relative_improvement <= plateau_threshold and accepted[1].relative_residual > threshold:
                emit("stall_recovery_reject", iteration=iteration + 1, direction=label,
                     relative_improvement=float(relative_improvement),
                     trial_relative_residual=float(accepted[1].relative_residual))
                accepted = None
        if accepted is None:
            status = "stagnated" if stalled_newton is not None else "line_search_failed"
            if stalled_newton is not None:
                emit("stagnation", iteration=iteration, attempted_iteration=iteration + 1,
                     reason="small_newton_step_and_no_meaningful_fallback_decrease",
                     relative_residual=float(current.relative_residual),
                     last_krylov_info=int(info), **stalled_newton)
            break
        if stalled_newton is not None:
            emit("stall_recovery", iteration=iteration + 1, status="accepted", direction=label,
                 relative_improvement=float(relative_improvement),
                 relative_residual=float(accepted[1].relative_residual), step=float(accepted[2]))
        if recycled_directions and (label != newton_label or accepted[2] <= 1.0 / 32.0):
            recycled_directions.clear()
            emit("krylov_recycle_reset", iteration=iteration + 1,
                 reason="fallback direction or strongly damped Newton step")
        if selected == "cached":
            previous_kernel, previous_residual = kernel_flat, packed_residual
        kernel_flat, trial, last_step = accepted
        kernel = _views(problem, kernel_flat)
        last_direction = label
        last_info, last_count = int(info), calls
        # Release the old derivative cache before promoting the new context.
        current = None
        del packed_residual, accepted
        evaluation_started = perf_counter()
        emit("evaluation_start", iteration=iteration + 1, scope="accepted_jacobian")
        current = _enable_jacobian(trial)
        emit("evaluation_end", iteration=iteration + 1, scope="accepted_jacobian", status="prepared",
             relative_residual=float(current.relative_residual), pole_margin=float(current.pole_margin),
             operation_elapsed_seconds=perf_counter() - evaluation_started)
        del trial
    return GammaPhResult(
        problem=problem, kernel=kernel, history=tuple(history), converged=converged,
        status=status, initial_diagnostics=initial, interaction_scale=scale,
    )


__all__ = [
    "GammaPhEvaluation", "GammaPhIteration", "GammaPhResult", "build_rpa_initial",
    "evaluate_gamma_ph", "solve_gamma_ph",
    "estimate_gamma_ph_memory",
    "project_physical_symmetry",
]
