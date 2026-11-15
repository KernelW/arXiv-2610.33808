r"""Odd, single-isospin superconductivity from the full parquet Gamma.

With ph legs (k1,k2;k3,k4)=(q-Q,q;p-Q,p), the Cooper slice is
Gamma_C[k,k'] = Gamma[Q=k'-k](q=k',p=-k). It is already antisymmetrized.
The full-grid gap equation uses V_pair=Gamma_C/2, reproducing the reference
RPA notebook's odd projection of its raw direct interaction in the tree
limit. No additional spinor overlaps or quadrature factors are inserted.

The finite-temperature pp susceptibility, BdG occupations, and result types
are adapted from active_ladder/superconductivity.py without its file loaders. The same weighted odd
linearized kernel and damped, number-conserving nonlinear method are adapted
to the open hexagonal mesh here. No HF calculation or HF data loading occurs.
One-isospin, one-valley pairing has physical pair momentum 2K, where k is
measured relative to the valley; this is not an intervalley/singlet solver.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import json
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.special import expit

# These pure scalar formulas and result containers are copied from the
# reference superconductivity.py, avoiding active_ladder.__init__ and its
# unrelated HDF5/HF file-loading dependency. The formulas are unchanged.
Array = np.ndarray


def fermi_centered(xi: np.ndarray, beta: float) -> np.ndarray:
    return expit(-_positive_beta(beta) * np.asarray(xi, dtype=float))


def cooper_susceptibility(
    xi_k: Array,
    xi_minus_k: Array,
    *,
    beta: float,
    degeneracy_tolerance: float = 1.0e-12,
) -> Array:
    beta = _positive_beta(beta)
    tolerance = float(degeneracy_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be finite and nonnegative")
    left, right = np.broadcast_arrays(
        np.asarray(xi_k, dtype=float), np.asarray(xi_minus_k, dtype=float)
    )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("xi contains non-finite values")

    denominator = left + right
    # 1-f(x)-f(y) = [tanh(beta*x/2)+tanh(beta*y/2)]/2.  The tanh form avoids
    # subtracting two nearly unit Fermi occupations.  Switch to the analytic
    # divided-difference limit when the separation is small on the thermal
    # scale, not only when it is small in absolute eV.
    numerator = 0.5 * (np.tanh(0.5 * beta * left) + np.tanh(0.5 * beta * right))
    result = np.empty_like(denominator, dtype=float)
    regular = (np.abs(denominator) > tolerance) & (
        0.5 * beta * np.abs(denominator) > 1.0e-7
    )
    np.divide(numerator, denominator, out=result, where=regular)

    mismatch = 0.5 * (left - right)
    f_mismatch = fermi_centered(mismatch, beta)
    limit = beta * f_mismatch * (1.0 - f_mismatch)
    result[~regular] = limit[~regular]

    # Roundoff can produce tiny negative values when numerator is obtained by
    # subtracting nearly equal Fermi factors.  A genuinely negative result
    # would violate monotonicity of n_F and indicates invalid input/numerics.
    scale = max(float(np.max(np.abs(result), initial=0.0)), 1.0)
    if float(np.min(result, initial=0.0)) < -1.0e-12 * scale:
        raise FloatingPointError("Cooper susceptibility became negative")
    return np.maximum(result, 0.0)


def nonlinear_cooper_susceptibility(
    xi_k: Array,
    xi_minus_k: Array,
    delta: Array,
    *,
    beta: float,
    degeneracy_tolerance: float = 1.0e-12,
) -> Array:
    beta = _positive_beta(beta)
    tolerance = float(degeneracy_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be finite and nonnegative")
    left, right, gap = np.broadcast_arrays(
        np.asarray(xi_k, dtype=float),
        np.asarray(xi_minus_k, dtype=float),
        np.asarray(delta, dtype=np.complex128),
    )
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or not np.all(np.isfinite(gap))
    ):
        raise ValueError("xi or Delta contains non-finite values")

    mean_xi = 0.5 * (left + right)
    mismatch = 0.5 * (left - right)
    energy = np.sqrt(mean_xi * mean_xi + np.abs(gap) ** 2)
    result = np.empty_like(energy, dtype=float)
    regular = (energy > tolerance) & (beta * energy > 1.0e-7)
    numerator = 0.5 * (
        np.tanh(0.5 * beta * (mismatch + energy))
        - np.tanh(0.5 * beta * (mismatch - energy))
    )
    np.divide(numerator, 2.0 * energy, out=result, where=regular)
    f_mismatch = fermi_centered(mismatch, beta)
    result[~regular] = (beta * f_mismatch * (1.0 - f_mismatch))[~regular]
    return np.maximum(result, 0.0)


def bdg_electron_occupation(
    xi_k: Array,
    xi_minus_k: Array,
    delta: Array,
    *,
    beta: float,
    degeneracy_tolerance: float = 1.0e-14,
) -> Array:
    beta = _positive_beta(beta)
    degeneracy_tolerance = float(degeneracy_tolerance)
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be finite and nonnegative")
    left, right, gap = np.broadcast_arrays(
        np.asarray(xi_k, dtype=float),
        np.asarray(xi_minus_k, dtype=float),
        np.asarray(delta, dtype=np.complex128),
    )
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or not np.all(np.isfinite(gap))
    ):
        raise ValueError("xi or Delta contains non-finite values")
    mean_xi = 0.5 * (left + right)
    mismatch = 0.5 * (left - right)
    energy = np.sqrt(mean_xi * mean_xi + np.abs(gap) ** 2)
    polarization = np.zeros_like(energy)
    np.divide(mean_xi, energy, out=polarization, where=energy > degeneracy_tolerance)
    lower_weight = 0.5 * (1.0 - polarization)
    upper_weight = 0.5 * (1.0 + polarization)
    return lower_weight * fermi_centered(
        mismatch - energy, beta
    ) + upper_weight * fermi_centered(mismatch + energy, beta)


@dataclass(frozen=True)
class LinearizedGapResult:

    leading_eigenvalue: float
    gap_mode: Array
    weighted_mode: Array
    susceptibility: Array
    residual: float
    converged: bool
    solver: str
    is_unstable: bool
    is_stable: bool
    classification: str
    instability_tolerance: float

    @property
    def distance_to_instability(self) -> float:
        """Return ``1-lambda_max``; negative means linearly unstable."""
        return 1.0 - self.leading_eigenvalue


@dataclass(frozen=True)
class NonlinearGapResult:

    delta: Array
    mu_shift: float
    centered_xi: Array
    occupation: Array
    filling: float
    target_filling: float
    gap_residual: float
    filling_residual: float
    iterations: int
    converged: bool
    is_superconducting: bool
    status: str
    linearized: LinearizedGapResult

    @property
    def max_gap(self) -> float:
        return float(np.max(np.abs(self.delta), initial=0.0))


def _positive_beta(beta: float) -> float:
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    return beta


@dataclass(frozen=True)
class PairingProblem:
    coords: np.ndarray
    weight: float
    beta: float
    target_filling: float

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords)
        if (coords.ndim != 2 or coords.shape[1] != 2
                or not np.all(np.isfinite(coords))
                or not np.array_equal(coords, coords.astype(np.int64))):
            raise ValueError("coords must contain integer pairs")
        coords = coords.astype(np.int64)
        if len({tuple(c) for c in coords}) != len(coords):
            raise ValueError("duplicate pair coordinates")
        for value, name in ((self.weight, "weight"), (self.beta, "beta")):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.target_filling < 1:
            raise ValueError("active-band filling must lie in (0,1)")
        object.__setattr__(self, "coords", coords)
        inversion_indices(self)

    @property
    def npoints(self) -> int:
        return len(self.coords)


def inversion_indices(problem: PairingProblem) -> np.ndarray:
    positions = {tuple(c): i for i, c in enumerate(problem.coords)}
    minus = np.asarray([positions.get(tuple(-c), -1) for c in problem.coords])
    if np.any(minus < 0) or not np.array_equal(minus[minus], np.arange(problem.npoints)):
        raise ValueError("pair coordinates are not closed under inversion")
    return minus


def odd_project_vector(problem: PairingProblem, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128)
    if vector.shape != (problem.npoints,):
        raise ValueError("gap vector shape does not match pair coordinates")
    return 0.5 * (vector - vector[inversion_indices(problem)])


def vertex_diagnostics(matrix: np.ndarray, coords: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (len(coords), len(coords)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Cooper matrix has invalid shape or non-finite entries")
    positions = {tuple(c): i for i, c in enumerate(coords)}
    minus = np.asarray([positions[tuple(-c)] for c in coords])
    scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
    return {
        "hermiticity_relative_error": float(np.linalg.norm(matrix - matrix.conj().T) / scale),
        "outgoing_pauli_relative_error": float(np.linalg.norm(matrix + matrix[minus, :]) / scale),
        "incoming_pauli_relative_error": float(np.linalg.norm(matrix + matrix[:, minus]) / scale),
        "maximum_absolute_vertex_eV": float(np.max(np.abs(matrix), initial=0.0)),
    }


@dataclass(frozen=True)
class CooperVertex:
    raw_gamma: np.ndarray
    interaction: np.ndarray
    coords: np.ndarray
    diagnostics: dict[str, Any]


def extract_cooper_vertex(
    problem: Any,
    gamma_blocks_or_getter: Sequence[np.ndarray] | np.ndarray | Callable[[int], np.ndarray],
    *,
    symmetry_tolerance: float = 1e-5,
) -> CooperVertex:
    """Extract from full Gamma blocks; a getter receives a sector integer.

    Production archives must contain Gamma=Lambda_irr-C Phi+Phi. Generic
    arrays/getters remain available for independently supplied test interactions.
    Material Pauli/Hermiticity defects raise before any projection is done.
    """
    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance must be finite and nonnegative")
    archive = getattr(gamma_blocks_or_getter, "__self__", None)
    archive_metadata = getattr(archive, 'metadata', None)
    if archive_metadata is not None:
        _validate_production_gamma_metadata(archive_metadata)
        if (hasattr(problem, 'source_signature')
                and archive_metadata.get('source_signature') != problem.source_signature):
            raise ValueError('Full Gamma and the Cooper problem use different sources.')
    coords = np.asarray(problem.system.coords, dtype=np.int64)
    positions = {tuple(c): i for i, c in enumerate(coords)}
    n = len(coords)
    raw = np.empty((n, n), dtype=np.complex128)
    written = np.zeros((n, n), dtype=bool)
    blocks = gamma_blocks_or_getter
    if isinstance(blocks, np.ndarray) and blocks.ndim == 1:
        blocks = problem.unpack(blocks)
    if not callable(blocks) and len(blocks) != len(problem.transfers):
        raise ValueError("number of Gamma blocks does not match transfer sectors")
    for sector, (transfer, domain) in enumerate(zip(problem.transfers, problem.domain_coords)):
        matrix = np.asarray(blocks(sector) if callable(blocks) else blocks[sector])
        if matrix.shape != (len(domain), len(domain)) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"invalid full Gamma block in sector {sector}")
        domain_positions = {tuple(c): i for i, c in enumerate(domain)}
        # q=k', k=q-Q, p=-k. Every pair (k,k') appears in exactly one Q.
        for row, q in enumerate(domain):
            k = q - transfer
            column = domain_positions.get(tuple(-k))
            if column is None:
                raise ValueError("Cooper routing leaves the finite ph domain")
            ik, iq = positions[tuple(k)], positions[tuple(q)]
            if written[ik, iq]:
                raise ValueError("duplicate Cooper routing")
            raw[ik, iq] = matrix[row, column]
            written[ik, iq] = True
    if not np.all(written):
        raise ValueError("transfer sectors do not cover the complete Cooper matrix")
    diagnostics: dict[str, Any] = vertex_diagnostics(raw, coords)
    errors = {name: value for name, value in diagnostics.items() if name.endswith("relative_error")}
    if max(errors.values()) > symmetry_tolerance:
        raise ValueError(f"full Gamma fails Cooper symmetry tolerance {symmetry_tolerance}: {errors}")
    diagnostics.update(
        symmetry_tolerance=float(symmetry_tolerance),
        antisymmetrized_vertex_factor=0.5,
        channel="same-isospin intravalley odd; physical total momentum 2K",
        routing="Gamma_C[k,kprime]=Gamma_Q(q=kprime,p=-k), Q=kprime-k",
        overlaps_applied_again=False,
    )
    if hasattr(problem, "source_signature"):
        diagnostics["source_signature"] = problem.source_signature
    if archive_metadata is not None:
        diagnostics["source_gamma_metadata"] = dict(archive_metadata)
        diagnostics["vertex_construction"] = archive_metadata['construction']
    return CooperVertex(raw, 0.5 * raw, coords.copy(), diagnostics)


def _validate_production_gamma_metadata(metadata):
    from .parquet import FULL_GAMMA_CONSTRUCTION, FULL_GAMMA_SCHEMA
    if (metadata.get('schema') != FULL_GAMMA_SCHEMA
            or metadata.get('construction') != FULL_GAMMA_CONSTRUCTION):
        raise ValueError('Legacy full Gamma is not accepted for superconductivity. '
                         'Reconstruct Gamma=Lambda_irr-C Phi+Phi from its converged K checkpoint.')
    if (not metadata.get('converged') or not metadata.get('production_eligible')
            or not metadata.get('target_reached')):
        raise ValueError('Superconductivity requires a converged production Gamma at target coupling.')


def _matrix(problem: PairingProblem, gamma: np.ndarray, tolerance: float) -> np.ndarray:
    matrix = np.asarray(gamma, dtype=np.complex128)
    if matrix.shape != (problem.npoints, problem.npoints) or not np.all(np.isfinite(matrix)):
        raise ValueError("pair interaction must be a finite N by N matrix")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("symmetry_tolerance must be finite and nonnegative")
    scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
    defect = float(np.linalg.norm(matrix - matrix.conj().T) / scale)
    if defect > tolerance:
        raise ValueError(f"Cooper Hermiticity defect {defect:.3e} exceeds {tolerance:.3e}")
    if defect > 1e-10:
        warnings.warn(
            f"Cooper Hermiticity defect {defect:.3e} is within the requested "
            "tolerance; the gap solver uses its explicitly Hermitian part.",
            RuntimeWarning, stacklevel=3,
        )
    return matrix


class _OddVertex:
    def __init__(self, problem: PairingProblem, matrix: np.ndarray, tolerance: float):
        matrix = _matrix(problem, matrix, tolerance)
        minus = inversion_indices(problem)
        self.n = problem.npoints
        self.positive = np.flatnonzero(np.arange(self.n) < minus)
        self.negative = minus[self.positive]
        if not len(self.positive):
            raise ValueError("pairing mesh has no odd subspace")
        p, m = self.positive, self.negative
        reduced = 0.5 * (matrix[np.ix_(p, p)] - matrix[np.ix_(p, m)]
                         - matrix[np.ix_(m, p)] + matrix[np.ix_(m, m)])
        self.reduced = 0.5 * (reduced + reduced.conj().T)

    def expand(self, reduced: np.ndarray) -> np.ndarray:
        full = np.zeros(self.n, dtype=np.complex128)
        full[self.positive] = reduced / np.sqrt(2.)
        full[self.negative] = -reduced / np.sqrt(2.)
        return full

    def reduce(self, full: np.ndarray) -> np.ndarray:
        return (full[self.positive] - full[self.negative]) / np.sqrt(2.)

    def action(self, full: np.ndarray) -> np.ndarray:
        return self.expand(self.reduced @ self.reduce(full))


def _phase(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128).copy()
    value = vector[np.argmax(np.abs(vector))]
    if abs(value):
        vector *= np.conj(value) / abs(value)
    return vector


def linearized_gap_diagnostic(
    problem: PairingProblem, interaction: np.ndarray, xi: np.ndarray, *,
    beta: float | None = None, symmetry_tolerance: float = 1e-5,
    eigensolver_tolerance: float = 1e-10, instability_tolerance: float = 1e-8,
    dense_threshold: int = 192, maxiter: int | None = None,
    _vertex: _OddVertex | None = None,
) -> LinearizedGapResult:
    """Largest eigenvalue of -w sqrt(chi_pp) P_- V P_- sqrt(chi_pp)."""
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (problem.npoints,) or not np.all(np.isfinite(xi)):
        raise ValueError("xi must be finite with one value per momentum")
    beta = problem.beta if beta is None else float(beta)
    for value in (eigensolver_tolerance, instability_tolerance):
        if not np.isfinite(value) or value < 0:
            raise ValueError("eigenvalue tolerances must be finite and nonnegative")
    vertex = _vertex or _OddVertex(problem, interaction, symmetry_tolerance)
    minus = inversion_indices(problem)
    chi = cooper_susceptibility(xi, xi[minus], beta=beta)
    sqrt_chi = np.sqrt(0.5 * (chi[vertex.positive] + chi[vertex.negative]))
    kernel = -problem.weight * sqrt_chi[:, None] * vertex.reduced * sqrt_chi[None, :]
    kernel = 0.5 * (kernel + kernel.conj().T)
    method = "dense-eigh"
    if len(kernel) <= max(2, int(dense_threshold)):
        values, vectors = np.linalg.eigh(kernel)
        leading, weighted = float(values[-1]), vectors[:, -1]
    else:
        rng = np.random.default_rng(0x5C0A)
        seed = rng.standard_normal(len(kernel)) + 1j * rng.standard_normal(len(kernel))
        try:
            values, vectors = eigsh(kernel, k=1, which="LA", v0=seed,
                                    tol=eigensolver_tolerance, maxiter=maxiter)
            leading, weighted = float(values[0].real), vectors[:, 0]
            method = "eigsh"
        except ArpackNoConvergence:
            values, vectors = np.linalg.eigh(kernel)
            leading, weighted = float(values[-1]), vectors[:, -1]
            method = "dense-eigh-fallback"
    weighted = _phase(weighted)
    weighted /= np.linalg.norm(weighted)
    absolute_residual = float(np.linalg.norm(kernel @ weighted - leading * weighted))
    acceptance = max(100 * np.finfo(float).eps,
                     min(1e-7, max(eigensolver_tolerance, 0.1 * instability_tolerance)))
    converged = absolute_residual <= acceptance
    uncertainty = max(absolute_residual, 100 * np.finfo(float).eps)
    unstable = bool(converged and leading - uncertainty > 1 + instability_tolerance)
    stable = bool(converged and leading + uncertainty <= 1 + instability_tolerance)
    gap = vertex.expand(-problem.weight * vertex.reduced @ (sqrt_chi * weighted))
    if not np.max(np.abs(gap), initial=0.):
        gap = vertex.expand(weighted)
    gap = _phase(gap / np.max(np.abs(gap)))
    return LinearizedGapResult(
        leading, gap, vertex.expand(weighted), chi,
        absolute_residual / max(abs(leading), 1.), bool(converged), method,
        unstable, stable, "unstable" if unstable else "stable" if stable else "indeterminate",
        float(instability_tolerance),
    )


def _number_solution(xi: np.ndarray, delta: np.ndarray, minus: np.ndarray,
                     target: float, beta: float) -> tuple[float, np.ndarray, np.ndarray]:
    padding = max(float(np.ptp(xi)), 4 * float(np.max(np.abs(delta))), 40 / beta, 1e-6)
    def residual(mu: float) -> float:
        shifted = xi - mu
        return float(np.mean(bdg_electron_occupation(
            shifted, shifted[minus], delta, beta=beta)) - target)
    mu = brentq(residual, float(xi.min() - padding), float(xi.max() + padding),
                xtol=1e-13, rtol=4 * np.finfo(float).eps)
    shifted = xi - mu
    occupation = bdg_electron_occupation(shifted, shifted[minus], delta, beta=beta)
    return float(mu), shifted, occupation


def solve_nonlinear_gap(
    problem: PairingProblem, interaction: np.ndarray, xi: np.ndarray, *,
    initial_gap: float = 1e-3, initial_delta: np.ndarray | None = None,
    mixing: float = 0.25, minimum_mixing: float = 0.02,
    max_iterations: int = 2000, gap_tolerance: float = 1e-8,
    filling_tolerance: float = 1e-11, minimum_nonzero_gap: float = 1e-10,
    symmetry_tolerance: float = 1e-5, enforce_linearized_gate: bool = True,
    linearized_kwargs: dict[str, Any] | None = None, progress: bool = False,
) -> NonlinearGapResult:
    """Damped odd gap map with the active-band number equation at each step."""
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (problem.npoints,) or not np.all(np.isfinite(xi)):
        raise ValueError("xi must be finite with one value per momentum")
    if not 0 < minimum_mixing <= mixing <= 1:
        raise ValueError("require 0 < minimum_mixing <= mixing <= 1")
    for value in (initial_gap, gap_tolerance, filling_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("initial gap and convergence tolerances must be positive")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(minimum_nonzero_gap) or minimum_nonzero_gap < 0:
        raise ValueError("minimum_nonzero_gap must be nonnegative")
    minus = inversion_indices(problem)
    zero = np.zeros(problem.npoints, dtype=np.complex128)
    normal_mu, normal_xi, normal_occ = _number_solution(
        xi, zero, minus, problem.target_filling, problem.beta)
    if abs(normal_mu) > 1e-6:
        raise ValueError("xi is not centered at the requested normal-state active filling")
    vertex = _OddVertex(problem, interaction, symmetry_tolerance)
    linear_options = dict(linearized_kwargs or {})
    if {"symmetry_tolerance", "_vertex", "beta"} & linear_options.keys():
        raise ValueError("linearized_kwargs may not override internal problem settings")
    linear = linearized_gap_diagnostic(problem, interaction, normal_xi,
                                      symmetry_tolerance=symmetry_tolerance,
                                      _vertex=vertex, **linear_options)
    if enforce_linearized_gate and not linear.is_unstable:
        stable = linear.is_stable
        return NonlinearGapResult(
            zero, normal_mu, normal_xi, normal_occ, float(np.mean(normal_occ)),
            problem.target_filling, 0. if stable else float("inf"),
            float(np.mean(normal_occ) - problem.target_filling), 0, stable, False,
            "normal-state-stable" if stable else "linearized-instability-indeterminate", linear)
    delta = (initial_gap * linear.gap_mode if initial_delta is None
             else odd_project_vector(problem, initial_delta))
    if not np.all(np.isfinite(delta)) or np.max(np.abs(delta)) == 0:
        raise ValueError("initial gap must have a finite nonzero odd component")
    current_mixing, previous = mixing, float("inf")
    converged = False
    status = "maximum-iterations-reached"
    for iteration in range(1, int(max_iterations) + 1):
        mu, shifted, occupation = _number_solution(
            xi, delta, minus, problem.target_filling, problem.beta)
        chi = nonlinear_cooper_susceptibility(shifted, shifted[minus], delta, beta=problem.beta)
        mapped = -problem.weight * vertex.action(chi * delta)
        residual = float(np.linalg.norm(mapped - delta) /
                         max(float(np.linalg.norm(delta)), initial_gap, 1e-15))
        if not np.isfinite(residual):
            status = "non-finite-iteration"
            break
        if residual < gap_tolerance:
            delta, converged = mapped, True
            break
        if residual > 1.2 * previous:
            current_mixing = max(minimum_mixing, 0.5 * current_mixing)
        elif residual < 0.7 * previous:
            current_mixing = min(mixing, 1.1 * current_mixing)
        delta = odd_project_vector(problem, delta + current_mixing * (mapped - delta))
        previous = residual
        if progress and (iteration == 1 or iteration % 25 == 0):
            print(f"gap {iteration}: residual={residual:.3e}, "
                  f"max|Delta|={np.max(np.abs(delta)):.6e} eV, mu_shift={mu:+.6e} eV")
    mu, shifted, occupation = _number_solution(
        xi, delta, minus, problem.target_filling, problem.beta)
    chi = nonlinear_cooper_susceptibility(shifted, shifted[minus], delta, beta=problem.beta)
    mapped = -problem.weight * vertex.action(chi * delta)
    residual = float(np.linalg.norm(mapped - delta) /
                     max(float(np.linalg.norm(delta)), initial_gap, 1e-15))
    filling = float(np.mean(occupation))
    filling_residual = filling - problem.target_filling
    converged = bool(converged and residual <= gap_tolerance
                     and abs(filling_residual) <= filling_tolerance)
    max_gap = float(np.max(np.abs(delta)))
    if converged and max_gap <= minimum_nonzero_gap:
        if linear.is_stable:
            delta, mu, shifted, occupation = zero, normal_mu, normal_xi, normal_occ
            filling = float(np.mean(normal_occ))
            filling_residual, residual = filling - problem.target_filling, 0.
            status = "converged-normal-state"
        else:
            converged, status = False, "collapsed-to-unverified-normal-state"
    superconducting = bool(converged and max_gap > minimum_nonzero_gap)
    if superconducting:
        status = "converged-superconducting"
    elif not converged and status == "maximum-iterations-reached" and iteration < max_iterations:
        status = "post-check-failed"
    return NonlinearGapResult(delta, mu, shifted, occupation, filling,
                              problem.target_filling, residual, filling_residual,
                              iteration, converged, superconducting, status, linear)


@dataclass(frozen=True)
class SuperconductivityResult:
    problem: PairingProblem
    xi: np.ndarray
    chemical_potential: float
    linearized: LinearizedGapResult
    nonlinear: NonlinearGapResult | None
    diagnostics: dict[str, Any]

    @property
    def target_filling(self) -> float:
        return self.problem.target_filling


def solve_superconductivity(
    system: Any, interaction: np.ndarray | CooperVertex, *,
    run_nonlinear: bool = True, symmetry_tolerance: float = 1e-5,
    linearized_kwargs: dict[str, Any] | None = None,
    nonlinear_kwargs: dict[str, Any] | None = None, progress: bool = False,
) -> SuperconductivityResult:
    """Use free active-band xi and its occupation at the original free mu.

    The density constraint holds the active-band population fixed, matching
    the reference superconductivity notebook. Other bands are frozen; this
    does not re-solve a multiband superconducting number equation.
    """
    xi = system.energies[:, system.active_band_index] - system.chemical_potential
    target = float(np.mean(expit(-system.beta * xi)))
    problem = PairingProblem(system.coords, system.weight, system.beta, target)
    if isinstance(interaction, CooperVertex):
        if not np.array_equal(interaction.coords, system.coords):
            raise ValueError("Cooper vertex and free system use different coordinates")
        matrix = interaction.interaction
        diagnostics = dict(interaction.diagnostics)
        if 'source_gamma_metadata' in diagnostics:
            _validate_production_gamma_metadata(diagnostics['source_gamma_metadata'])
            diagnostics['vertex_construction'] = diagnostics['source_gamma_metadata']['construction']
    else:
        matrix = np.asarray(interaction, dtype=np.complex128)
        diagnostics = vertex_diagnostics(matrix, system.coords)
    diagnostics.update(normal_state="free bands; no HF", quadrature_weight=float(system.weight),
                       active_band_index=int(system.active_band_index),
                       number_constraint="active-band mean occupation fixed",
                       symmetry_tolerance=float(symmetry_tolerance),
                       hermitian_part_used=True,
                       beta_per_eV=float(system.beta), temperature_eV=float(1 / system.beta))
    if is_dataclass(system.config):
        diagnostics["model_config"] = asdict(system.config)
    if run_nonlinear:
        options = dict(nonlinear_kwargs or {})
        options.setdefault("progress", progress)
        nonlinear = solve_nonlinear_gap(problem, matrix, xi,
                                        symmetry_tolerance=symmetry_tolerance,
                                        linearized_kwargs=linearized_kwargs, **options)
        linearized = nonlinear.linearized
    else:
        nonlinear = None
        linearized = linearized_gap_diagnostic(problem, matrix, xi,
                                               symmetry_tolerance=symmetry_tolerance,
                                               **dict(linearized_kwargs or {}))
    return SuperconductivityResult(problem, xi, float(system.chemical_potential),
                                    linearized, nonlinear, diagnostics)


def save_superconductivity(path: str | Path, result: SuperconductivityResult) -> Path:
    if 'source_gamma_metadata' in result.diagnostics:
        _validate_production_gamma_metadata(result.diagnostics['source_gamma_metadata'])
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    linear, nonlinear = result.linearized, result.nonlinear
    metadata = {
        "schema": "crpa_parquet_superconductivity_v1", **result.diagnostics,
        "normal_mu_eV": result.chemical_potential, "active_filling": result.target_filling,
        "temperature_eV": 1 / result.problem.beta, "beta_per_eV": result.problem.beta,
        "lambda_max": linear.leading_eigenvalue, "classification": linear.classification,
        "eigen_residual": linear.residual, "linearized_converged": linear.converged,
        "nonlinear_requested": nonlinear is not None,
    }
    arrays = dict(pair_coords=result.problem.coords, xi=result.xi,
                  leading_gap_mode=linear.gap_mode, susceptibility=linear.susceptibility,
                  leading_eigenvalue=np.asarray(linear.leading_eigenvalue))
    if nonlinear is not None:
        metadata.update(status=nonlinear.status, converged=nonlinear.converged,
                        is_superconducting=nonlinear.is_superconducting,
                        gap_residual=nonlinear.gap_residual, filling_residual=nonlinear.filling_residual,
                        iterations=nonlinear.iterations)
        arrays.update(delta=nonlinear.delta, centered_xi=nonlinear.centered_xi,
                      occupation=nonlinear.occupation, mu_shift=np.asarray(nonlinear.mu_shift))
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
    return path
