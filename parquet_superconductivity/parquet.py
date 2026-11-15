"""The reference all-Q Newton solver with a constrained-RPA *source*.

The imported solver's historical ``bare`` field stores Lambda_c here.  Its
nonlinear equation is unchanged: R(K)=K-g Lambda_c+C Phi(K)=0, with
(I-K B) Phi=K B K and B<=0.  No Hartree--Fock self-consistency is performed.
The initial seed is a pole-guarded g Lambda_c. By default subsequent stages
use only converged K; the opt-in relaxed policy also seeds from explicitly
unconverged diagnostic K. cRPA is fixed at the physical coupling; g scales that effective
source as a numerical continuation parameter and does not recompute cRPA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
import uuid
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve

from static_parquet_memory.solver import build_problem, estimate_problem_memory
from static_parquet_memory.gamma_ph_only import (
    GammaPhIteration, GammaPhResult, _PhysicalSymmetryProjector, _guard_seed,
    evaluate_gamma_ph, estimate_gamma_ph_memory, solve_gamma_ph,
)

SOURCE_SCHEMA = 'abca_crpa_parquet_source_v1'
CHECKPOINT_SCHEMA = 'abca_crpa_parquet_checkpoint_v1'
FULL_GAMMA_SCHEMA = 'abca_crpa_full_gamma_v2'
FULL_GAMMA_CONSTRUCTION = 'explicit_lambda_minus_crossed_phi_plus_phi_v1'
FULL_GAMMA_FORMULA = 'Gamma = g Lambda_c - C[Phi] + Phi; Phi = (I-K B)^(-1) K B K'


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True,
                                        allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def build_screened_problem(system, crpa, *, maximum_memory_mb=12000.0):
    """Build exact finite-mesh crossings and replace both source legs by W_c.

    The replacement writes into the existing shared source buffer. It does not
    alter the free system, its propagators, or the reference solver module.
    """
    if asdict(system.config) != asdict(crpa.system.config):
        raise ValueError('cRPA and parquet physical configurations differ')
    if not np.array_equal(system.coords, crpa.system.coords):
        raise ValueError('cRPA and parquet meshes differ')
    problem = build_problem(system, maximum_memory_mb=maximum_memory_mb,
                            memory_profile='gamma-ph')
    active = system.active_band_index
    for matrix, transfer, coords, rows, shifted in zip(
            problem.bare, problem.transfers, problem.domain_coords,
            problem.right_rows, problem.left_rows):
        left = system.spinors[shifted, :, active]
        right = system.spinors[rows, :, active]
        density = np.sum(left.conj() * right, axis=1)
        delta = coords[None, :, :] - coords[:, None, :]
        matrix[:] = (complex(crpa.interaction(-transfer)) * np.outer(density, density.conj())
                     - crpa.interaction(delta) * (left.conj() @ left.T)
                     * (right @ right.conj().T))
    norm2 = defect2 = 0.0
    for start in range(0, problem.total_entries, 65536):
        stop = min(start + 65536, problem.total_entries)
        current = problem.bare_flat[start:stop]
        delta = current + problem.bare_flat[problem.crossing_permutation[start:stop]]
        norm2 += float(np.vdot(current, current).real)
        defect2 += float(np.vdot(delta, delta).real)
    defect = np.sqrt(defect2 / norm2) if norm2 else np.sqrt(defect2)
    if defect > 1e-10:
        raise ValueError(f'cRPA source violates crossing: {defect:.3e}')
    hermiticity, interchange = _PhysicalSymmetryProjector(problem).defects(problem.bare)
    problem.source_metadata = {
        'schema': SOURCE_SCHEMA,
        'source': 'antisymmetrized constrained-RPA direct plus exchange',
        'equation': 'K - g Lambda_c + C[(I-K B)^(-1) K B K] = 0',
        'hf_used': False,
        'config': asdict(system.config),
        'crpa_metadata': crpa.metadata,
        'source_crossing_defect': float(defect),
        'source_hermiticity_defect': float(hermiticity),
        'source_pair_interchange_defect': float(interchange),
        'screened_interaction_sha256': _digest_arrays(crpa.transfer_coords, crpa.screened_interaction),
        'basis_and_propagators_sha256': _digest_arrays(
            system.coords, system.basis, system.energies, system.spinors,
            np.asarray([system.chemical_potential, system.weight]), *problem.bubble),
    }
    signature_fields = {key: problem.source_metadata[key] for key in (
        'schema', 'config', 'screened_interaction_sha256', 'basis_and_propagators_sha256')}
    problem.source_signature = hashlib.sha256(json.dumps(
        _json_value(signature_fields), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    problem.memory_diagnostics['source'] = SOURCE_SCHEMA
    problem.memory_diagnostics['source_crossing_defect'] = float(defect)
    problem.memory_diagnostics['bare_crossing_defect'] = float(defect)
    return problem


def _validate_result(problem, kernel, scale, tolerance):
    evaluation = evaluate_gamma_ph(problem, kernel, interaction_scale=scale,
                                  want_jacobian=False, memory_mode='streamed')
    h, p = _PhysicalSymmetryProjector(problem).defects(kernel)
    if evaluation.relative_residual > tolerance or max(h, p) > 1e-8:
        raise ValueError('checkpoint is not a converged solution of the current cRPA equation: '
                         f'residual={evaluation.relative_residual:.3e}, H={h:.3e}, P={p:.3e}')
    return evaluation, h, p


def save_checkpoint(result, directory, *, target_scale, continuation=None, tolerance=1e-7):
    """Atomically commit a converged source-identified K; keep one checkpoint."""
    if not result.converged:
        raise ValueError('only converged parquet stages may be checkpointed')
    problem = result.problem
    evaluation, h, p = _validate_result(problem, result.kernel,
                                       result.interaction_scale, tolerance)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'checkpoint.json'
    previous_file = None
    if manifest.is_file():
        previous_file = json.loads(manifest.read_text()).get('kernel_file')
    filename = 'gamma_ph_' + uuid.uuid4().hex + '.npy'
    output = directory / filename
    mapped = np.lib.format.open_memmap(output, mode='w+', dtype=np.complex128,
                                      shape=(problem.total_entries,))
    for i, block in enumerate(result.kernel):
        mapped[problem.offsets[i]:problem.offsets[i + 1]] = block.reshape(-1)
    mapped.flush()
    del mapped
    metadata = {
        'schema': CHECKPOINT_SCHEMA, 'source_signature': problem.source_signature,
        'source_metadata': problem.source_metadata, 'kernel_file': filename,
        'interaction_scale': float(result.interaction_scale),
        'target_scale': float(target_scale),
        'target_reached': bool(result.interaction_scale == target_scale),
        'converged': True, 'status': result.status,
        'relative_residual': float(evaluation.relative_residual),
        'hermiticity_defect': h, 'pair_interchange_defect': p,
        'history': [asdict(item) for item in result.history],
        'initial_diagnostics': result.initial_diagnostics,
        'continuation': continuation,
    }
    try:
        _atomic_json(manifest, metadata)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    if previous_file and previous_file != filename:
        # Only our generated files are eligible for cleanup.
        if Path(previous_file).name == previous_file and previous_file.startswith('gamma_ph_'):
            try:
                (directory / previous_file).unlink(missing_ok=True)
            except PermissionError:
                # A caller may still hold this older mmap on Windows. The
                # newly committed checkpoint is valid; defer stale cleanup.
                pass
    return manifest


def load_checkpoint(problem, directory, *, tolerance=1e-7, target_scale=1.0):
    """Validate provenance, actual equation, and H/P symmetry before reuse."""
    path = Path(directory).expanduser().resolve() / 'checkpoint.json'
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    if metadata.get('schema') != CHECKPOINT_SCHEMA or not metadata.get('converged'):
        raise ValueError('checkpoint is not a converged constrained-RPA parquet checkpoint')
    if metadata.get('source_signature') != problem.source_signature:
        raise ValueError('checkpoint source/model signature differs; use a separate output directory')
    scale = float(metadata['interaction_scale'])
    if not 0 < scale <= target_scale:
        raise ValueError('checkpoint coupling exceeds the requested target')
    name = metadata['kernel_file']
    if Path(name).name != name:
        raise ValueError('checkpoint kernel_file must be a local filename')
    # The residual check visits every entry. An ordinary owned array also
    # avoids pinning an obsolete checkpoint file through caller-held views.
    flat = np.load(path.parent / name, allow_pickle=False)
    if flat.dtype != np.complex128 or flat.shape != (problem.total_entries,):
        raise ValueError('checkpoint kernel dimensions or dtype differ')
    kernel = problem.unpack(flat)
    evaluation, h, p = _validate_result(problem, kernel, scale, tolerance)
    record = GammaPhIteration(iteration=0, relative_residual=evaluation.relative_residual,
        residual_norm=evaluation.residual_norm, pole_margin=evaluation.pole_margin,
        step=0., krylov_info=0, krylov_iterations=0, direction='revalidated cRPA checkpoint',
        hermiticity_defect=h, pair_interchange_defect=p, krylov_method='lgmres')
    return GammaPhResult(problem=problem, kernel=kernel, history=(record,), converged=True,
                         status='revalidated_checkpoint', interaction_scale=scale,
                         initial_diagnostics={**metadata.get('initial_diagnostics', {}),
                                              'kind': 'revalidated cRPA checkpoint',
                                              'source_interaction': SOURCE_SCHEMA,
                                              'checkpoint': str(path)})


def solve_screened_parquet(problem, *, target_scale=1.0, starting_scale=0.05,
        stages=None, minimum_scale_step=1e-4, maximum_attempts=100,
        checkpoint_directory=None, resume=True, progress=None, detailed_progress=True,
        nonconvergence_policy='adaptive', tolerance_relaxation_factor=3.0,
        failed_iterate_directory=None, **solver_options):
    """Reference Newton/LGMRES procedure, adaptive coupling, converged-only resume.

    ``progress`` accepts dictionaries. The default nonmonotone minimum-step
    rule follows the reference n=6 notebook; pass minimum_line_search_step=0
    to require monotone residual decrease. Returned incomplete states must
    never be used as target-coupling superconductivity input.

    The opt-in ``relax_and_continue`` policy archives each failed stage and
    uses its actual K as a numerical seed for the next prescribed coupling,
    whose tolerance becomes ``tolerance_relaxation_factor * achieved_residual``.
    Failed stages are never relabeled converged; a failed final stage does not
    reach the target. The default ``adaptive`` policy is unchanged.
    """
    if nonconvergence_policy not in ('adaptive', 'relax_and_continue'):
        raise ValueError('nonconvergence_policy must be adaptive or relax_and_continue')
    if nonconvergence_policy == 'relax_and_continue':
        from .failed_iterate import solve_relaxed_continuation
        return solve_relaxed_continuation(
            problem, target_scale=target_scale, starting_scale=starting_scale,
            stages=stages, maximum_attempts=maximum_attempts,
            checkpoint_directory=checkpoint_directory, resume=resume,
            progress=progress, detailed_progress=detailed_progress,
            tolerance_relaxation_factor=tolerance_relaxation_factor,
            failed_iterate_directory=failed_iterate_directory, solver_options=solver_options)
    if not hasattr(problem, 'source_signature'):
        raise ValueError('construct the problem with build_screened_problem first')
    target, start = float(target_scale), float(starting_scale)
    if not np.isfinite(target) or target <= 0 or not np.isfinite(start) or start <= 0:
        raise ValueError('target_scale and starting_scale must be positive and finite')
    if not np.isfinite(minimum_scale_step) or minimum_scale_step <= 0:
        raise ValueError('minimum_scale_step must be positive and finite')
    if not isinstance(maximum_attempts, int) or maximum_attempts < 1:
        raise ValueError('maximum_attempts must be a positive integer')
    if 'initial_kernel' in solver_options or 'interaction_scale' in solver_options:
        raise ValueError('continuation owns its initial kernel and interaction scale')
    defaults = dict(tolerance=1e-7, maximum_iterations=100,
        krylov_method='lgmres', krylov_restart=24, krylov_maximum_iterations=3,
        krylov_outer_k=3, maximum_backtracks=20, minimum_line_search_step=0.05,
        pole_ceiling=0.8, preserve_physical_symmetry=True, stagnation_window=0,
        memory_mode='streamed', maximum_memory_mb=12000)
    defaults.update(solver_options)
    if stages is None:
        schedule = [min(start, target)]
        while schedule[-1] < target:
            following = min(target, schedule[-1] + start)
            if following <= schedule[-1]:
                raise ValueError('starting_scale is too small to advance continuation')
            schedule.append(following)
    else:
        schedule = [float(value) for value in stages]
        if (not schedule or any(not np.isfinite(x) or x <= 0 or x > target for x in schedule)
                or any(a >= b for a, b in zip(schedule, schedule[1:]))):
            raise ValueError('stages must be positive, increasing, and at most target_scale')
        if schedule[-1] != target:
            schedule.append(target)
    previous = (load_checkpoint(problem, checkpoint_directory,
                tolerance=defaults['tolerance'], target_scale=target)
                if resume and checkpoint_directory is not None else None)
    resumed_scale = None if previous is None else previous.interaction_scale
    if previous is not None:
        schedule = [x for x in schedule if x > previous.interaction_scale]
    records = []
    last_attempt = None
    started = perf_counter()
    status = 'target_reached' if not schedule else 'maximum_attempts'
    for attempt in range(1, maximum_attempts + 1):
        if not schedule:
            status = 'target_reached'
            break
        goal = schedule.pop(0)
        stage_started = perf_counter()
        seed_diagnostics = None
        if previous is None:
            seed = problem.unpack(goal * problem.bare_flat)
            seed, seed_diagnostics = _guard_seed(problem, seed, defaults['pole_ceiling'])
        else:
            seed = previous.kernel
        if progress is not None:
            progress({'event': 'stage_start', 'attempt': attempt, 'coupling': goal,
                      'initial': 'guarded scaled Lambda_c' if previous is None else 'previous converged K',
                      'elapsed_seconds': perf_counter() - started, 'stage_elapsed_seconds': 0.0})
        options = dict(defaults)
        if progress is not None:
            options['progress'] = lambda item: progress({
                'event': 'iteration', 'attempt': attempt, 'coupling': goal,
                'elapsed_seconds': perf_counter() - started,
                'stage_elapsed_seconds': perf_counter() - stage_started, **asdict(item)})
            if detailed_progress:
                options['work_progress'] = lambda event: progress({**event,
                    'attempt': attempt, 'coupling': goal,
                    'elapsed_seconds': perf_counter() - started,
                    'stage_elapsed_seconds': perf_counter() - stage_started})
        last_attempt = None
        result = None
        result = solve_gamma_ph(problem, interaction_scale=goal, initial_kernel=seed, **options)
        del seed
        result.initial_diagnostics.update(
            source_interaction=SOURCE_SCHEMA,
            kind='guarded scaled Lambda_c' if previous is None else 'previous converged cRPA K',
            seed_guard=seed_diagnostics)
        last_attempt = result
        record = {'attempt': attempt, 'coupling': goal, 'converged': result.converged,
                  'status': result.status, 'final': asdict(result.history[-1]),
                  'history': [asdict(x) for x in result.history]}
        records.append(record)
        if result.converged:
            previous = result
            if checkpoint_directory is not None:
                save_checkpoint(result, checkpoint_directory, target_scale=target,
                                continuation={'attempts': records}, tolerance=defaults['tolerance'])
            if goal == target:
                status = 'target_reached'
        else:
            lower = 0.0 if previous is None else previous.interaction_scale
            if goal - lower <= minimum_scale_step:
                status = 'minimum_scale_step'
            else:
                schedule[:0] = [0.5 * (lower + goal), goal]
        if progress is not None:
            progress({'event': 'stage_end', **record, 'elapsed_seconds': perf_counter() - started,
                      'stage_elapsed_seconds': perf_counter() - stage_started})
        if status in ('target_reached', 'minimum_scale_step'):
            break
    retained = previous if previous is not None else last_attempt
    if retained is None:
        raise RuntimeError('continuation produced no iterate')
    info = {'target_scale': target, 'target_reached': bool(retained.converged
            and retained.interaction_scale == target), 'retained_scale': retained.interaction_scale,
            'status': status, 'attempts': records, 'resume_from_scale': resumed_scale,
            'elapsed_seconds': perf_counter() - started, 'source_signature': problem.source_signature}
    return retained, info


def iter_full_gamma(result):
    """Yield the explicit parquet vertex, with a disk-backed crossed Phi.

    A yielded sector is an owned array, so it remains valid after this iterator
    closes its temporary mapping. The Newton equation and its K are unchanged.
    """
    if not result.converged:
        raise ValueError('full Gamma requires a converged Gamma_ph')
    with tempfile.TemporaryDirectory(prefix='abca_explicit_gamma_') as directory:
        path = Path(directory) / 'gamma.npy'
        _write_explicit_full_gamma(result, path)
        mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            for index, transfer in enumerate(result.problem.transfers):
                start, stop = result.problem.offsets[index:index + 2]
                size = len(result.problem.domain_coords[index])
                yield index, transfer, np.array(mapped[start:stop].reshape(size, size), copy=True)
        finally:
            mapped._mmap.close()


def _write_explicit_full_gamma(result, path):
    """Write g Lambda-C Phi+Phi with one mapped tensor and sector scratch.

    This primitive also serves failed-iterate diagnostics. It does not certify
    Newton convergence or production eligibility. C is the exact involution
    C[X]_Q(q,p)=X_(p-q)(p-Q,p), without a conjugation or additional sign.
    """
    problem = result.problem
    if not hasattr(problem, 'source_signature') or not hasattr(problem, 'source_metadata'):
        raise ValueError('explicit Gamma requires the constrained-RPA source identity')
    scale = float(result.interaction_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('explicit Gamma coupling must be positive and finite')
    kernel = problem._validate_blocks(result.kernel, 'kernel')
    source = problem.bare_flat
    source_norm2 = source_crossing2 = 0.0
    for start in range(0, problem.total_entries, 65536):
        stop = min(start + 65536, problem.total_entries)
        block = source[start:stop]
        if not np.all(np.isfinite(block)):
            raise ValueError('constrained-RPA source contains nonfinite entries')
        difference = block + source[problem.crossing_permutation[start:stop]]
        source_norm2 += float(np.vdot(block, block).real)
        source_crossing2 += float(np.vdot(difference, difference).real)
    source_crossing = float(np.sqrt(source_crossing2 / (source_norm2 or 1.0)))
    if not np.isfinite(source_crossing) or source_crossing > 1e-10:
        raise ValueError('constrained-RPA source violates crossing')
    projector = _PhysicalSymmetryProjector(problem)
    kernel_h, kernel_p = projector.defects(kernel)
    if (not np.isfinite(kernel_h) or not np.isfinite(kernel_p)
            or max(kernel_h, kernel_p) > 1e-8):
        raise ValueError('Gamma_ph violates physical Hermiticity or pair interchange')
    path = Path(path)
    mapped = np.lib.format.open_memmap(path, mode='w+', dtype=np.complex128,
                                      shape=(problem.total_entries,))
    phi_norm2 = phi_solve2 = legacy_norm2 = 0.0
    try:
        for index, (matrix, bubble) in enumerate(zip(kernel, problem.bubble)):
            response = np.eye(len(matrix), dtype=np.complex128) - matrix * bubble[None, :]
            rhs = (matrix * bubble[None, :]) @ matrix
            with warnings.catch_warnings():
                warnings.simplefilter('error', LinAlgWarning)
                phi = solve(response, rhs, assume_a='gen', overwrite_a=True,
                            overwrite_b=False, check_finite=True)
            if not np.all(np.isfinite(phi)):
                raise np.linalg.LinAlgError(f'nonfinite Phi at sector {index}')
            residual = phi - rhs - (matrix * bubble[None, :]) @ phi
            phi_solve2 += float(np.vdot(residual, residual).real)
            phi_norm2 += float(np.vdot(phi, phi).real)
            legacy = matrix + phi
            legacy_norm2 += float(np.vdot(legacy, legacy).real)
            mapped[problem.offsets[index]:problem.offsets[index + 1]] = phi.reshape(-1)
        # Each involution pair is visited once. Read both original Phi entries
        # before writing either Gamma entry; no full crossed tensor is needed.
        kernel_flat = problem.pack(kernel, copy=False)
        nonlinear2 = identity_error2 = 0.0
        for start in range(0, problem.total_entries, 65536):
            stop = min(start + 65536, problem.total_entries)
            indices = np.arange(start, stop, dtype=np.int64)
            partners = problem.crossing_permutation[start:stop]
            chosen = indices <= partners
            left, right = indices[chosen], partners[chosen]
            phi_left, phi_right = np.asarray(mapped[left]), np.asarray(mapped[right])
            left_value = scale * source[left] + phi_left - phi_right
            right_value = scale * source[right] + phi_right - phi_left
            left_r = kernel_flat[left] - scale * source[left] + phi_right
            right_r = kernel_flat[right] - scale * source[right] + phi_left
            unequal = left != right
            nonlinear2 += float(np.vdot(left_r, left_r).real)
            nonlinear2 += float(np.vdot(right_r[unequal], right_r[unequal]).real)
            error = left_value - (kernel_flat[left] + phi_left) + left_r
            identity_error2 += float(np.vdot(error, error).real)
            error = right_value - (kernel_flat[right] + phi_right) + right_r
            identity_error2 += float(np.vdot(error[unequal], error[unequal]).real)
            mapped[left], mapped[right] = left_value, right_value
        del kernel_flat
        norm2 = bse2 = crossing2 = 0.0
        blocks = problem.unpack(mapped)
        for gamma, matrix, bubble in zip(blocks, kernel, problem.bubble):
            residual = gamma - matrix - (matrix * bubble[None, :]) @ gamma
            norm2 += float(np.vdot(gamma, gamma).real)
            bse2 += float(np.vdot(residual, residual).real)
        for start in range(0, problem.total_entries, 65536):
            stop = min(start + 65536, problem.total_entries)
            difference = mapped[start:stop] + mapped[problem.crossing_permutation[start:stop]]
            crossing2 += float(np.vdot(difference, difference).real)
        hermiticity, interchange = projector.defects(blocks)
        denominator = norm2 or 1.0
        metadata = {
            'construction': FULL_GAMMA_CONSTRUCTION, 'formula': FULL_GAMMA_FORMULA,
            'crossing_routing': 'C[Phi]_Q(q,p)=Phi_(p-q)(p-Q,p)',
            'crossing_defect': float(np.sqrt(crossing2 / denominator)),
            'hermiticity_defect': float(hermiticity), 'pair_interchange_defect': float(interchange),
            'kernel_hermiticity_defect': float(kernel_h), 'kernel_pair_interchange_defect': float(kernel_p),
            'source_crossing_defect': source_crossing,
            'phi_linear_relative_residual': float(np.sqrt(phi_solve2 / (phi_norm2 or 1.0))),
            'legacy_bse_relative_residual': float(np.sqrt(phi_solve2 / (legacy_norm2 or 1.0))),
            'bse_relative_residual': float(np.sqrt(bse2 / denominator)),
            'bse_residual_note': 'Explicit Gamma has BSE defect -(I-K B) R away from a Newton root.',
            'gamma_ph_residual_norm': float(np.sqrt(nonlinear2)),
            'gamma_ph_relative_residual': float(np.sqrt(nonlinear2) / (scale * np.sqrt(source_norm2) or 1.0)),
            'legacy_difference_norm': float(np.sqrt(nonlinear2)),
            'legacy_difference_relative_norm': float(np.sqrt(nonlinear2 / (legacy_norm2 or 1.0))),
            'legacy_difference_identity': 'Gamma_explicit - Gamma_BSE = -R; R=K-g Lambda_c+C[Phi]',
            'legacy_difference_identity_error': float(np.sqrt(identity_error2 / denominator)),
        }
        for key, value in metadata.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise np.linalg.LinAlgError(f'nonfinite explicit Gamma diagnostic: {key}')
        if metadata['phi_linear_relative_residual'] > 1e-10:
            raise np.linalg.LinAlgError('Phi linear solve did not meet its accuracy threshold')
        mapped.flush()
        return metadata
    finally:
        mapped._mmap.close()


@dataclass
class GammaArchive:
    problem: object
    path: Path
    flat: np.ndarray
    metadata: dict

    def block(self, index):
        size = len(self.problem.domain_coords[index])
        return self.flat[self.problem.offsets[index]:self.problem.offsets[index + 1]].reshape(size, size)

    @property
    def blocks(self):
        return tuple(self.block(i) for i in range(len(self.problem.transfers)))

    def vertex_at(self, transfer, q, p):
        index = self.problem._sector_indices[tuple(transfer)]
        positions = self.problem._position_maps[index]
        return complex(self.block(index)[positions[tuple(q)], positions[tuple(p)]])


def reconstruct_full_gamma(result, output_directory, *, require_target=True, target_scale=1.0,
                           maximum_symmetry_error=1e-5):
    """Save the explicit parquet Gamma with finite-residual diagnostics.

    The linear Phi solve is checked independently. The resulting Gamma is
    crossing antisymmetric by construction; its BSE defect is measured and
    retained, not relabeled zero at a finite nonlinear residual.
    """
    if not result.converged:
        raise ValueError('cannot reconstruct production Gamma from a failed stage')
    if require_target and result.interaction_scale != target_scale:
        raise ValueError('target coupling was not reached; superconductivity is blocked')
    problem = result.problem
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / 'full_gamma.json'
    previous_metadata = json.loads(manifest.read_text()) if manifest.is_file() else None
    previous_file = previous_metadata.get('gamma_file') if previous_metadata else None
    filename = 'gamma_full_' + uuid.uuid4().hex + '.npy'
    path = output_directory / filename
    temporary = output_directory / ('.' + filename)
    try:
        diagnostics = _write_explicit_full_gamma(result, temporary)
        metadata = {**diagnostics, 'schema': FULL_GAMMA_SCHEMA,
            'source_signature': problem.source_signature, 'hf_used': False,
            'source_metadata': problem.source_metadata,
            'interaction_scale': result.interaction_scale, 'converged': True, 'production_eligible': True,
            'target_reached': result.interaction_scale == target_scale,
            'gamma_ph_recorded_relative_residual': result.history[-1].relative_residual,
            'requested_tolerance': result.initial_diagnostics.get('base_requested_tolerance',
                result.initial_diagnostics.get('requested_tolerance')),
            'effective_tolerance': result.initial_diagnostics.get('requested_tolerance'),
            'nonconvergence_policy': result.initial_diagnostics.get('nonconvergence_policy', 'adaptive'),
            'tolerance_relaxation_history': result.initial_diagnostics.get('tolerance_relaxation_history', []),
            'routing': '(q-Q,q;p-Q,p)',
            'cooper_routing': 'Gamma_(kprime-k)(kprime,-k)',
            'matrix_offsets': problem.offsets.tolist(),
            'transfers': problem.transfers.tolist()}
        if max(metadata[key] for key in ('crossing_defect', 'hermiticity_defect',
                                        'pair_interchange_defect')) > maximum_symmetry_error:
            raise ValueError('full Gamma symmetry error exceeds the pairing acceptance threshold; '
                             'tighten the nonlinear tolerance before pairing')
        metadata['gamma_file'] = filename
        metadata['gamma_file_sha256'] = _file_sha256(temporary)
        temporary.replace(path)
        try:
            if previous_metadata and previous_metadata.get('construction') != FULL_GAMMA_CONSTRUCTION:
                _preserve_legacy_gamma(output_directory, previous_metadata)
            _atomic_json(manifest, metadata)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        if (previous_file and previous_metadata.get('construction') == FULL_GAMMA_CONSTRUCTION
                and Path(previous_file).name == previous_file
                and previous_file.startswith('gamma_full_')):
            try:
                (output_directory / previous_file).unlink(missing_ok=True)
            except PermissionError:
                # An older GammaArchive can remain alive in a caller.
                pass
    finally:
        temporary.unlink(missing_ok=True)
    return GammaArchive(problem=problem, path=path,
                        flat=np.load(path, mmap_mode='r', allow_pickle=False), metadata=metadata)


def load_full_gamma(problem, directory, *, require_target=True, target_scale=1.0,
                    maximum_symmetry_error=1e-5):
    """Load full Gamma only for this source, mesh, gauge and target coupling."""
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / 'full_gamma.json').read_text())
    if (metadata.get('schema') != FULL_GAMMA_SCHEMA
            or metadata.get('construction') != FULL_GAMMA_CONSTRUCTION):
        raise ValueError('full Gamma uses an old construction; regenerate the explicit parquet vertex from K')
    if metadata.get('source_signature') != problem.source_signature:
        raise ValueError('full Gamma source/model signature differs')
    if not metadata.get('converged') or metadata.get('production_eligible') is not True:
        raise ValueError('full Gamma was not generated from converged Gamma_ph')
    if require_target and (not metadata.get('target_reached')
            or metadata.get('interaction_scale') != target_scale):
        raise ValueError('full Gamma did not reach the requested target coupling')
    for key in ('crossing_defect', 'hermiticity_defect', 'pair_interchange_defect'):
        value = float(metadata[key])
        if not np.isfinite(value) or value > maximum_symmetry_error:
            raise ValueError(f'full Gamma {key} exceeds the acceptance threshold')
    for key, threshold in (('phi_linear_relative_residual', 1e-10),
                           ('kernel_hermiticity_defect', 1e-8),
                           ('kernel_pair_interchange_defect', 1e-8),
                           ('source_crossing_defect', 1e-10)):
        value = float(metadata[key])
        if not np.isfinite(value) or value < 0 or value > threshold:
            raise ValueError(f'full Gamma {key} exceeds the acceptance threshold')
    for key in ('bse_relative_residual', 'gamma_ph_relative_residual',
                'legacy_difference_relative_norm'):
        value = float(metadata[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'full Gamma has invalid {key}')
    if (not np.array_equal(metadata['matrix_offsets'], problem.offsets)
            or not np.array_equal(metadata['transfers'], problem.transfers)):
        raise ValueError('full Gamma transfer domains differ')
    filename = metadata['gamma_file']
    if Path(filename).name != filename:
        raise ValueError('full Gamma file must be a local filename')
    path = directory / filename
    if _file_sha256(path) != metadata['gamma_file_sha256']:
        raise ValueError('full Gamma content checksum differs')
    flat = np.load(path, mmap_mode='r', allow_pickle=False)
    if flat.dtype != np.complex128 or flat.shape != (problem.total_entries,):
        flat._mmap.close()
        raise ValueError('full Gamma dtype or dimensions differ')
    for start in range(0, problem.total_entries, 65536):
        if not np.all(np.isfinite(flat[start:start + 65536])):
            flat._mmap.close()
            raise ValueError('full Gamma contains nonfinite entries')
    return GammaArchive(problem=problem, path=path, flat=flat, metadata=metadata)


def _preserve_legacy_gamma(directory, metadata):
    """Keep a self-contained immutable copy before replacing an old formula."""
    directory = Path(directory)
    filename = metadata.get('gamma_file')
    if not filename or Path(filename).name != filename:
        raise ValueError('legacy Gamma must refer to a local data file')
    source = directory / filename
    if not source.is_file():
        raise FileNotFoundError('legacy Gamma data is missing; cannot preserve its archive')
    history_root = directory / 'legacy_history'
    history_root.mkdir(parents=True, exist_ok=True)
    # Keep this component short: production directory labels and the tensor's
    # generation filename already consume much of the Windows MAX_PATH budget.
    for _ in range(100):
        history = history_root / uuid.uuid4().hex[:12]
        try:
            history.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError('could not allocate a unique legacy Gamma history directory')
    try:
        # Both names refer to immutable generation files on the same volume;
        # this preserves the old tensor without another full disk allocation.
        os.link(source, history / filename)
    except OSError:
        shutil.copy2(source, history / filename)
    _atomic_json(history / 'full_gamma.json', metadata)
    # Leave the original data too: an open caller-owned GammaArchive can keep
    # using it. Both the old bytes and its manifest remain explicitly labelled.
    return history


def ensure_full_gamma(problem, directory, *, target_scale=1.0, require_target=True,
                      maximum_symmetry_error=1e-5, checkpoint_directory=None, tolerance=None):
    """Load the current formula, or rebuild a legacy/missing archive from K.

    Never runs Newton. Legacy migration requires a source-validated, converged
    checkpoint at the requested coupling. Existing legacy data is preserved.
    """
    directory = Path(directory).expanduser().resolve()
    manifest = directory / 'full_gamma.json'
    old = json.loads(manifest.read_text()) if manifest.is_file() else None
    if old is not None and old.get('construction') == FULL_GAMMA_CONSTRUCTION:
        return load_full_gamma(problem, directory, require_target=require_target,
            target_scale=target_scale, maximum_symmetry_error=maximum_symmetry_error)
    if old is not None:
        if old.get('schema') != 'abca_crpa_full_gamma_v1':
            raise ValueError('unrecognized full Gamma archive; explicit migration is required')
        if old.get('source_signature') != problem.source_signature:
            raise ValueError('legacy full Gamma source/model signature differs')
    checkpoint_directory = directory / 'checkpoints' if checkpoint_directory is None else Path(checkpoint_directory)
    checkpoint_path = checkpoint_directory / 'checkpoint.json'
    if not checkpoint_path.is_file():
        raise ValueError('explicit Gamma reconstruction requires a converged saved K checkpoint')
    saved = json.loads(checkpoint_path.read_text())
    controls = saved.get('initial_diagnostics') or {}
    relaxed = (saved.get('continuation') or {}).get('relaxation_state') or {}
    recorded_tolerance = relaxed.get('effective_tolerance', controls.get('requested_tolerance'))
    tolerance_source = 'caller'
    if tolerance is None:
        if recorded_tolerance is not None:
            tolerance = float(recorded_tolerance)
            tolerance_source = 'saved_effective_tolerance'
        else:
            # Original checkpoints did not save their requested tolerance.
            # Validate against their achieved residual, with rounding slack,
            # without claiming a new requested convergence tolerance.
            tolerance = max(float(saved['relative_residual']) * (1.0 + 1e-8) + 1e-12, 1e-12)
            tolerance_source = 'saved_achieved_residual_with_rounding_margin'
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('checkpoint validation tolerance must be finite and positive')
    result = load_checkpoint(problem, checkpoint_directory, tolerance=tolerance, target_scale=target_scale)
    if result is None or (require_target and result.interaction_scale != target_scale):
        raise ValueError('converged K checkpoint did not reach the requested coupling')
    result.initial_diagnostics.update(migration_validation_tolerance=tolerance,
                                     migration_validation_tolerance_source=tolerance_source)
    archive = reconstruct_full_gamma(result, directory, require_target=require_target,
        target_scale=target_scale, maximum_symmetry_error=maximum_symmetry_error)
    archive.metadata['checkpoint_validation_tolerance'] = tolerance
    archive.metadata['checkpoint_validation_tolerance_source'] = tolerance_source
    _atomic_json(manifest, archive.metadata)
    return archive
