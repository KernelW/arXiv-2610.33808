"""Unconverged diagnostic archives and opt-in tolerance-relaxed continuation.

These archives are never production checkpoints. A diagnostic full Gamma uses
g Lambda_c-C[Phi]+Phi from the saved, unconverged K; its nonlinear residual
and finite BSE defect remain explicit.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import uuid

import numpy as np
import scipy

FAILED_SCHEMA = 'abca_crpa_failed_iterate_v1'
STATE_SCHEMA = 'abca_crpa_relaxed_continuation_v1'


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True,
                                        allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _close_map(mapped):
    if mapped is not None:
        try:
            mapped.flush()
        finally:
            mapped._mmap.close()


def _provenance():
    paths = [Path(__file__), Path(__file__).with_name('parquet.py')]
    module = sys.modules.get('static_parquet_memory.gamma_ph_only')
    if module is not None and getattr(module, '__file__', None):
        paths.append(Path(module.__file__))
    return {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'code_sha256': {str(p.resolve()): _sha256(p) for p in paths if p.is_file()}}


def save_failed_iterate(result, directory, *, requested_tolerance, target_scale,
                        maximum_iterations, solver_controls=None, continuation=None):
    """Commit the actual failed K first, then attempt a diagnostic full Gamma.

    Each attempt has unique filenames; earlier diagnostics and converged
    checkpoints remain intact. Mappings are closed before Windows rename/unlink.
    """
    if result.converged:
        raise ValueError('save_failed_iterate requires an unconverged result')
    tolerance = float(requested_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('requested_tolerance must be positive and finite')
    if not result.history:
        raise ValueError('failed result must have an iteration history')
    problem = result.problem
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    kernel_path = directory / f'failed_gamma_ph_{generation}.npy'
    gamma_path = directory / f'failed_gamma_full_{generation}.npy'
    manifest = directory / f'failed_iterate_{generation}.json'
    temporary = kernel_path.with_name('.' + kernel_path.name + '.tmp')
    mapped = None
    try:
        mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.complex128,
                                          shape=(problem.total_entries,))
        for i, block in enumerate(result.kernel):
            mapped[problem.offsets[i]:problem.offsets[i + 1]] = block.reshape(-1)
        _close_map(mapped)
        mapped = None
        temporary.replace(kernel_path)
    finally:
        _close_map(mapped)
        temporary.unlink(missing_ok=True)
    final = asdict(result.history[-1])
    metadata = {
        'schema': FAILED_SCHEMA, 'converged': False, 'production_eligible': False,
        'target_reached': False, 'coupling_equals_target': result.interaction_scale == target_scale,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': result.status, 'source_signature': problem.source_signature,
        'source_metadata': problem.source_metadata, 'provenance': _provenance(),
        'coupling': float(result.interaction_scale), 'interaction_scale': float(result.interaction_scale),
        'target_scale': float(target_scale), 'requested_tolerance': tolerance,
        'achieved_relative_residual': final['relative_residual'],
        'achieved_residual_norm': final['residual_norm'], 'iteration': int(final['iteration']),
        'maximum_iterations': int(maximum_iterations), 'final_diagnostics': final,
        'history': [asdict(item) for item in result.history],
        'initial_diagnostics': result.initial_diagnostics, 'solver_controls': solver_controls,
        'continuation': continuation, 'matrix_offsets': problem.offsets.tolist(),
        'transfers': problem.transfers.tolist(), 'kernel_file': kernel_path.name,
        'kernel_sha256': _sha256(kernel_path), 'kernel_path': str(kernel_path),
        'manifest_path': str(manifest), 'full_gamma_path': None,
        'full_gamma_status': 'pending',
        'full_gamma_note': 'Explicit parquet reconstruction from unconverged K; not eligible for pairing.',
    }
    def publish():
        _atomic_json(manifest, metadata)
        _atomic_json(directory / 'latest_failed_iterate.json', metadata)
    publish()  # The actual K remains discoverable even if full-Gamma work is interrupted.
    temporary = gamma_path.with_name('.' + gamma_path.name + '.tmp')
    try:
        from .parquet import _write_explicit_full_gamma, FULL_GAMMA_SCHEMA, FULL_GAMMA_CONSTRUCTION
        diagnostics = _write_explicit_full_gamma(result, temporary)
        temporary.replace(gamma_path)
        metadata.update(diagnostics, full_gamma_status='saved_unconverged_diagnostic',
                        full_gamma_schema=FULL_GAMMA_SCHEMA,
                        full_gamma_construction=FULL_GAMMA_CONSTRUCTION,
                        full_gamma_path=str(gamma_path), full_gamma_file=gamma_path.name,
                        full_gamma_sha256=_sha256(gamma_path),
                        full_gamma_formula=diagnostics['formula'])
    except Exception as exc:
        metadata.update(full_gamma_status='reconstruction_failed',
                        full_gamma_error=f'{type(exc).__name__}: {exc}')
    except BaseException as exc:
        metadata.update(full_gamma_status='interrupted', full_gamma_error=type(exc).__name__)
        publish()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    publish()
    return {key: _json_value(metadata[key]) for key in (
        'manifest_path', 'kernel_path', 'full_gamma_path', 'full_gamma_status',
        'coupling', 'requested_tolerance', 'achieved_relative_residual', 'iteration',
        'maximum_iterations', 'status')}


def _load_failed_seed(problem, manifest_path, *, revalidate_residual=True):
    from static_parquet_memory.gamma_ph_only import GammaPhIteration, GammaPhResult, evaluate_gamma_ph
    path = Path(manifest_path).expanduser().resolve()
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if (metadata.get('schema') != FAILED_SCHEMA or metadata.get('converged') is not False
            or metadata.get('source_signature') != problem.source_signature):
        raise ValueError('failed-seed provenance differs from the current problem')
    if (not np.array_equal(metadata['matrix_offsets'], problem.offsets)
            or not np.array_equal(metadata['transfers'], problem.transfers)):
        raise ValueError('failed-seed transfer domains differ')
    name = metadata['kernel_file']
    if Path(name).name != name:
        raise ValueError('failed-seed kernel must be a local filename')
    kernel_path = path.parent / name
    if _sha256(kernel_path) != metadata['kernel_sha256']:
        raise ValueError('failed-seed kernel checksum differs')
    flat = np.load(kernel_path, allow_pickle=False)
    if flat.dtype != np.complex128 or flat.shape != (problem.total_entries,):
        raise ValueError('failed-seed kernel dimensions or dtype differ')
    kernel = problem.unpack(flat)
    if revalidate_residual:
        evaluation = evaluate_gamma_ph(problem, kernel, interaction_scale=metadata['coupling'],
                                      want_jacobian=False, memory_mode='streamed')
        if not np.isclose(evaluation.relative_residual, metadata['achieved_relative_residual'],
                          rtol=1e-8, atol=1e-12):
            raise ValueError('failed-seed actual residual differs from its diagnostic record')
    history = tuple(GammaPhIteration(**record) for record in metadata['history'])
    return GammaPhResult(problem=problem, kernel=kernel, history=history, converged=False,
                         status=metadata['status'], interaction_scale=metadata['coupling'],
                         initial_diagnostics={'kind': 'revalidated unconverged diagnostic seed',
                                              'manifest_path': str(path)})


def solve_relaxed_continuation(problem, *, target_scale, starting_scale, stages,
        maximum_attempts, checkpoint_directory, resume, progress, detailed_progress,
        tolerance_relaxation_factor, failed_iterate_directory, solver_options):
    """Prescribed-stage continuation with durable, source-validated adaptive tolerance."""
    from . import parquet as pq
    if not hasattr(problem, 'source_signature'):
        raise ValueError('construct the problem with build_screened_problem first')
    target, start, factor = map(float, (target_scale, starting_scale, tolerance_relaxation_factor))
    if any(not np.isfinite(x) or x <= 0 for x in (target, start)):
        raise ValueError('target_scale and starting_scale must be positive and finite')
    if not np.isfinite(factor) or factor <= 1:
        raise ValueError('tolerance_relaxation_factor must be finite and greater than one')
    if not isinstance(maximum_attempts, int) or maximum_attempts < 1:
        raise ValueError('maximum_attempts must be a positive integer')
    if 'initial_kernel' in solver_options or 'interaction_scale' in solver_options:
        raise ValueError('continuation owns its initial kernel and interaction scale')
    defaults = dict(tolerance=1e-7, maximum_iterations=100, krylov_method='lgmres',
        krylov_restart=24, krylov_maximum_iterations=3, krylov_outer_k=3,
        maximum_backtracks=20, minimum_line_search_step=.05, pole_ceiling=.8,
        preserve_physical_symmetry=True, stagnation_window=0, memory_mode='streamed',
        maximum_memory_mb=12000)
    defaults.update(solver_options)
    base_tolerance = float(defaults['tolerance'])
    if not np.isfinite(base_tolerance) or base_tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    if stages is None:
        schedule = [min(start, target)]
        while schedule[-1] < target:
            value = min(target, schedule[-1] + start)
            if value <= schedule[-1]:
                raise ValueError('starting_scale is too small to advance continuation')
            schedule.append(value)
    else:
        schedule = [float(x) for x in stages]
        if (not schedule or any(not np.isfinite(x) or not 0 < x <= target for x in schedule)
                or any(a >= b for a, b in zip(schedule, schedule[1:]))):
            raise ValueError('stages must be positive, increasing and at most target_scale')
        if schedule[-1] != target:
            schedule.append(target)
    checkpoints = None if checkpoint_directory is None else Path(checkpoint_directory).expanduser().resolve()
    if failed_iterate_directory is None:
        if checkpoints is None:
            raise ValueError('failed_iterate_directory or checkpoint_directory is required')
        failed_iterate_directory = checkpoints.parent / 'failed_iterates'
    failures_dir = Path(failed_iterate_directory).expanduser().resolve()
    state_path = (checkpoints.parent if checkpoints is not None else failures_dir) / 'relaxation_state.json'
    controls = {k: v for k, v in defaults.items() if k not in ('progress', 'work_progress')}
    contract = {'source_signature': problem.source_signature, 'target_scale': target,
                'stages': schedule, 'solver_controls': _json_value(controls), 'factor': factor}
    contract_signature = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    records, relaxations, failures = [], [], []
    seed_result = previous = None
    last_converged_scale = last_converged_tolerance = None
    next_index, current_tolerance, effective_tolerance = 0, base_tolerance, base_tolerance
    resumed_scale = None
    candidates = []
    if resume and state_path.is_file():
        candidates.append(json.loads(state_path.read_text(encoding='utf-8')))
    checkpoint_metadata = None
    if resume and checkpoints is not None and (checkpoints / 'checkpoint.json').is_file():
        checkpoint_metadata = json.loads((checkpoints / 'checkpoint.json').read_text())
        embedded = (checkpoint_metadata.get('continuation') or {}).get('relaxation_state')
        if embedded is not None:
            candidates.append(embedded)
    for saved in candidates:
        if saved.get('schema') != STATE_SCHEMA or saved.get('contract_signature') != contract_signature:
            raise ValueError('adaptive-tolerance state differs; use a separate output directory')
    state = max(candidates, key=lambda item: item['next_stage_index']) if candidates else None
    if checkpoint_metadata is not None:
        checkpoint_tolerance = base_tolerance
        embedded = (checkpoint_metadata.get('continuation') or {}).get('relaxation_state')
        if embedded is not None:
            checkpoint_tolerance = embedded['effective_tolerance']
        previous = pq.load_checkpoint(problem, checkpoints, tolerance=checkpoint_tolerance, target_scale=target)
        last_converged_scale = previous.interaction_scale
        last_converged_tolerance = checkpoint_tolerance
    if state is not None:
        records, relaxations, failures = state['attempts'], state['relaxations'], state['failed_iterates']
        next_index, current_tolerance = state['next_stage_index'], state['next_tolerance']
        effective_tolerance = state['effective_tolerance']
        if state['seed']['kind'] == 'failed':
            seed_result = _load_failed_seed(problem, state['seed']['manifest_path'],
                revalidate_residual=current_tolerance is not None)
            # Keep only the actual next seed during Newton. A last converged
            # checkpoint can be loaded again if needed for an incomplete return.
            previous = None
        else:
            seed_result = previous
        if seed_result is None or seed_result.interaction_scale != state['seed']['coupling']:
            raise ValueError('adaptive-tolerance seed and durable state disagree')
        resumed_scale = seed_result.interaction_scale
    elif previous is not None:
        seed_result = previous
        resumed_scale = previous.interaction_scale
        next_index = sum(g <= previous.interaction_scale for g in schedule)
    started = perf_counter()
    status = state.get('status', 'maximum_attempts') if state else 'maximum_attempts'

    def information():
        reached = bool(seed_result is not None and seed_result.converged
                       and seed_result.interaction_scale == target)
        retained_scale = last_converged_scale if last_converged_scale is not None else (
            None if seed_result is None else seed_result.interaction_scale)
        return _json_value({'target_scale': target, 'target_reached': reached,
                'retained_scale': retained_scale,
                'last_attempt_scale': None if seed_result is None else seed_result.interaction_scale,
                'last_converged_scale': last_converged_scale,
                'status': 'target_reached' if reached else status, 'attempts': records,
                'resume_from_scale': resumed_scale, 'elapsed_seconds': perf_counter() - started,
                'source_signature': problem.source_signature, 'nonconvergence_policy': 'relax_and_continue',
                'requested_tolerance': base_tolerance, 'effective_tolerance': effective_tolerance,
                'next_tolerance': current_tolerance, 'tolerance_relaxation_factor': factor,
                'relaxations': relaxations, 'failed_iterates': failures,
                'failed_iterate': failures[-1] if failures else None,
                'relaxation_state_path': str(state_path)})

    for local_attempt in range(maximum_attempts):
        if next_index >= len(schedule) or current_tolerance is None:
            break
        goal = schedule[next_index]
        effective_tolerance = current_tolerance
        stage_started = perf_counter()
        attempt = len(records) + 1
        seed_diagnostics = None
        if seed_result is None:
            seed = problem.unpack(goal * problem.bare_flat)
            seed, seed_diagnostics = pq._guard_seed(problem, seed, defaults['pole_ceiling'])
            seed_kind = 'guarded scaled Lambda_c'
        else:
            seed = seed_result.kernel
            seed_kind = 'previous converged K' if seed_result.converged else 'previous unconverged diagnostic K'
        if progress is not None:
            progress({'event': 'stage_start', 'attempt': attempt, 'coupling': goal,
                      'initial': seed_kind, 'requested_tolerance': effective_tolerance,
                      'elapsed_seconds': perf_counter() - started, 'stage_elapsed_seconds': 0.})
        options = dict(defaults, tolerance=effective_tolerance)
        if progress is not None:
            options['progress'] = lambda item: progress({
                'event': 'iteration', 'attempt': attempt, 'coupling': goal,
                'elapsed_seconds': perf_counter() - started,
                'stage_elapsed_seconds': perf_counter() - stage_started, **asdict(item)})
            if detailed_progress:
                options['work_progress'] = lambda event: progress({**event, 'attempt': attempt,
                    'coupling': goal, 'elapsed_seconds': perf_counter() - started,
                    'stage_elapsed_seconds': perf_counter() - stage_started})
        result = pq.solve_gamma_ph(problem, interaction_scale=goal, initial_kernel=seed, **options)
        del seed
        result.initial_diagnostics.update(kind=seed_kind, seed_guard=seed_diagnostics,
            source_interaction=pq.SOURCE_SCHEMA, requested_tolerance=effective_tolerance,
            base_requested_tolerance=base_tolerance, nonconvergence_policy='relax_and_continue',
            tolerance_relaxation_history=list(relaxations))
        record = {'attempt': attempt, 'coupling': goal, 'converged': result.converged,
                  'status': result.status, 'tolerance': effective_tolerance,
                  'requested_tolerance': effective_tolerance,
                  'final': asdict(result.history[-1]), 'history': [asdict(x) for x in result.history]}
        records.append(record)
        next_index += 1
        seed_result = result
        if result.converged:
            previous = result
            last_converged_scale, last_converged_tolerance = goal, effective_tolerance
            status = 'target_reached' if goal == target else 'maximum_attempts'
            seed_info = {'kind': 'checkpoint', 'coupling': goal}
        else:
            failure = save_failed_iterate(result, failures_dir,
                requested_tolerance=effective_tolerance, target_scale=target,
                maximum_iterations=defaults['maximum_iterations'],
                solver_controls=dict(controls, tolerance=effective_tolerance),
                continuation={'attempts': records, 'last_converged_scale': last_converged_scale})
            failures.append(failure)
            seed_info = {'kind': 'failed', 'coupling': goal, 'manifest_path': failure['manifest_path']}
            achieved = result.history[-1].relative_residual
            proposed = factor * achieved
            if not np.isfinite(achieved) or achieved <= 0 or not np.isfinite(proposed) or proposed <= 0:
                current_tolerance, status = None, 'invalid_relaxation_residual'
            elif next_index == len(schedule):
                current_tolerance, status = None, 'final_stage_nonconverged'
            else:
                current_tolerance, status = float(proposed), 'maximum_attempts'
                relaxations.append({**failure, 'next_coupling': schedule[next_index],
                                    'next_tolerance': current_tolerance})
            if checkpoints is not None:
                previous = None
        state = {'schema': STATE_SCHEMA, 'contract_signature': contract_signature,
                 'source_signature': problem.source_signature, 'contract': contract,
                 'next_stage_index': next_index, 'next_tolerance': current_tolerance,
                 'requested_tolerance': base_tolerance, 'effective_tolerance': effective_tolerance,
                 'seed': seed_info, 'status': status, 'attempts': records,
                 'relaxations': relaxations, 'failed_iterates': failures}
        if result.converged and checkpoints is not None:
            pq.save_checkpoint(result, checkpoints, target_scale=target,
                               continuation={'relaxation_state': state}, tolerance=effective_tolerance)
        # Without checkpoint storage a successful seed cannot be resumed; avoid promising it.
        if checkpoints is not None or not result.converged:
            _atomic_json(state_path, state)
        if checkpoints is not None:
            _atomic_json(checkpoints.parent / 'continuation.json', information())
        if progress is not None:
            progress({'event': 'stage_end', **record, 'elapsed_seconds': perf_counter() - started,
                      'stage_elapsed_seconds': perf_counter() - stage_started,
                      'next_tolerance': current_tolerance})
        if goal == target or current_tolerance is None:
            break
    if previous is None and last_converged_scale is not None and checkpoints is not None:
        previous = pq.load_checkpoint(problem, checkpoints, tolerance=last_converged_tolerance,
                                      target_scale=target)
    retained = previous if previous is not None else seed_result
    if retained is None:
        raise RuntimeError('continuation produced no iterate')
    return retained, information()
