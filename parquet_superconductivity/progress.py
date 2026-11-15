"""Bounded progress output for the all-transfer solver."""
from dataclasses import asdict
import numpy as np

class GammaPhProgressPrinter:
    """Print phases and accepted states without storing any vertex arrays.

    Inner GMRES output is throttled at callback boundaries, not by a timer
    thread. One expensive Jacobian action can exceed the requested interval.
    """

    def __init__(self, interval_seconds=30.0):
        self.interval_seconds = float(interval_seconds)
        if not np.isfinite(self.interval_seconds) or self.interval_seconds < 0.0:
            raise ValueError('progress interval must be finite and nonnegative')
        self._last_inner_print = -float('inf')

    def accepted(self, item):
        """Adapter for solve_gamma_ph's accepted-state callback."""
        self({'event': 'iteration', **asdict(item)})

    def __call__(self, event):
        name = event['event']
        krylov_name = event.get('krylov_method', 'gmres').upper()
        elapsed = event.get('stage_elapsed_seconds', event.get('elapsed_seconds', 0.0))
        timing = f" [{elapsed:.1f}s]" if 'elapsed_seconds' in event else ''
        if name == 'stage_start':
            self._last_inner_print = -float('inf')
            message = f"Stage {event['attempt']}: g={event['coupling']:g}, seed={event['initial']}"
        elif name == 'stage_end':
            final = event['final']
            message = (f"  {event['status']}: nonlinear residual={final['relative_residual']:.3e}, "
                       f"Hermiticity={final['hermiticity_defect']:.3e}, "
                       f"pole distance={final['pole_margin']:.3e}" + timing)
        elif name == 'iteration':
            message = (f"  Newton {event['iteration']}: nonlinear residual={event['relative_residual']:.3e}, "
                       f"Hermiticity={event['hermiticity_defect']:.3e}, "
                       f"pole distance={event['pole_margin']:.3e}")
            if event['iteration']:
                message += (f", step={event['step']:.3g}, {krylov_name} iterations={event['krylov_iterations']}, "
                            f"info={event['krylov_info']}, direction={event['direction']}")
            message += timing
        elif name == 'initialization_start':
            message = f"  Preparing {event['seed_kind']} seed and checking poles..." + timing
        elif name == 'evaluation_start' and event.get('scope') == 'initial':
            message = "  Computing the initial all-Q nonlinear residual..." + timing
        elif name == 'krylov_start':
            self._last_inner_print = -float('inf')
            message = (f"  Newton {event['iteration']}: {krylov_name} started, "
                       f"linear rtol={event['inner_tolerance']:.2e}, "
                       f"at most {event['maximum_inner_iterations']} inner iterations" + timing)
            if krylov_name == 'LGMRES':
                message += (f"; basis={event['restart']} + up to {event['outer_k']} recycled, "
                            f"cycles={event['maximum_restart_cycles']}")
        elif name == 'krylov_iteration':
            if elapsed - self._last_inner_print < self.interval_seconds:
                return
            self._last_inner_print = elapsed
            message = (f"    {krylov_name} {event['inner_iteration']}: "
                       f"linear residual={event['inner_relative_residual']:.3e}" + timing)
        elif name == 'krylov_matvec':
            if elapsed - self._last_inner_print < self.interval_seconds:
                return
            self._last_inner_print = elapsed
            message = (f"    {krylov_name} cycle {event['cycle']}: "
                       f"Krylov action {event['inner_iteration']} completed" + timing)
        elif name == 'krylov_cycle':
            message = (f"    {krylov_name} cycle {event['cycle']} finished: "
                       f"true linear residual={event['linear_relative_residual']:.3e}, "
                       f"recycled directions={event['recycled_vectors']}" + timing)
        elif name == 'krylov_end':
            message = (f"    {krylov_name} finished: info={event['info']}, "
                       f"iterations={event['inner_iterations']}, "
                       f"duration={event['operation_elapsed_seconds']:.1f}s" + timing)
        elif name == 'krylov_recycle_reset':
            message = "    Cleared recycled directions after a fallback or strongly damped step." + timing
        elif name == 'line_search_start':
            message = f"    Checking {event['direction']} direction and backtracking..." + timing
        elif name == 'direction_check':
            message = (f"    Direction: true linear residual={event['true_linear_relative_residual']:.3e}, "
                       f"relative correction={event['relative_correction']:.3e}" + timing)
        elif name == 'stalled_newton':
            message = (f"    Newton step={event['step']:.3g} improves residual by only "
                       f"{100 * event['relative_improvement']:.3g}%; trying a fallback..." + timing)
        elif name == 'stall_recovery':
            message = (f"    Fallback {event['direction']} improves residual by "
                       f"{100 * event['relative_improvement']:.3g}%" + timing)
        elif name == 'stagnation':
            message = "    Residual stalled; stopping this coupling attempt for continuation retry." + timing
        elif name == 'line_search_trial':
            message = f"    Trial {event['trial']}: step={event['step']:.3g}, evaluating all Q..." + timing
        elif name == 'line_search_reject':
            residual = event.get('relative_residual')
            detail = '' if residual is None else f", nonlinear residual={residual:.3e}"
            message = f"    Rejected: {event['reason']}" + detail + timing
        elif name == 'line_search_accept' and event.get('reason') == 'minimum_step_reached':
            message = (f"    Accepted at minimum step={event['step']:g}: nonlinear residual "
                       f"{event['previous_relative_residual']:.3e} -> {event['relative_residual']:.3e}" + timing)
        elif name == 'line_search_end' and event['status'] != 'accepted':
            message = f"    Backtracking: {event['status']}" + timing
        elif name == 'merit_gradient_start':
            message = "    Computing fallback merit gradient..." + timing
        else:
            return
        print(message, flush=True)
