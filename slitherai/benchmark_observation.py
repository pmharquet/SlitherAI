"""Small repeatable CUDA sweep for the observation sensor chunk size.

This is an offline benchmark helper. It builds deterministic worlds and never
changes the trainer's saved configuration or policy.
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import pickle
import statistics
import time
from pathlib import Path

import numpy as np

import torch

from . import evaluation
from .config import SimConfig
from .schema import INPUTS, OUTPUTS, VERSION
from .sim import WorldBatch, adaptive_sensor_tiling


def _fixed_actions(world):
    """Give worms stable, varied headings while warming up the world."""
    m, w = world.c.maps, world.c.worms
    phase = torch.arange(m * w, device=world.device, dtype=torch.float32).reshape(m, w)
    direction = world.heading + .18 * torch.sin(phase * .37)
    actions = torch.zeros((m, w, 2), device=world.device)
    actions[..., 1] = torch.remainder(direction / math.tau, 1.)
    return actions


def _median_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.)
    return statistics.median(samples)


def sensor_ray_block_count(config, sensor_chunk):
    observers = config.maps * config.worms
    block_size = sensor_chunk * config.worms
    return math.ceil(observers / block_size)


def run_sweep(config, chunks=(4, 8, 16), seed=1, warmup_ticks=12, repeats=10):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark')
    if repeats < 1 or warmup_ticks < 0 or not chunks:
        raise ValueError('Need positive repeats, nonnegative warmup ticks, and at least one chunk')

    results = []
    reference = None
    for chunk in chunks:
        c = dataclasses.replace(config, sensor_chunk=chunk).validate()
        world = WorldBatch(c, device='cuda', seed=seed)
        actions = _fixed_actions(world)
        for _ in range(warmup_ticks):
            world.observe()
            world.step(actions)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        observation = world.observe()
        if reference is None:
            reference = observation.detach().clone()
        difference = (observation - reference).abs()
        exact = torch.equal(observation, reference)
        observe_ms = _median_ms(world.observe, repeats)
        tick_ms = _median_ms(lambda: (world.observe(), world.step(actions)), repeats)
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results.append(dict(
            sensor_chunk=chunk,
            raycast_blocks=sensor_ray_block_count(c, chunk),
            observation_shape=list(observation.shape),
            observation_exact_to_reference=exact,
            observation_max_abs_difference=float(difference.max().item()),
            observe_median_ms=observe_ms,
            observe_and_world_step_median_ms=tick_ms,
            peak_allocated_mib=peak_mib,
            alive=int(world.alive.sum().item()),
            active_body_points=world.active_body_points(),
        ))
        del world, actions, observation, difference
        torch.cuda.synchronize()

    return dict(
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        reference_chunk=chunks[0],
        seed=seed,
        warmup_ticks=warmup_ticks,
        repeats=repeats,
        config=dataclasses.asdict(config),
        results=results,
    )


def _tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode('utf-8'))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            digest.update(str(tensor.dtype).encode('ascii'))
            digest.update(str(tuple(tensor.shape)).encode('ascii'))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode('ascii'))
    return digest.hexdigest()


_TRACE_STATE_FIELDS = (
    'head', 'body', 'heading', 'target', 'mass', 'alive', 'boost', 'speed',
    'food', 'food_mass', 'food_size', 'age', 'gained', 'spent', 'kills',
    'decisions', 'boost_steps', 'turn_sum', 'border_deaths', 'collision_deaths',
    'prey_heading',
)


def _freeze_trace_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().cpu().clone()
    return value


def _field_difference(left, right):
    """Describe exact equality and numeric distance without a tolerance."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        a, b = left.detach().cpu(), right.detach().cpu()
    else:
        a, b = torch.as_tensor(left), torch.as_tensor(right)
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return dict(exact_equal=False, left_shape=list(a.shape), right_shape=list(b.shape),
                    bytewise_equal=False,
                    left_dtype=str(a.dtype), right_dtype=str(b.dtype),
                    different_count=None, max_abs_difference=None)
    equal = torch.equal(a, b)
    bytewise_equal = torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                 b.contiguous().reshape(-1).view(torch.uint8))
    if not a.numel():
        different_count, max_abs = 0, 0.
    elif a.dtype.is_floating_point or a.dtype.is_complex:
        different_count = int((a != b).sum().item())
        if a.dtype.is_complex:
            delta = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
        else:
            delta = (a.double() - b.double()).abs()
        finite = torch.isfinite(delta)
        max_abs = float(delta[finite].max().item()) if finite.any() else None
    else:
        different_count = int((a != b).sum().item())
        delta = (a.to(torch.int64) - b.to(torch.int64)).abs()
        max_abs = float(delta.max().item())
    return dict(exact_equal=equal, bytewise_equal=bytewise_equal,
                shape=list(a.shape), dtype=str(a.dtype),
                different_count=different_count, max_abs_difference=max_abs)


def _event_difference(left_values, right_values):
    left = dict(left_values)
    right = dict(right_values)
    names = list(dict.fromkeys([*left, *right]))
    fields = {}
    for name in names:
        if name not in left or name not in right:
            fields[name] = dict(exact_equal=False,
                                missing_from='left' if name not in left else 'right',
                                different_count=None, max_abs_difference=None)
        else:
            fields[name] = _field_difference(left[name], right[name])
    return fields


def _divergence_record(group, step, fields):
    different = [name for name, value in fields.items()
                 if not value.get('bytewise_equal', value['exact_equal'])]
    result = dict(step=step, first_field=different[0] if different else None, fields=fields)
    if group == 'states':
        rng = fields.get('rng_state')
        result['rng_state_equal'] = None if rng is None else rng['exact_equal']
        result['non_float_state_equal'] = {
            name: value['exact_equal'] for name, value in fields.items()
            if value.get('dtype', value.get('left_dtype', '')).startswith(
                ('torch.bool', 'torch.int', 'torch.uint'))
        }
    return result


def _trace_event_tensor_bytes(values):
    return sum(value.numel() * value.element_size() for _, value in values
               if isinstance(value, torch.Tensor))


@contextlib.contextmanager
def _capture_episode_trace(references=(), retain_values=False, retain_steps=None,
                           max_retained_snapshot_bytes=128 * 1024 * 1024):
    """Hash each policy tick and optionally compare detailed values as they stream."""
    groups = ('observations', 'actions', 'states', 'rewards')
    references = list(references)
    if retain_values and retain_steps is not None:
        raise ValueError('Choose full trace retention or selected-step retention, not both')
    if retain_steps is not None and references:
        raise ValueError('Selected-step retention cannot be used with streaming references')
    selected_steps = None
    if retain_steps is not None:
        selected_steps = {name: set(int(step) for step in retain_steps.get(name, ()))
                          for name in groups}
        if any(step < 1 for steps in selected_steps.values() for step in steps):
            raise ValueError('Selected trace steps must be one-based positive integers')
    trace = dict(observations=[], actions=[], states=[], rewards=[], comparisons={},
                 snapshots=({name: [] for name in groups} if retain_values else
                            {name: {} for name in groups} if selected_steps is not None else None),
                 retained_snapshot_tensor_bytes=0)
    original_observe, original_step = WorldBatch.observe, WorldBatch.step

    for ref_name, ref_trace in references:
        if ref_trace.get('snapshots') is None:
            raise ValueError(f'Trace reference {ref_name!r} has no retained values')
        trace['comparisons'][ref_name] = {
            name: dict(steps_compared=0, first_divergence=None) for name in groups
        }

    def record(group, values):
        current_digest = _tensor_digest(values)
        trace[group].append(current_digest)
        event_index = len(trace[group]) - 1
        step_number = event_index + 1
        frozen = None
        for ref_name, ref_trace in references:
            comparison = trace['comparisons'][ref_name][group]
            reference_events = ref_trace['snapshots'][group]
            if event_index >= len(reference_events):
                if comparison['first_divergence'] is None:
                    comparison['first_divergence'] = _divergence_record(
                        group, event_index + 1,
                        {'__event__': dict(exact_equal=False,
                            reason='reference_trace_ended', different_count=None,
                            max_abs_difference=None)})
                continue
            comparison['steps_compared'] += 1
            if current_digest == ref_trace[group][event_index]:
                continue
            if frozen is None:
                frozen = [(name, _freeze_trace_value(value)) for name, value in values]
            fields = _event_difference(reference_events[event_index], frozen)
            if comparison['first_divergence'] is None and any(
                    not field.get('bytewise_equal', field['exact_equal'])
                    for field in fields.values()):
                comparison['first_divergence'] = _divergence_record(
                    group, event_index + 1, fields)
        should_retain = (retain_values or
                         (selected_steps is not None and step_number in selected_steps[group]))
        if should_retain:
            event_bytes = _trace_event_tensor_bytes(values)
            if trace['retained_snapshot_tensor_bytes'] + event_bytes > max_retained_snapshot_bytes:
                raise MemoryError(
                    f'Trace snapshot tensor payload would exceed {max_retained_snapshot_bytes} bytes')
            if frozen is None:
                frozen = [(name, _freeze_trace_value(value)) for name, value in values]
            trace['retained_snapshot_tensor_bytes'] += event_bytes
            if retain_values:
                trace['snapshots'][group].append(frozen)
            else:
                trace['snapshots'][group][step_number] = frozen

    def observe(world):
        result = original_observe(world)
        record('observations', [('observation', result)])
        return result

    def step(world, actions):
        record('actions', [('actions', actions)])
        result = original_step(world, actions)
        state = [(name, getattr(world, name)) for name in _TRACE_STATE_FIELDS]
        state += [('arena', world.arena), ('hotspots', world.hotspots),
                  ('initial_mass', world.initial_mass), ('rng_state', world.rng.get_state()),
                  ('steps', world.steps), ('elapsed', world.elapsed),
                  ('shed_cursor', world.shed_cursor)]
        record('states', state)
        rewards = [('fitness', world.fitness())]
        rewards.extend((name, value) for name, value in world.fitness_terms().items())
        record('rewards', rewards)
        return result

    WorldBatch.observe, WorldBatch.step = observe, step
    try:
        yield trace
    finally:
        WorldBatch.observe, WorldBatch.step = original_observe, original_step


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def _record_sensor_candidates():
    """Capture per-observer capsule relevance counts without changing top-k."""
    original_topk = torch.topk
    samples = []

    def topk(input, k, *args, **kwargs):
        if input.ndim == 3 and input.dtype == torch.uint8:
            samples.append(input.sum(-1).detach().cpu())
        return original_topk(input, k, *args, **kwargs)

    torch.topk = topk
    try:
        yield samples
    finally:
        torch.topk = original_topk


def sensor_candidate_profile(world):
    with _record_sensor_candidates() as samples:
        world.observe()
    if not samples:
        return dict(minimum=0, median=0., maximum=0, distinct=0, nonzero=0)
    counts = torch.cat([sample.reshape(-1) for sample in samples])
    return dict(minimum=int(counts.min()), median=float(counts.float().median()),
                maximum=int(counts.max()), distinct=int(counts.unique().numel()),
                nonzero=int((counts > 0).sum()))


def _play(genome, neat_config, config, device, seed, seconds, trace=False,
          trace_references=(), retain_trace_values=False,
          retain_trace_steps=None, max_retained_snapshot_bytes=128 * 1024 * 1024,
          adaptive_work_budget_elements=None, collect_tile_stats=True,
          adaptive_tile_choices=(4, 8, 16)):
    focal = np.arange(config.maps, dtype=np.int64) * 7 % config.worms
    genomes = [genome] * config.maps
    tile_context = (adaptive_sensor_tiling(adaptive_work_budget_elements,
                                           choices=adaptive_tile_choices,
                                           collect_stats=collect_tile_stats)
                    if adaptive_work_budget_elements is not None
                    else contextlib.nullcontext(None))
    with tile_context as tile_policy:
        if trace:
            with _capture_episode_trace(
                    trace_references, retain_trace_values, retain_trace_steps,
                    max_retained_snapshot_bytes) as captured:
                metrics = evaluation.play_episode(
                    genomes, neat_config, config, device, seed, focal, seconds,
                    shared_random=False, policy='neat')
            _sync(device)
            return metrics, captured, (tile_policy.summary() if tile_policy is not None else None)

        _sync(device)
        started = time.perf_counter()
        metrics = evaluation.play_episode(
            genomes, neat_config, config, device, seed, focal, seconds,
            shared_random=False, policy='neat')
        _sync(device)
        return metrics, time.perf_counter() - started, (tile_policy.summary() if tile_policy is not None else None)


def _metric_summary(metrics):
    keys = ('fitness', 'alive', 'food_gain', 'boost_spent', 'kills',
            'border_death', 'collision_death')
    return {key: float(np.mean(metrics[key])) for key in keys}


def _trace_parity(left, right):
    fields = ('observations', 'actions', 'states', 'rewards')
    per_field = {
        name: len(left[name]) == len(right[name]) and left[name] == right[name]
        for name in fields
    }
    per_field['steps'] = len(left['observations']) == len(right['observations'])
    first_digest_divergence = {}
    for name in fields:
        for step_index, (left_digest, right_digest) in enumerate(zip(left[name], right[name])):
            if left_digest != right_digest:
                first_digest_divergence[name] = dict(step=step_index + 1)
                break
        else:
            if len(left[name]) != len(right[name]):
                first_digest_divergence[name] = dict(
                    step=min(len(left[name]), len(right[name])) + 1,
                    reason='trace_length_mismatch')
    return dict(
        equal=all(per_field.values()),
        steps_left=len(left['observations']),
        steps_right=len(right['observations']),
        fields=per_field,
        first_digest_divergence=first_digest_divergence,
        method='per-step SHA-256 of exact tensor bytes',
    )


def run_paired_policy_episode(config, genome, neat_config, seeds=(938271, 1038282),
                              seconds=15., device='cuda', chunks=(4, 16),
                              diagnostic_seeds=(), repeat_chunks=()):
    """Compare policy episodes; timing and traced parity runs are separate."""
    if len(chunks) < 2 or len(set(chunks)) != len(chunks) or any(chunk < 1 for chunk in chunks):
        raise ValueError('Provide at least two distinct positive sensor chunk values')
    if seconds <= 0 or not seeds:
        raise ValueError('Need positive episode seconds and at least one common seed')
    if not set(repeat_chunks).issubset(chunks):
        raise ValueError('Repeat chunks must be included in the requested chunk list')
    if repeat_chunks and not diagnostic_seeds:
        raise ValueError('Same-chunk repetitions require at least one diagnostic seed')
    if not set(diagnostic_seeds).issubset(seeds):
        raise ValueError('Diagnostic seeds must also be included in the timing seeds')
    ordered = sorted(chunks)
    diagnostic_seeds = set(int(seed) for seed in diagnostic_seeds)
    repeat_chunks = tuple(dict.fromkeys(repeat_chunks))
    timed, parity, repeats = [], [], []
    for seed_index, seed in enumerate(seeds):
        order = ordered if seed_index % 2 == 0 else list(reversed(ordered))
        traces = {}
        for chunk in order:
            c = dataclasses.replace(config, sensor_chunk=chunk).validate()
            metrics, elapsed, _ = _play(genome, neat_config, c, device, seed, seconds)
            row = dict(seed=int(seed), sensor_chunk=chunk, wall_ms=elapsed * 1000.,
                       raycast_blocks=sensor_ray_block_count(c, chunk),
                       requested_sim_seconds=seconds, metrics=_metric_summary(metrics))
            timed.append(row)
        reference_chunk = ordered[0]
        detailed = int(seed) in diagnostic_seeds
        for chunk in ordered:
            c = dataclasses.replace(config, sensor_chunk=chunk).validate()
            references = ()
            if detailed and chunk != reference_chunk:
                references = ((f'chunk_{reference_chunk}', traces[reference_chunk][1]),)
            retain = detailed and (chunk == reference_chunk or chunk in repeat_chunks)
            metrics, trace, _ = _play(
                genome, neat_config, c, device, seed, seconds, trace=True,
                trace_references=references, retain_trace_values=retain)
            traces[chunk] = (metrics, trace)
            if chunk == reference_chunk:
                continue
            reference_metrics, reference_trace = traces[reference_chunk]
            trace_result = _trace_parity(reference_trace, trace)
            trace_result['summary_metrics_equal'] = reference_metrics == metrics
            trace_result.update(seed=int(seed), left_chunk=reference_chunk, right_chunk=chunk)
            if detailed:
                trace_result['numeric_diagnostics'] = trace['comparisons'][f'chunk_{reference_chunk}']
            parity.append(trace_result)

        if detailed:
            for chunk in repeat_chunks:
                reference_metrics, reference_trace = traces[chunk]
                c = dataclasses.replace(config, sensor_chunk=chunk).validate()
                metrics, repeated_trace, _ = _play(
                    genome, neat_config, c, device, seed, seconds, trace=True,
                    trace_references=((f'chunk_{chunk}_first_run', reference_trace),))
                repeat_result = _trace_parity(reference_trace, repeated_trace)
                repeat_result['summary_metrics_equal'] = reference_metrics == metrics
                repeat_result['numeric_diagnostics'] = repeated_trace['comparisons'][
                    f'chunk_{chunk}_first_run']
                repeat_result.update(seed=int(seed), sensor_chunk=chunk,
                                     comparison='same_chunk_reproducibility')
                repeats.append(repeat_result)
                reference_trace['snapshots'] = None
            for _, trace in traces.values():
                trace['snapshots'] = None

    timing_summary = {}
    for chunk in ordered:
        samples = [row['wall_ms'] for row in timed if row['sensor_chunk'] == chunk]
        timing_summary[str(chunk)] = dict(median_wall_ms=statistics.median(samples),
                                           samples=len(samples))
    if 4 in ordered and 16 in ordered:
        timing_summary['ratio_chunk4_over_chunk16'] = (
            timing_summary['4']['median_wall_ms'] / timing_summary['16']['median_wall_ms']
        )

    return dict(
        mode='paired_policy_episode',
        device=str(device),
        maps=config.maps,
        worms=config.worms,
        seeds=[int(seed) for seed in seeds],
        seconds=seconds,
        focal_positions='map_index * 7 modulo worms (same for both chunks)',
        randomization='same episode seed per chunk; independent private maps',
        fixed_opponents='slitherai.evaluation.heuristic',
        timing='untraced play_episode wall time; includes world setup, policy, opponents, simulation and metrics',
        parity='separate instrumented play_episode; traces are excluded from timing',
        timed_episodes=timed,
        parity_episodes=parity,
        same_chunk_repeats=repeats,
        diagnostic_seeds=sorted(diagnostic_seeds),
        repeated_chunks=list(repeat_chunks),
        timing_summary=timing_summary,
        all_parity_equal=(all(row['equal'] and row['summary_metrics_equal'] for row in parity)
                          and all(row['equal'] and row['summary_metrics_equal'] for row in repeats)),
    )


def run_adaptive_tile_sweep(config, genome, neat_config, work_budgets,
                            seeds=(938271, 1038282), seconds=15., device='cuda',
                            baseline_chunk=4, choices=(4, 8, 16), warmup_seconds=None):
    """Compare an opt-in work-budget tile policy with fixed baseline chunk episodes."""
    budgets = tuple(int(value) for value in work_budgets)
    if not budgets or any(value < 1 for value in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError('Provide distinct positive work budgets')
    if seconds <= 0 or not seeds:
        raise ValueError('Need positive episode seconds and at least one common seed')
    if warmup_seconds is None:
        warmup_seconds = config.dt
    if warmup_seconds < 0:
        raise ValueError('warmup_seconds must be nonnegative')
    if (not choices or len(set(choices)) != len(choices) or any(chunk < 1 for chunk in choices)
            or baseline_chunk not in choices):
        raise ValueError('baseline_chunk must be one of the adaptive choices')

    fixed_config = dataclasses.replace(config, sensor_chunk=baseline_chunk).validate()
    timed = []
    parity = []
    for seed_index, seed in enumerate(seeds):
        conditions = [None, *budgets]
        if seed_index % 2:
            conditions.reverse()

        # Exercise each path before timing so the first CUDA context/kernel setup
        # does not become a chunk-specific timing penalty.
        if warmup_seconds:
            for budget in conditions:
                _play(genome, neat_config, fixed_config, device, seed, warmup_seconds,
                      adaptive_work_budget_elements=budget,
                      collect_tile_stats=False, adaptive_tile_choices=choices)

        metrics_by_condition = {}
        for budget in conditions:
            metrics, elapsed, _ = _play(
                genome, neat_config, fixed_config, device, seed, seconds,
                adaptive_work_budget_elements=budget,
                collect_tile_stats=False, adaptive_tile_choices=choices)
            metrics_by_condition[budget] = metrics
            timed.append(dict(seed=int(seed), sensor_chunk=(baseline_chunk if budget is None else 'adaptive'),
                              work_budget_elements=budget, wall_ms=elapsed * 1000.,
                              metrics=_metric_summary(metrics)))

        fixed_metrics, fixed_trace, _ = _play(
            genome, neat_config, fixed_config, device, seed, seconds, trace=True)
        if _metric_summary(fixed_metrics) != _metric_summary(metrics_by_condition[None]):
            raise RuntimeError('Fixed trace metrics differ from the timed fixed run')
        for budget in budgets:
            adaptive_metrics, adaptive_trace, tile_profile = _play(
                genome, neat_config, fixed_config, device, seed, seconds, trace=True,
                adaptive_work_budget_elements=budget, collect_tile_stats=True,
                adaptive_tile_choices=choices)
            comparison = _trace_parity(fixed_trace, adaptive_trace)
            comparison['summary_metrics_equal'] = _metric_summary(fixed_metrics) == _metric_summary(adaptive_metrics)
            comparison.update(seed=int(seed), baseline_chunk=baseline_chunk,
                              work_budget_elements=budget, tile_profile=tile_profile)
            parity.append(comparison)

    fixed_samples = [row['wall_ms'] for row in timed if row['work_budget_elements'] is None]
    adaptive_summary = {}
    for budget in budgets:
        samples = [row['wall_ms'] for row in timed if row['work_budget_elements'] == budget]
        adaptive_summary[str(budget)] = dict(
            median_wall_ms=statistics.median(samples), samples=len(samples),
            speedup_over_fixed=(statistics.median(fixed_samples) / statistics.median(samples)))

    return dict(
        mode='adaptive_candidate_work_tile_sweep',
        runtime_override='opt-in context only; no SimConfig/checkpoint setting changed',
        device=str(device),
        torch_version=torch.__version__,
        accelerator=(torch.cuda.get_device_name(torch.device(device))
                     if torch.device(device).type == 'cuda' and torch.cuda.is_available() else None),
        tile_policy=dict(name='candidate_work_budget_v1',
                         choices=list(sorted(set(choices))),
                         estimate='min(tile_maps,maps) * worms * rays * max_candidates',
                         selection='largest tile whose estimate is <= work_budget_elements; otherwise smallest tile'),
        config=dataclasses.asdict(config),
        baseline_chunk=baseline_chunk,
        work_budgets=list(budgets),
        seeds=[int(seed) for seed in seeds],
        seconds=seconds,
        warmup_seconds=warmup_seconds,
        timing='synchronized untraced full play_episode wall time; includes setup, policy, opponents, simulation and metrics',
        parity='separate SHA-256 per-step comparisons for observations, actions, full simulator state and rewards; no tensor snapshots retained',
        timed_episodes=timed,
        timing_summary=dict(fixed_chunk=dict(median_wall_ms=statistics.median(fixed_samples),
                                             samples=len(fixed_samples)),
                            adaptive=adaptive_summary),
        parity_episodes=parity,
        all_parity_equal=all(row['equal'] and row['summary_metrics_equal'] for row in parity),
    )


def run_tiling_trace_diagnostic(config, genome, neat_config, seed, seconds,
                                adaptive_work_budget, device='cuda',
                                snapshot_tensor_byte_cap=128 * 1024 * 1024):
    """Diagnose repeatability and adaptive-vs-fixed trace differences with bounded snapshots.

    Digest-only passes cover the complete short episode. Numeric tensors are captured only
    at the first digest-divergent event for each comparison and are released after comparing.
    """
    if seconds <= 0 or seconds > 15.:
        raise ValueError('Diagnostic trace horizon must be positive and at most 15 simulated seconds')
    if adaptive_work_budget < 1:
        raise ValueError('Adaptive work budget must be positive')
    if snapshot_tensor_byte_cap < 1:
        raise ValueError('Snapshot tensor byte cap must be positive')

    baseline_chunk, diagnostic_chunk = 4, 16
    conditions = {
        'fixed4_first': dict(chunk=baseline_chunk),
        'fixed4_repeat': dict(chunk=baseline_chunk),
        'fixed16_first': dict(chunk=diagnostic_chunk),
        'fixed16_repeat': dict(chunk=diagnostic_chunk),
        'adaptive': dict(chunk=baseline_chunk, work_budget=adaptive_work_budget),
    }
    traces, metric_summaries, tile_profiles = {}, {}, {}

    def play(label, retain_steps=None, max_snapshot_bytes=snapshot_tensor_byte_cap):
        condition = conditions[label]
        c = dataclasses.replace(config, sensor_chunk=condition['chunk']).validate()
        budget = condition.get('work_budget')
        metrics, trace, profile = _play(
            genome, neat_config, c, device, seed, seconds, trace=True,
            retain_trace_steps=retain_steps,
            max_retained_snapshot_bytes=max_snapshot_bytes,
            adaptive_work_budget_elements=budget,
            collect_tile_stats=(budget is not None))
        return metrics, trace, profile

    for label in conditions:
        metrics, trace, profile = play(label)
        traces[label] = trace
        metric_summaries[label] = _metric_summary(metrics)
        if profile is not None:
            tile_profiles[label] = profile

    pair_specs = (
        ('fixed4_first', 'fixed4_repeat', 'same_chunk_4_repeatability'),
        ('fixed16_first', 'fixed16_repeat', 'same_chunk_16_repeatability'),
        ('fixed4_first', 'adaptive', 'fixed4_vs_adaptive'),
    )
    comparisons = []
    peak_snapshot_payload = 0
    for left_label, right_label, comparison_name in pair_specs:
        left_trace, right_trace = traces[left_label], traces[right_label]
        comparison = _trace_parity(left_trace, right_trace)
        comparison.update(
            name=comparison_name, seed=int(seed), seconds=seconds,
            left=left_label, right=right_label,
            summary_metrics_equal=(metric_summaries[left_label] == metric_summaries[right_label]))
        target_steps = {
            group: int(detail['step'])
            for group, detail in comparison['first_digest_divergence'].items()
            if isinstance(detail.get('step'), int)
            and 1 <= detail['step'] <= min(comparison['steps_left'], comparison['steps_right'])
        }
        numeric = {}
        retained_pair_bytes = 0
        if target_steps:
            selected_steps = {group: {step} for group, step in target_steps.items()}
            _, left_detail, _ = play(left_label, retain_steps=selected_steps)
            left_bytes = left_detail['retained_snapshot_tensor_bytes']
            remaining_cap = snapshot_tensor_byte_cap - left_bytes
            if remaining_cap < 1:
                raise MemoryError('Reference snapshots exhausted the diagnostic tensor byte cap')
            _, right_detail, _ = play(right_label, retain_steps=selected_steps,
                                      max_snapshot_bytes=remaining_cap)
            right_bytes = right_detail['retained_snapshot_tensor_bytes']
            retained_pair_bytes = left_bytes + right_bytes
            peak_snapshot_payload = max(peak_snapshot_payload, retained_pair_bytes)
            for group, step in target_steps.items():
                left_event = left_detail['snapshots'][group].get(step)
                right_event = right_detail['snapshots'][group].get(step)
                if left_event is None or right_event is None:
                    numeric[group] = dict(step=step, first_field=None,
                                          replay_snapshot_missing=True)
                    continue
                fields = _event_difference(left_event, right_event)
                item = _divergence_record(group, step, fields)
                item.update(
                    left_replay_digest_matches_first_pass=(
                        left_detail[group][step - 1] == left_trace[group][step - 1]),
                    right_replay_digest_matches_first_pass=(
                        right_detail[group][step - 1] == right_trace[group][step - 1]))
                numeric[group] = item
            # Do not retain event tensors across comparisons.
            del left_detail, right_detail
        comparison['numeric_diagnostics'] = numeric
        comparison['retained_snapshot_tensor_bytes_for_pair'] = retained_pair_bytes
        comparisons.append(comparison)

    return dict(
        mode='bounded_tiling_trace_diagnostic',
        device=str(device), torch_version=torch.__version__,
        maps=config.maps, worms=config.worms, body_points=config.body_points,
        seed=int(seed), seconds=seconds,
        simulated_step_limit=math.ceil(seconds / config.dt),
        conditions={label: dict(sensor_chunk=condition['chunk'],
                                adaptive_work_budget=condition.get('work_budget'))
                    for label, condition in conditions.items()},
        adaptive_tile_profile=tile_profiles.get('adaptive'),
        snapshot_policy='digest-only full horizon; numeric tensor snapshots only at each pair first-difference step',
        snapshot_tensor_byte_cap=snapshot_tensor_byte_cap,
        peak_pair_snapshot_tensor_bytes=peak_snapshot_payload,
        comparisons=comparisons,
        all_parity_equal=all(row['equal'] and row['summary_metrics_equal']
                             for row in comparisons),
    )


def build_long_body_fixture(config, device='cpu', seed=917):
    """Dense, long-body maps with deaths and deliberately varied sight lines."""
    world = WorldBatch(config, device=device, seed=seed)
    m, w, p = config.maps, config.worms, config.body_points
    if w < 4 or p < 16:
        raise ValueError('Long-body fixture needs at least four worms and 16 body points')

    index = torch.arange(w, device=world.device, dtype=torch.float32)
    angles = index * (math.tau / w)
    headings = angles + .23 * torch.sin(index * .71)
    mass = 400. + index * (1000. / max(1, w - 1))
    for arena in range(m):
        mode = arena % 4
        if mode in (0, 2):
            radius = 220. if mode == 0 else 410.
            world.head[arena, :, 0] = radius * angles.cos()
            world.head[arena, :, 1] = radius * angles.sin()
        else:
            columns = torch.arange(w, device=world.device).remainder(4).float()
            rows = torch.arange(w, device=world.device).div(4, rounding_mode='floor').float()
            world.head[arena, :, 0] = (columns - 1.5) * 850.
            world.head[arena, :, 1] = (rows - max(0., (w - 1) / 8.)) * 700.
        world.heading[arena] = headings
        world.mass[arena] = mass * (1.0 + .1 * (arena % 3))
        if mode == 2:
            world.alive[arena, ::2] = False
        elif mode == 3:
            world.alive[arena, 0] = False

    direction = torch.stack((world.heading.cos(), world.heading.sin()), -1)
    distances = torch.arange(p, device=world.device, dtype=torch.float32)[None, None] * world.spacing[..., None]
    world.body.copy_(world.head[:, :, None] - direction[:, :, None] * distances[..., None])
    world.boost &= world.alive
    world.food_size = (world.food_mass * 3).clamp(2, 20)
    return world


_FIXTURE_STATE_FIELDS = _TRACE_STATE_FIELDS + ('arena', 'hotspots', 'initial_mass')


def _fixture_state_snapshot(world):
    return {name: getattr(world, name).detach().cpu().clone()
            for name in _FIXTURE_STATE_FIELDS}


def _fixture_state_matches(world, snapshot):
    return all(torch.equal(getattr(world, name).detach().cpu(), expected)
               for name, expected in snapshot.items())


def _fixture_observe(world, work_budget_elements=None, collect_tile_stats=False,
                     choices=(4, 8, 16)):
    if work_budget_elements is None:
        return world.observe(), None
    with adaptive_sensor_tiling(work_budget_elements, choices=choices,
                                collect_stats=collect_tile_stats) as stats:
        observation = world.observe()
    return observation, stats


def run_long_body_fixture(config, seed=917, chunks=(4, 16), device='cuda',
                          repeats=5, warmup_repeats=1, adaptive_work_budgets=(),
                          adaptive_choices=(4, 8, 16)):
    if not torch.cuda.is_available() and torch.device(device).type == 'cuda':
        raise RuntimeError('CUDA is required for the CUDA fixture')
    budgets = tuple(int(value) for value in adaptive_work_budgets)
    if repeats < 1 or warmup_repeats < 0:
        raise ValueError('Need positive repeats and nonnegative warmup repeats')
    if not chunks or len(set(chunks)) != len(chunks) or any(chunk < 1 for chunk in chunks):
        raise ValueError('Provide distinct positive fixed sensor chunks')
    if any(value < 1 for value in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError('Adaptive work budgets must be distinct positive integers')
    choices = tuple(sorted(set(int(chunk) for chunk in adaptive_choices)))
    if not choices or choices[0] < 1 or (budgets and chunks[0] not in choices):
        raise ValueError('Need positive adaptive choices including the baseline chunk')

    conditions = []
    for chunk in chunks:
        c = dataclasses.replace(config, sensor_chunk=chunk).validate()
        conditions.append(dict(kind='fixed', sensor_chunk=chunk, config=c,
                               key=('fixed', chunk)))
    for budget in budgets:
        c = dataclasses.replace(config, sensor_chunk=chunks[0]).validate()
        conditions.append(dict(kind='adaptive', work_budget_elements=budget,
                               sensor_chunk='adaptive', config=c,
                               key=('adaptive', budget)))

    reference_world = build_long_body_fixture(
        conditions[0]['config'], device=device, seed=seed)
    reference_state = _fixture_state_snapshot(reference_world)
    del reference_world
    _sync(device)

    def build_condition_world(condition):
        world = build_long_body_fixture(condition['config'], device=device, seed=seed)
        if not _fixture_state_matches(world, reference_state):
            raise RuntimeError('Fixture worlds do not have identical seeded initial state')
        return world

    def observe_condition(condition, world, collect_tile_stats=False):
        return _fixture_observe(
            world,
            work_budget_elements=condition.get('work_budget_elements'),
            collect_tile_stats=collect_tile_stats, choices=choices)

    # Use only one live fixture world at a time so the peak allocation reflects
    # that condition rather than all comparison worlds held on the device.
    # Alternate condition order each repeat to reduce clock and thermal bias.
    timed_samples = {condition['key']: [] for condition in conditions}
    peak_samples = {condition['key']: [] for condition in conditions}
    for repeat_index in range(repeats):
        order = conditions if repeat_index % 2 == 0 else list(reversed(conditions))
        for condition in order:
            world = build_condition_world(condition)
            if repeat_index == 0:
                for _ in range(warmup_repeats):
                    observe_condition(condition, world)
            _sync(device)
            if torch.device(device).type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            observation, _ = observe_condition(condition, world)
            _sync(device)
            timed_samples[condition['key']].append((time.perf_counter() - started) * 1000.)
            peak_samples[condition['key']].append(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if torch.device(device).type == 'cuda' else None)
            del observation, world
            _sync(device)

    reference = None
    results = []
    for condition in conditions:
        world = build_condition_world(condition)
        observation, tile_stats = observe_condition(
            condition, world, collect_tile_stats=condition['kind'] == 'adaptive')
        _sync(device)
        actual = observation.detach().cpu()
        if reference is None:
            reference = actual.clone()
        difference = (actual - reference).abs()
        candidate_profile = sensor_candidate_profile(world)
        samples = timed_samples[condition['key']]
        peak_mib_samples = [sample for sample in peak_samples[condition['key']] if sample is not None]
        row = dict(
            condition=condition['kind'],
            sensor_chunk=condition['sensor_chunk'],
            work_budget_elements=condition.get('work_budget_elements'),
            geometry_exact_to_reference=True,
            raycast_blocks=(sensor_ray_block_count(condition['config'], condition['sensor_chunk'])
                            if condition['kind'] == 'fixed' else None),
            observe_ms=statistics.median(samples),
            observe_samples_ms=samples,
            repeats=repeats,
            warmup_repeats=warmup_repeats,
            observation_exact_to_reference=torch.equal(actual, reference),
            observation_max_abs_difference=float(difference.max()) if difference.numel() else 0.,
            alive=int(world.alive.sum().item()),
            dead=int((~world.alive).sum().item()),
            active_body_points=world.active_body_points(),
            candidate_count_profile=candidate_profile,
            peak_allocated_mib=max(peak_mib_samples) if peak_mib_samples else None,
        )
        if tile_stats is not None:
            profile = tile_stats.summary()
            raycast_blocks_by_chunk = {
                chunk: sensor_ray_block_count(condition['config'], int(chunk))
                for chunk in profile['selected_chunk_counts']
            }
            row['tile_profile'] = profile
            row['raycast_blocks_by_chunk'] = raycast_blocks_by_chunk
            row['raycast_blocks'] = sum(
                count * raycast_blocks_by_chunk[str(chunk)]
                for chunk, count in profile['selected_chunk_counts'].items())
        results.append(row)
        del observation, actual, world
        _sync(device)

    fixed_summary = {}
    for chunk in chunks:
        row = next(item for item in results
                   if item['condition'] == 'fixed' and item['sensor_chunk'] == chunk)
        fixed_summary[str(chunk)] = dict(median_observe_ms=row['observe_ms'], samples=repeats)
    adaptive_summary = {
        str(budget): dict(median_observe_ms=next(
            row['observe_ms'] for row in results
            if row['condition'] == 'adaptive' and row['work_budget_elements'] == budget),
            samples=repeats)
        for budget in budgets
    }
    return dict(mode='long_body_death_crowding_fixture', device=str(device), seed=seed,
                config=dataclasses.asdict(config), reference_chunk=chunks[0],
                repeats=repeats, warmup_repeats=warmup_repeats,
                timing='median synchronized observe() time; condition order alternates every repeat; warmups and parity/profile passes excluded',
                fixed_timing_summary=fixed_summary, adaptive_timing_summary=adaptive_summary,
                results=results,
                all_geometry_equal=all(row['geometry_exact_to_reference'] for row in results),
                all_parity_equal=all(row['observation_exact_to_reference'] for row in results))


def load_run_champion(run_dir, model_payload=None):
    run = Path(run_dir)
    settings_path = run / 'settings.json'
    with settings_path.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    genome_path = Path(model_payload) if model_payload is not None else run / 'champion.pkl'
    with genome_path.open('rb') as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or 'genome' not in payload or 'config' not in payload:
        raise ValueError(f'Expected project genome payload in {genome_path}')
    if payload.get('schema') != VERSION:
        raise ValueError(f'Genome payload schema {payload.get("schema")!r} does not match {VERSION!r}')
    genome, neat_config = payload['genome'], payload['config']
    genome_config = getattr(neat_config, 'genome_config', None)
    if (genome_config is None or len(getattr(genome_config, 'input_keys', ())) != INPUTS
            or len(getattr(genome_config, 'output_keys', ())) != len(OUTPUTS)):
        raise ValueError(f'Genome payload must configure {INPUTS} inputs and {len(OUTPUTS)} outputs')
    if not hasattr(genome, 'key') or not hasattr(genome, 'nodes') or not hasattr(genome, 'connections'):
        raise ValueError(f'Genome payload in {genome_path} has an unsupported genome object')
    gene_data = dict(
        nodes=[dict(id=int(key), genes=dict(sorted(vars(gene).items())))
               for key, gene in sorted(genome.nodes.items())],
        connections=[dict(source=int(key[0]), target=int(key[1]),
                           genes=dict(sorted(vars(gene).items())))
                     for key, gene in sorted(genome.connections.items())])
    gene_fingerprint = hashlib.sha256(json.dumps(
        gene_data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return SimConfig.from_dict(settings['config']).validate(), payload['genome'], payload['config'], dict(
        run=str(run.resolve()), settings_file=str(settings_path.resolve()),
        settings_file_sha256=hashlib.sha256(settings_path.read_bytes()).hexdigest(),
        genome_file=str(genome_path.resolve()),
        genome_file_sha256=hashlib.sha256(genome_path.read_bytes()).hexdigest(),
        genome_generation=payload.get('generation'), genome_id=genome.key,
        genome_gene_sha256=gene_fingerprint,
        schema=payload.get('schema'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=int)
    parser.add_argument('--worms', type=int)
    parser.add_argument('--foods', type=int)
    parser.add_argument('--body-points', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--warmup-ticks', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--chunks', type=int, nargs='+')
    parser.add_argument('--run', type=Path, help='training run containing settings.json and champion.pkl')
    parser.add_argument('--model-payload', type=Path,
                        help='explicit saved genome payload to use with the run settings')
    parser.add_argument('--paired-policy', action='store_true',
                        help='run untraced policy timings and separate per-step parity traces')
    parser.add_argument('--seconds', type=float, default=15., help='simulated duration for each paired policy episode')
    parser.add_argument('--seeds', type=int, nargs='+', default=[938271, 1038282])
    parser.add_argument('--diagnostic-seeds', type=int, nargs='*', default=[],
                        help='seeds to trace in detail for first divergent step/field')
    parser.add_argument('--repeat-chunks', type=int, nargs='*', default=[],
                        help='repeat these chunks on diagnostic seeds to test same-chunk determinism')
    parser.add_argument('--adaptive-work-budgets', type=int, nargs='+',
                        help='opt-in sweep of candidate-work budgets; does not change saved settings')
    parser.add_argument('--diagnose-tiling-budget', type=int,
                        help='bounded first-divergence diagnostics for chunk 4/16 repeats and this adaptive budget')
    parser.add_argument('--fixture-repeats', type=int, default=5,
                        help='synchronized observation timing repeats for the long-body fixture')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cuda-long-fixture', action='store_true',
                        help='compare observation parity on crowded long bodies and deaths')
    parser.add_argument('--out', type=Path, help='optional JSON result path; defaults to run/analysis for policy jobs')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if (args.paired_policy or args.cuda_long_fixture or args.adaptive_work_budgets
            or args.diagnose_tiling_budget is not None):
        root = Path(__file__).resolve().parents[1]
        run = args.run
        if run is None:
            last = json.loads((root / 'runs' / 'last-run.json').read_text(encoding='utf-8'))
            run = root / 'runs' / last['name']
        config, genome, neat_config, source = load_run_champion(run, args.model_payload)
        if args.maps is not None:
            config = dataclasses.replace(config, maps=args.maps)
        if args.worms is not None:
            config = dataclasses.replace(config, worms=args.worms)
        if args.foods is not None:
            config = dataclasses.replace(config, foods=args.foods)
        if args.body_points is not None:
            config = dataclasses.replace(config, body_points=args.body_points)
        config.validate()
        chunks = tuple(args.chunks or (4, 8, 16))
        output = dict(mode='combined_policy_and_fixture', champion=source)
        if args.paired_policy:
            policy = run_paired_policy_episode(config, genome, neat_config,
                seeds=args.seeds, seconds=args.seconds, device=args.device, chunks=chunks,
                diagnostic_seeds=args.diagnostic_seeds, repeat_chunks=args.repeat_chunks)
            output['policy'] = policy
        if args.adaptive_work_budgets:
            output['adaptive_tile_sweep'] = run_adaptive_tile_sweep(
                config, genome, neat_config, work_budgets=args.adaptive_work_budgets,
                seeds=args.seeds, seconds=args.seconds, device=args.device)
        if args.diagnose_tiling_budget is not None:
            if len(args.seeds) != 1:
                raise ValueError('--diagnose-tiling-budget requires exactly one --seeds value')
            output['tiling_trace_diagnostic'] = run_tiling_trace_diagnostic(
                config, genome, neat_config, seed=args.seeds[0], seconds=args.seconds,
                adaptive_work_budget=args.diagnose_tiling_budget, device=args.device)
        if args.cuda_long_fixture:
            fixture_config = dataclasses.replace(
                config, maps=min(config.maps, 16), worms=min(config.worms, 8),
                foods=min(config.foods, 256), preys=0, body_points=max(config.body_points, 96))
            output['long_body_fixture'] = run_long_body_fixture(
                fixture_config, seed=args.seed, chunks=chunks, device=args.device,
                repeats=args.fixture_repeats,
                adaptive_work_budgets=args.adaptive_work_budgets or ())
        rendered = json.dumps(output, indent=2)
        out = args.out
        if out is None:
            out = run / 'analysis' / f'chunk-policy-{time.strftime("%Y%m%d-%H%M%S")}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + '\n', encoding='utf-8')
        print(rendered)
        print(f'Wrote JSON analysis to {out}')
        return

    config = SimConfig(maps=args.maps or 64, worms=args.worms or 16,
                       foods=args.foods or 1024, body_points=args.body_points or 96).validate()
    print(json.dumps(run_sweep(config, args.chunks or (4, 8, 16), args.seed,
                               args.warmup_ticks, args.repeats), indent=2))


if __name__ == '__main__':
    main()
