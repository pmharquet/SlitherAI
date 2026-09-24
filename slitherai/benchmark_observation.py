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
                    left_dtype=str(a.dtype), right_dtype=str(b.dtype),
                    different_count=None, max_abs_difference=None)
    equal = torch.equal(a, b)
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
    return dict(exact_equal=equal, shape=list(a.shape), dtype=str(a.dtype),
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
    different = [name for name, value in fields.items() if not value['exact_equal']]
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


@contextlib.contextmanager
def _capture_episode_trace(references=(), retain_values=False):
    """Hash each policy tick and optionally compare detailed values as they stream."""
    groups = ('observations', 'actions', 'states', 'rewards')
    trace = dict(observations=[], actions=[], states=[], rewards=[], comparisons={},
                 snapshots={name: [] for name in groups} if retain_values else None)
    original_observe, original_step = WorldBatch.observe, WorldBatch.step

    references = list(references)
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
                    not field['exact_equal'] for field in fields.values()):
                comparison['first_divergence'] = _divergence_record(
                    group, event_index + 1, fields)
        if retain_values:
            if frozen is None:
                frozen = [(name, _freeze_trace_value(value)) for name, value in values]
            trace['snapshots'][group].append(frozen)

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
            with _capture_episode_trace(trace_references, retain_trace_values) as captured:
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


def run_long_body_fixture(config, seed=917, chunks=(4, 16), device='cuda'):
    if not torch.cuda.is_available() and torch.device(device).type == 'cuda':
        raise RuntimeError('CUDA is required for the CUDA fixture')
    results, reference = [], None
    for chunk in chunks:
        c = dataclasses.replace(config, sensor_chunk=chunk).validate()
        world = build_long_body_fixture(c, device=device, seed=seed)
        _sync(device)
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        observation = world.observe()
        _sync(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.
        if reference is None:
            reference = observation.detach().cpu().clone()
        actual = observation.detach().cpu()
        peak_mib = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if torch.device(device).type == 'cuda' else None)
        candidate_profile = sensor_candidate_profile(world)
        results.append(dict(
            sensor_chunk=chunk,
            raycast_blocks=sensor_ray_block_count(c, chunk),
            observe_ms=elapsed_ms,
            observation_exact_to_reference=torch.equal(actual, reference),
            observation_max_abs_difference=float((actual - reference).abs().max()),
            alive=int(world.alive.sum().item()),
            dead=int((~world.alive).sum().item()),
            active_body_points=world.active_body_points(),
            candidate_count_profile=candidate_profile,
            peak_allocated_mib=peak_mib,
        ))
        del world, observation, actual
        _sync(device)
    return dict(mode='long_body_death_crowding_fixture', device=str(device), seed=seed,
                config=dataclasses.asdict(config), results=results,
                all_parity_equal=all(row['observation_exact_to_reference'] for row in results))


def load_run_champion(run_dir):
    run = Path(run_dir)
    with (run / 'settings.json').open(encoding='utf-8') as stream:
        settings = json.load(stream)
    with (run / 'champion.pkl').open('rb') as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or 'genome' not in payload or 'config' not in payload:
        raise ValueError(f'Expected project genome payload in {run / "champion.pkl"}')
    return SimConfig.from_dict(settings['config']).validate(), payload['genome'], payload['config'], dict(
        run=str(run.resolve()), genome_file=str((run / 'champion.pkl').resolve()),
        genome_generation=payload.get('generation'), genome_id=payload['genome'].key,
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
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cuda-long-fixture', action='store_true',
                        help='compare observation parity on crowded long bodies and deaths')
    parser.add_argument('--out', type=Path, help='optional JSON result path; defaults to run/analysis for policy jobs')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.paired_policy or args.cuda_long_fixture or args.adaptive_work_budgets:
        root = Path(__file__).resolve().parents[1]
        run = args.run
        if run is None:
            last = json.loads((root / 'runs' / 'last-run.json').read_text(encoding='utf-8'))
            run = root / 'runs' / last['name']
        config, genome, neat_config, source = load_run_champion(run)
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
        if args.cuda_long_fixture:
            fixture_config = dataclasses.replace(
                config, maps=min(config.maps, 16), worms=min(config.worms, 8),
                foods=min(config.foods, 256), preys=0, body_points=max(config.body_points, 96))
            output['long_body_fixture'] = run_long_body_fixture(
                fixture_config, seed=args.seed, chunks=chunks, device=args.device)
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
