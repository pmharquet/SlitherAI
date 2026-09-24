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
from .sim import WorldBatch


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


@contextlib.contextmanager
def _capture_episode_trace():
    """Hash each production observation, action, post-step state and reward."""
    trace = dict(observations=[], actions=[], states=[], rewards=[])
    original_observe, original_step = WorldBatch.observe, WorldBatch.step

    def observe(world):
        result = original_observe(world)
        trace['observations'].append(_tensor_digest([('observation', result)]))
        return result

    def step(world, actions):
        trace['actions'].append(_tensor_digest([('actions', actions)]))
        result = original_step(world, actions)
        state = [(name, getattr(world, name)) for name in _TRACE_STATE_FIELDS]
        state += [('arena', world.arena), ('hotspots', world.hotspots),
                  ('initial_mass', world.initial_mass), ('rng_state', world.rng.get_state()),
                  ('steps', world.steps), ('elapsed', world.elapsed),
                  ('shed_cursor', world.shed_cursor)]
        trace['states'].append(_tensor_digest(state))
        rewards = [('fitness', world.fitness())]
        rewards.extend((name, value) for name, value in world.fitness_terms().items())
        trace['rewards'].append(_tensor_digest(rewards))
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


def _play(genome, neat_config, config, device, seed, seconds, trace=False):
    focal = np.arange(config.maps, dtype=np.int64) * 7 % config.worms
    genomes = [genome] * config.maps
    if trace:
        with _capture_episode_trace() as captured:
            metrics = evaluation.play_episode(
                genomes, neat_config, config, device, seed, focal, seconds,
                shared_random=False, policy='neat')
        _sync(device)
        return metrics, captured

    _sync(device)
    started = time.perf_counter()
    metrics = evaluation.play_episode(
        genomes, neat_config, config, device, seed, focal, seconds,
        shared_random=False, policy='neat')
    _sync(device)
    return metrics, time.perf_counter() - started


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
    return dict(
        equal=all(per_field.values()),
        steps_left=len(left['observations']),
        steps_right=len(right['observations']),
        fields=per_field,
        method='per-step SHA-256 of exact tensor bytes',
    )


def run_paired_policy_episode(config, genome, neat_config, seeds=(938271, 1038282),
                              seconds=15., device='cuda', chunks=(4, 16)):
    """Compare policy episodes; timing and traced parity runs are separate."""
    if len(chunks) != 2 or chunks[0] == chunks[1]:
        raise ValueError('Provide exactly two distinct sensor chunk values')
    if seconds <= 0 or not seeds:
        raise ValueError('Need positive episode seconds and at least one common seed')
    ordered = sorted(chunks)
    timed, parity = [], []
    by_seed_chunk = {}
    for seed_index, seed in enumerate(seeds):
        order = ordered if seed_index % 2 == 0 else list(reversed(ordered))
        for chunk in order:
            c = dataclasses.replace(config, sensor_chunk=chunk).validate()
            metrics, elapsed = _play(genome, neat_config, c, device, seed, seconds)
            row = dict(seed=int(seed), sensor_chunk=chunk, wall_ms=elapsed * 1000.,
                       raycast_blocks=sensor_ray_block_count(c, chunk),
                       requested_sim_seconds=seconds, metrics=_metric_summary(metrics))
            timed.append(row)
            by_seed_chunk[(int(seed), chunk)] = metrics

        left_chunk, right_chunk = ordered
        c_left = dataclasses.replace(config, sensor_chunk=left_chunk).validate()
        c_right = dataclasses.replace(config, sensor_chunk=right_chunk).validate()
        left_metrics, left_trace = _play(genome, neat_config, c_left, device, seed, seconds, trace=True)
        right_metrics, right_trace = _play(genome, neat_config, c_right, device, seed, seconds, trace=True)
        trace_result = _trace_parity(left_trace, right_trace)
        trace_result['summary_metrics_equal'] = left_metrics == right_metrics
        trace_result.update(seed=int(seed), left_chunk=left_chunk, right_chunk=right_chunk)
        parity.append(trace_result)

    timing_summary = {}
    for chunk in ordered:
        samples = [row['wall_ms'] for row in timed if row['sensor_chunk'] == chunk]
        timing_summary[str(chunk)] = dict(median_wall_ms=statistics.median(samples),
                                           samples=len(samples))
    if len(ordered) == 2:
        timing_summary['ratio_chunk4_over_chunk16'] = (
            timing_summary['4']['median_wall_ms'] / timing_summary['16']['median_wall_ms']
            if ordered == [4, 16] else None)

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
        timing_summary=timing_summary,
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
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cuda-long-fixture', action='store_true',
                        help='compare observation parity on crowded long bodies and deaths')
    parser.add_argument('--out', type=Path, help='optional JSON result path; defaults to run/analysis for policy jobs')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.paired_policy or args.cuda_long_fixture:
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
        chunks = tuple(args.chunks or (4, 16))
        output = dict(mode='combined_policy_and_fixture', champion=source)
        if args.paired_policy:
            policy = run_paired_policy_episode(config, genome, neat_config,
                seeds=args.seeds, seconds=args.seconds, device=args.device, chunks=chunks)
            output['policy'] = policy
        if args.cuda_long_fixture:
            fixture_config = dataclasses.replace(
                config, maps=min(config.maps, 16), worms=min(config.worms, 8),
                foods=min(config.foods, 256), preys=0, body_points=max(config.body_points, 96))
            output['long_body_fixture'] = run_long_body_fixture(
                fixture_config, seed=args.seed, chunks=(4, 16), device=args.device)
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
