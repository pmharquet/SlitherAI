"""Profile synchronized components of one short SlitherAI policy episode.

This diagnostic loads a saved genome and run settings, never starts a trainer,
and writes its report only to stdout. Run it after any active training window.
Example::

    python -m scripts.profile_episode_components --run runs/my-run \
        --model-payload runs/my-run/best-validation.pkl --device cuda

Component timings synchronize before and after each operation, so they expose
device work plus host dispatch latency but serialize the measured phases. The
separate unprofiled total omits those per-phase barriers and is the episode
throughput estimate to compare across runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from slitherai.benchmark_observation import load_run_champion
from slitherai.evaluation import heuristic
from slitherai.network import BatchedNetwork
from slitherai.sim import WorldBatch


COMPONENTS = ('observe', 'heuristic', 'network_activate', 'world_step')


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _measure(fn, device):
    """Measure one operation as synchronized host wall time."""
    _synchronize(device)
    started = time.perf_counter()
    result = fn()
    _synchronize(device)
    return result, time.perf_counter() - started


def _candidate_slots(maps, worms, candidates_per_map):
    if not 1 <= candidates_per_map <= worms:
        raise ValueError('candidates_per_map must be between 1 and worms')
    return np.arange(maps, dtype=np.int64)[:, None] * worms + np.arange(
        candidates_per_map, dtype=np.int64)[None, :]


@torch.inference_mode()
def run_episode(genome, neat_config, config, *, seed, seconds, device,
                candidates_per_map, profile_components):
    """Run a fresh deterministic episode and optionally time each component."""
    config.validate()
    device = torch.device(device)
    world = WorldBatch(config, device, seed, shared_random=True)
    slots = config.maps * config.worms
    candidates = _candidate_slots(config.maps, config.worms, candidates_per_map).reshape(-1)
    heuristic_slots = np.setdiff1d(np.arange(slots, dtype=np.int64), candidates,
                                   assume_unique=True)
    candidate_world_slots = torch.as_tensor(candidates, dtype=torch.long, device=device)
    heuristic_world_slots = torch.as_tensor(heuristic_slots, dtype=torch.long, device=device)
    network = BatchedNetwork([genome], neat_config,
                             assignment=[0] * len(candidates), device=device)
    steps_limit = round(float(seconds) / config.dt)
    if steps_limit < 1:
        raise ValueError('seconds must include at least one simulation step')
    timings = {name: 0. for name in COMPONENTS}
    steps = 0

    _synchronize(device)
    episode_started = time.perf_counter()
    for _ in range(steps_limit):
        def observe():
            return world.observe().reshape(slots, -1)

        if profile_components:
            observations, elapsed = _measure(observe, device)
            timings['observe'] += elapsed
        else:
            observations = observe()

        actions = torch.empty((slots, 2), dtype=observations.dtype, device=device)
        if profile_components:
            def choose_heuristic():
                if len(heuristic_slots):
                    actions[heuristic_world_slots] = heuristic(observations[heuristic_world_slots])
            _, elapsed = _measure(choose_heuristic, device)
            timings['heuristic'] += elapsed
        elif len(heuristic_slots):
            actions[heuristic_world_slots] = heuristic(observations[heuristic_world_slots])

        if profile_components:
            def activate_network():
                actions[candidate_world_slots] = network.activate(
                    observations[candidate_world_slots])
            _, elapsed = _measure(activate_network, device)
            timings['network_activate'] += elapsed
        else:
            actions[candidate_world_slots] = network.activate(
                observations[candidate_world_slots])

        if profile_components:
            _, elapsed = _measure(
                lambda: world.step(actions.reshape(config.maps, config.worms, 2)), device)
            timings['world_step'] += elapsed
        else:
            world.step(actions.reshape(config.maps, config.worms, 2))

        steps += 1
        if not bool(world.alive.reshape(-1)[candidate_world_slots].any()):
            break

    _synchronize(device)
    total_seconds = time.perf_counter() - episode_started
    candidate_fitness = world.fitness().reshape(-1)[candidate_world_slots]
    result = dict(
        seed=int(seed), simulated_seconds=round(world.elapsed, 6), steps=steps,
        total_synchronized_seconds=total_seconds,
        candidate_alive=int(world.alive.reshape(-1)[candidate_world_slots].sum().item()),
        candidate_fitness_mean=float(candidate_fitness.mean().item()),
    )
    if profile_components:
        result['component_seconds'] = timings
        result['component_sum_seconds'] = sum(timings.values())
    if device.type == 'cuda':
        result['cuda_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 1e6
        result['cuda_peak_reserved_mb'] = torch.cuda.max_memory_reserved(device) / 1e6
    else:
        result['cuda_peak_allocated_mb'] = None
        result['cuda_peak_reserved_mb'] = None
    return result


def profile_episode(genome, neat_config, config, *, seed=123457, seconds=2.,
                    device='cuda', candidates_per_map=1, warmup_episodes=1,
                    repeats=3):
    """Warm kernels, collect synchronized phase samples and unprofiled totals."""
    if warmup_episodes < 0 or repeats < 1:
        raise ValueError('warmup_episodes must be nonnegative and repeats positive')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    warmup_started = time.perf_counter()
    for _ in range(warmup_episodes):
        run_episode(genome, neat_config, config, seed=seed, seconds=seconds,
                    device=device, candidates_per_map=candidates_per_map,
                    profile_components=False)
    _synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    samples = []
    for repeat in range(repeats):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        profiled = run_episode(
            genome, neat_config, config, seed=seed, seconds=seconds,
            device=device, candidates_per_map=candidates_per_map,
            profile_components=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        unprofiled = run_episode(
            genome, neat_config, config, seed=seed, seconds=seconds,
            device=device, candidates_per_map=candidates_per_map,
            profile_components=False)
        samples.append(dict(repeat=repeat, profiled=profiled, unprofiled=unprofiled))

    medians = {
        name: statistics.median(row['profiled']['component_seconds'][name]
                                for row in samples)
        for name in COMPONENTS
    }
    return dict(
        timing_method=(
            'synchronized perf_counter wall time before/after each component; '
            'phase barriers serialize profiled episode; total uses a separate unprofiled pass'),
        device=str(device), seed=int(seed), seconds=float(seconds), repeats=int(repeats),
        warmup_episodes=int(warmup_episodes), warmup_wall_seconds=warmup_seconds,
        config=dataclasses.asdict(config), candidates_per_map=int(candidates_per_map),
        component_median_seconds=medians,
        component_median_ms_per_tick={
            name: 1000 * statistics.median(
                row['profiled']['component_seconds'][name] / row['profiled']['steps']
                for row in samples)
            for name in COMPONENTS
        },
        profiled_total_median_seconds=statistics.median(
            row['profiled']['total_synchronized_seconds'] for row in samples),
        profiled_component_sum_median_seconds=statistics.median(
            row['profiled']['component_sum_seconds'] for row in samples),
        unprofiled_total_median_seconds=statistics.median(
            row['unprofiled']['total_synchronized_seconds'] for row in samples),
        samples=samples,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True,
                        help='run directory containing settings.json')
    parser.add_argument('--model-payload', type=Path,
                        help='local saved genome payload; defaults to run/champion.pkl')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--maps', type=int,
                        help='override maps for this diagnostic only; defaults to run settings')
    parser.add_argument('--candidates-per-map', type=int,
                        help='default comes from the saved protocol, or one for reference runs')
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=123457)
    parser.add_argument('--warmup-episodes', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)

    config, genome, neat_config, source = load_run_champion(
        args.run, model_payload=args.model_payload)
    settings = json.loads((args.run / 'settings.json').read_text(encoding='utf-8'))
    if args.maps is not None:
        if args.maps < 1:
            parser.error('--maps must be positive')
        config = dataclasses.replace(config, maps=args.maps).validate()
    saved_protocol = settings.get('protocol', {})
    saved_mode = settings.get('opponent_mode', saved_protocol.get('opponent_mode', 'reference'))
    default_candidates = saved_protocol.get(
        'candidate_slots_per_map', config.worms if saved_mode == 'selfplay' else 1)
    candidates_per_map = (default_candidates if args.candidates_per_map is None
                          else args.candidates_per_map)
    result = profile_episode(
        genome, neat_config, config, seed=args.seed, seconds=args.seconds,
        device=args.device, candidates_per_map=candidates_per_map,
        warmup_episodes=args.warmup_episodes, repeats=args.repeats)
    result['source'] = source
    result['genome_id'] = genome.key
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
