"""Small repeatable CUDA sweep for the observation sensor chunk size.

This is an offline benchmark helper. It builds deterministic worlds and never
changes the trainer's saved configuration or policy.
"""
import argparse
import dataclasses
import json
import math
import statistics
import time

import torch

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=int, default=64)
    parser.add_argument('--worms', type=int, default=16)
    parser.add_argument('--foods', type=int, default=1024)
    parser.add_argument('--body-points', type=int, default=96)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--warmup-ticks', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--chunks', type=int, nargs='+', default=[4, 8, 16])
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = SimConfig(maps=args.maps, worms=args.worms, foods=args.foods,
                       body_points=args.body_points).validate()
    print(json.dumps(run_sweep(config, args.chunks, args.seed,
                               args.warmup_ticks, args.repeats), indent=2))


if __name__ == '__main__':
    main()
