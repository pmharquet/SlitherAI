"""Replay saved genomes and summarize focal-worm behavior near arena borders.

This read-only diagnostic reuses the project's saved payload loader,
WorldBatch, BatchedNetwork, and heuristic opponents. It does not start a trainer
or modify run artifacts. By default it replays the fixed validation seed and
focal-seat schedule; reduce maps and seconds for a quick smoke run.

Example::

    python -m scripts.diagnose_border_behavior \
      --candidate runs/run-a runs/run-a/best-validation.pkl \
      --candidate runs/run-b runs/run-b/best-validation.pkl \
      --device cuda --output border-diagnostic.json

Quick CPU smoke run::

    python -m scripts.diagnose_border_behavior \
      --candidate runs/run-a runs/run-a/best-validation.pkl \
      --maps 2 --seconds 0.2 --device cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from slitherai.benchmark_observation import load_run_champion
from slitherai.evaluation import heuristic
from slitherai.network import BatchedNetwork
from slitherai.sim import WorldBatch


VALIDATION_SEED = 938271
BORDER_FRACTIONS = (0.05, 0.10, 0.20)


@torch.inference_mode()
def replay(genome, neat_config, config, *, seed, seconds, device='cpu'):
    """Replay the standard focal-versus-heuristic game and collect tick stats."""
    config.validate()
    maps, worms = config.maps, config.worms
    device = torch.device(device)
    world = WorldBatch(config, device=device, seed=seed)
    rows = torch.arange(maps, device=device)
    focal_np = np.arange(maps, dtype=np.int64) * 7 % worms
    focal = torch.as_tensor(focal_np, dtype=torch.long, device=device)
    network = BatchedNetwork([genome] * maps, neat_config, device=device)
    limits = round(float(seconds) / config.dt)
    if limits < 1:
        raise ValueError('seconds must include at least one simulation tick')

    ticks = 0
    alive_ticks = 0
    near_ticks = {str(fraction): 0 for fraction in BORDER_FRACTIONS}
    total_possible_ticks = 0
    near_food = {str(fraction): 0.0 for fraction in BORDER_FRACTIONS}
    far_food = {str(fraction): 0.0 for fraction in BORDER_FRACTIONS}
    abs_turn_radians = 0.0
    signed_turn_radians = 0.0
    turn_observations = 0
    alive_clearance_sum = 0.0
    alive_clearance_count = 0

    for _ in range(limits):
        observations = world.observe().reshape(maps, worms, -1)
        inputs = observations[rows, focal]
        actions = heuristic(observations.flatten(0, 1)).reshape(maps, worms, 2)
        actions[rows, focal] = network.activate(inputs)

        was_alive = world.alive[rows, focal].clone()
        old_heading = world.heading[rows, focal].clone()
        old_gained = world.gained[rows, focal].clone()
        clearance = (world.arena - world.head[rows, focal].norm(dim=-1)
                     - world.radius[rows, focal]).clamp_min(0)
        arena_radius = world.arena.clamp_min(1)
        relative_clearance = clearance / arena_radius
        total_possible_ticks += maps
        alive_ticks += int(was_alive.sum().item())
        if bool(was_alive.any()):
            alive_clearance_sum += float(relative_clearance[was_alive].sum().item())
            alive_clearance_count += int(was_alive.sum().item())
        for fraction in BORDER_FRACTIONS:
            near = was_alive & (relative_clearance < fraction)
            near_ticks[str(fraction)] += int(near.sum().item())

        world.step(actions)
        gained_now = (world.gained[rows, focal] - old_gained).clamp_min(0)
        for fraction in BORDER_FRACTIONS:
            near = was_alive & (relative_clearance < fraction)
            near_food[str(fraction)] += float(gained_now[near].sum().item())
            far_food[str(fraction)] += float(gained_now[was_alive & ~near].sum().item())

        delta = torch.atan2((world.heading[rows, focal] - old_heading).sin(),
                            (world.heading[rows, focal] - old_heading).cos())
        turns = delta[was_alive]
        abs_turn_radians += float(turns.abs().sum().item())
        signed_turn_radians += float(turns.sum().item())
        turn_observations += int(turns.numel())
        ticks += 1
        if not bool(world.alive[rows, focal].any()):
            break

    fitness = world.fitness()[rows, focal]
    return {
        'ticks': ticks,
        'simulated_seconds': round(world.elapsed, 6),
        'alive_maps': int(world.alive[rows, focal].sum().item()),
        'survival_fraction': float(world.alive[rows, focal].float().mean().item()),
        'fitness_mean': float(fitness.mean().item()),
        'fitness_by_map': [float(v) for v in fitness.detach().cpu().tolist()],
        'food_gain_by_map': [float(v) for v in world.gained[rows, focal].detach().cpu().tolist()],
        'food_gain_mean': float(world.gained[rows, focal].mean().item()),
        'border_deaths': int(world.border_deaths[rows, focal].sum().item()),
        'collision_deaths': int(world.collision_deaths[rows, focal].sum().item()),
        'alive_tick_fraction': (alive_ticks / total_possible_ticks
                                if total_possible_ticks else 0.0),
        'border': {
            f'within_{int(fraction * 100)}pct_arena_radius': {
                'alive_tick_fraction': (near_ticks[str(fraction)] / alive_ticks
                                        if alive_ticks else 0.0),
                'alive_ticks': near_ticks[str(fraction)],
                'food_gained_near': near_food[str(fraction)],
                'food_gained_far': far_food[str(fraction)],
            }
            for fraction in BORDER_FRACTIONS
        },
        'mean_clearance_fraction_while_alive': (
            alive_clearance_sum / alive_clearance_count if alive_clearance_count else None),
        'mean_absolute_heading_change_degrees_per_alive_tick': (
            math.degrees(abs_turn_radians / turn_observations)
            if turn_observations else None),
        'mean_signed_heading_change_degrees_per_alive_tick': (
            math.degrees(signed_turn_radians / turn_observations)
            if turn_observations else None),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', nargs=2, action='append', required=True,
                        metavar=('RUN_DIR', 'MODEL_PAYLOAD'),
                        help='repeat once per candidate; each run supplies settings.json')
    parser.add_argument('--maps', type=int, default=32,
                        help='number of fixed-seed validation maps (default: 32)')
    parser.add_argument('--seconds', type=float, default=90.,
                        help='replay duration, as in fixed validation (default: 90)')
    parser.add_argument('--seed', type=int, default=VALIDATION_SEED,
                        help=f'validation seed (default: {VALIDATION_SEED})')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='execution device (default: cpu)')
    parser.add_argument('--output', type=Path,
                        help='optional JSON output path; stdout is always written')
    args = parser.parse_args(argv)
    if not 1 <= len(args.candidate) <= 2:
        parser.error('provide one or two --candidate pairs')
    if args.maps < 1 or args.seconds <= 0:
        parser.error('--maps and --seconds must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda requested but CUDA is unavailable')

    candidates = []
    shared_config = None
    for run_arg, payload_arg in args.candidate:
        run, payload_path = Path(run_arg), Path(payload_arg)
        config, genome, neat_config, source = load_run_champion(run, payload_path)
        config = dataclasses.replace(config, maps=args.maps).validate()
        config_identity = dataclasses.asdict(dataclasses.replace(config, maps=1))
        if shared_config is None:
            shared_config = config_identity
        elif config_identity != shared_config:
            raise ValueError('Candidates must have identical saved simulation settings to replay the same maps')
        result = replay(genome, neat_config, config, seed=args.seed,
                        seconds=args.seconds, device=args.device)
        result.update({
            'run_settings': source['settings_file'],
            'run_settings_sha256': source['settings_file_sha256'],
            'model_payload': source['genome_file'],
            'model_payload_sha256': source['genome_file_sha256'],
            'genome_id': source['genome_id'],
            'genome_generation': source['genome_generation'],
            'genome_gene_sha256': source['genome_gene_sha256'],
            'sensor_version': source['sensor_version'],
        })
        candidates.append(result)

    report = {
        'diagnostic': 'border_behavior_replay_v1',
        'device': args.device,
        'seed': args.seed,
        'maps': args.maps,
        'seconds': args.seconds,
        'focal_seat_schedule': 'map_index * 7 % worms (same as fixed_validation)',
        'opponents': 'existing slitherai.evaluation.heuristic',
        'border_distance_definition': 'arena_radius - head_radius_from_center - worm_radius; clipped at zero',
        'near_food_attribution': 'food gained on tick assigned using focal position before that tick',
        'heading_change': 'wrapped heading delta after each world.step; averaged over worms alive before tick',
        'config': shared_config,
        'candidates': candidates,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
