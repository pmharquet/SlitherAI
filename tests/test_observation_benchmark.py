import dataclasses
import os
import random

import neat
import pytest
import torch

from slitherai.benchmark_observation import (
    build_long_body_fixture, run_long_body_fixture, run_paired_policy_episode,
    sensor_candidate_profile, sensor_ray_block_count,
)
from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.sim import WorldBatch


def _long_mixed_world(sensor_chunk):
    config = SimConfig(maps=2, worms=4, foods=32, preys=0, body_points=16,
                       arena_radius=1200, sensor_chunk=sensor_chunk)
    world = WorldBatch(config, 'cpu', seed=71)
    world.head[:] = torch.tensor([
        [[0., 0.], [150., 0.], [460., 300.], [900., -200.]],
        [[-800., 0.], [20., 40.], [700., 100.], [0., 900.]],
    ])
    world.heading[:] = torch.tensor([[0., .2, -.3, .5], [-.1, .4, -.6, .8]])
    world.mass[:] = torch.tensor([[450., 180., 35., 900.], [600., 35., 150., 35.]])
    world.alive[0, 3] = False
    world.alive[1, 0] = False
    direction = torch.stack((world.heading.cos(), world.heading.sin()), -1)
    distances = torch.arange(config.body_points, dtype=torch.float32)[None, None] * world.spacing[..., None]
    world.body[:] = world.head[:, :, None] - direction[:, :, None] * distances[..., None]

    food_points = torch.tensor([[50., 0.], [-150., 30.], [400., 200.], [900., -200.],
                                [-800., 0.], [0., 450.]])
    world.food[:, :len(food_points)] = food_points
    world.food_mass[:, :len(food_points)] = torch.tensor([1., 2., 3., 1.5, 2.5, .5])
    world.food_size = (world.food_mass * 3).clamp(2, 20)
    return world


def test_sensor_chunk_preserves_long_body_and_death_observations():
    worlds = [_long_mixed_world(chunk) for chunk in (1, 2, 4)]
    observations = [world.observe() for world in worlds]

    assert worlds[0].active_body_points() == worlds[1].active_body_points() == worlds[2].active_body_points() == 15
    assert int(worlds[0].alive.sum()) == 6
    assert all(torch.isfinite(observation).all() for observation in observations)
    assert observations[0].shape == (8, 530)
    assert torch.equal(observations[0], observations[1])
    assert torch.equal(observations[0], observations[2])


def test_paired_policy_episode_has_per_step_observation_action_reward_and_state_parity():
    random.seed(71)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    config = SimConfig(maps=5, worms=2, foods=32, preys=0, body_points=16,
                       arena_radius=1200)

    result = run_paired_policy_episode(config, genome, neat_config,
        seeds=(71,), seconds=.3, device='cpu', chunks=(4, 16))

    parity = result['parity_episodes'][0]
    assert parity['equal'] and parity['summary_metrics_equal']
    assert parity['steps_left'] == parity['steps_right'] == 3
    assert all(parity['fields'][field] for field in ('observations', 'actions', 'states', 'rewards'))
    assert result['all_parity_equal']
    assert {row['sensor_chunk']: row['raycast_blocks'] for row in result['timed_episodes']} == {4: 2, 16: 1}
    assert result['timing_summary']['4']['samples'] == result['timing_summary']['16']['samples'] == 1


def test_cpu_long_body_death_and_crowding_fixture_is_exact_across_chunks():
    config = SimConfig(maps=5, worms=8, foods=128, preys=0, body_points=96,
                       arena_radius=2400)
    worlds = [build_long_body_fixture(
        dataclasses.replace(config, sensor_chunk=chunk), 'cpu', seed=917)
        for chunk in (4, 16)]
    observations = [world.observe() for world in worlds]

    assert worlds[0].active_body_points() == worlds[1].active_body_points() == 95
    assert int(worlds[0].alive.sum()) == int(worlds[1].alive.sum()) == 35
    assert torch.equal(observations[0], observations[1])
    assert sensor_ray_block_count(config, 4) == 2
    assert sensor_ray_block_count(config, 16) == 1
    profile = sensor_candidate_profile(worlds[0])
    assert profile['distinct'] > 1 and profile['nonzero'] > 0


@pytest.mark.skipif(
    os.environ.get('SLITHERAI_RUN_CUDA_FIXTURE') != '1' or not torch.cuda.is_available(),
    reason='Run only inside an approved CUDA benchmark window',
)
def test_cuda_long_body_death_and_crowding_fixture():
    config = SimConfig(maps=16, worms=8, foods=256, preys=0, body_points=96,
                       arena_radius=2400)
    result = run_long_body_fixture(config, seed=917, chunks=(4, 16), device='cuda')

    assert result['all_parity_equal']
    assert all(row['active_body_points'] == 95 for row in result['results'])
    assert all(row['alive'] == 108 and row['dead'] == 20 for row in result['results'])
    assert {row['sensor_chunk']: row['raycast_blocks'] for row in result['results']} == {4: 4, 16: 1}
