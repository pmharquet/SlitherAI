import dataclasses
import os
import random

import neat
import pytest
import torch

from slitherai.benchmark_observation import (
    _capture_episode_trace, _event_difference, _fixed_actions, build_long_body_fixture,
    run_adaptive_tile_sweep, run_long_body_fixture, run_paired_policy_episode,
    sensor_candidate_profile, sensor_ray_block_count,
)
from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.schema import ANGLES
from slitherai.sim import (
    WorldBatch, adaptive_sensor_tiling, choose_sensor_chunk, sensor_tile_work_elements,
)


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


def test_adaptive_sensor_chunk_budget_boundaries_and_block_counts():
    maps, worms, rays, candidates = 17, 8, len(ANGLES), 20
    work = {chunk: sensor_tile_work_elements(chunk, maps, worms, rays, candidates)
            for chunk in (4, 8, 16)}

    assert choose_sensor_chunk(candidates, maps, worms, rays, work[4] - 1) == 4
    assert choose_sensor_chunk(candidates, maps, worms, rays, work[4]) == 4
    assert choose_sensor_chunk(candidates, maps, worms, rays, work[8] - 1) == 4
    assert choose_sensor_chunk(candidates, maps, worms, rays, work[8]) == 8
    assert choose_sensor_chunk(candidates, maps, worms, rays, work[16] - 1) == 8
    assert choose_sensor_chunk(candidates, maps, worms, rays, work[16]) == 16
    config = SimConfig(maps=maps, worms=worms, foods=32, preys=0, body_points=96)
    assert [sensor_ray_block_count(config, chunk) for chunk in (4, 8, 16)] == [5, 3, 2]


def test_paired_policy_episode_has_per_step_observation_action_reward_and_state_parity():
    random.seed(71)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    config = SimConfig(maps=17, worms=2, foods=32, preys=0, body_points=16,
                       arena_radius=1200)

    result = run_paired_policy_episode(config, genome, neat_config,
        seeds=(71,), seconds=.2, device='cpu', chunks=(4, 8, 16),
        diagnostic_seeds=(71,), repeat_chunks=(4, 16))

    parity = result['parity_episodes'][0]
    assert parity['equal'] and parity['summary_metrics_equal']
    assert parity['steps_left'] == parity['steps_right'] == 2
    assert all(parity['fields'][field] for field in ('observations', 'actions', 'states', 'rewards'))
    assert result['all_parity_equal']
    assert {row['sensor_chunk']: row['raycast_blocks'] for row in result['timed_episodes']} == {4: 5, 8: 3, 16: 2}
    assert result['timing_summary']['4']['samples'] == result['timing_summary']['8']['samples'] == result['timing_summary']['16']['samples'] == 1
    assert [row['sensor_chunk'] for row in result['same_chunk_repeats']] == [4, 16]
    assert all(row['equal'] and row['summary_metrics_equal'] for row in result['same_chunk_repeats'])
    assert all(row['numeric_diagnostics']['states']['first_divergence'] is None
               for row in result['same_chunk_repeats'] if row['sensor_chunk'] == 4)
    assert all(row['numeric_diagnostics']['states']['first_divergence'] is None
               for row in result['same_chunk_repeats'] if row['sensor_chunk'] == 16)


def test_trace_numeric_diagnostic_reports_first_field_delta_and_rng_equality():
    left = [
        ('floating', torch.tensor([1., 2., 3.])),
        ('alive', torch.tensor([True, False])),
        ('rng_state', torch.tensor([3, 4, 5], dtype=torch.uint8)),
    ]
    right = [
        ('floating', torch.tensor([1., 2.25, 3.])),
        ('alive', torch.tensor([True, False])),
        ('rng_state', torch.tensor([3, 4, 5], dtype=torch.uint8)),
    ]

    details = _event_difference(left, right)

    assert not details['floating']['exact_equal']
    assert details['floating']['different_count'] == 1
    assert details['floating']['max_abs_difference'] == .25
    assert details['alive']['exact_equal']
    assert details['rng_state']['exact_equal']


def test_trace_comparison_finds_first_state_field_and_rng_status(monkeypatch):
    config = SimConfig(maps=1, worms=2, foods=16, preys=0, body_points=16,
                       arena_radius=1200)
    reference_world = WorldBatch(config, 'cpu', seed=81)
    with _capture_episode_trace(retain_values=True) as reference:
        reference_world.observe()
        reference_world.step(_fixed_actions(reference_world))

    original_step = WorldBatch.step

    def shift_head(world, actions):
        result = original_step(world, actions)
        with torch.inference_mode():
            world.head[0, 0, 0] += .25
        return result

    monkeypatch.setattr(WorldBatch, 'step', shift_head)
    comparison_world = WorldBatch(config, 'cpu', seed=81)
    with _capture_episode_trace(references=(('same_chunk_repeat', reference),)) as comparison:
        comparison_world.observe()
        comparison_world.step(_fixed_actions(comparison_world))

    first = comparison['comparisons']['same_chunk_repeat']['states']['first_divergence']
    assert first['step'] == 1
    assert first['first_field'] == 'head'
    assert first['rng_state_equal'] is True
    assert first['non_float_state_equal']['alive'] is True
    assert first['fields']['head']['exact_equal'] is False
    assert first['fields']['head']['different_count'] == 1
    assert first['fields']['head']['max_abs_difference'] == .25
    assert first['fields']['rng_state']['exact_equal'] is True
    assert first['fields']['alive']['exact_equal'] is True


def test_cpu_long_body_death_and_crowding_fixture_is_exact_across_chunks():
    config = SimConfig(maps=17, worms=8, foods=128, preys=0, body_points=96,
                       arena_radius=2400)
    worlds = [build_long_body_fixture(
        dataclasses.replace(config, sensor_chunk=chunk), 'cpu', seed=917)
        for chunk in (4, 8, 16)]
    observations = [world.observe() for world in worlds]

    assert worlds[0].active_body_points() == worlds[1].active_body_points() == 95
    assert int(worlds[0].alive.sum()) == int(worlds[1].alive.sum()) == int(worlds[2].alive.sum()) == 116
    assert torch.equal(observations[0], observations[1])
    assert torch.equal(observations[0], observations[2])
    assert [sensor_ray_block_count(config, chunk) for chunk in (4, 8, 16)] == [5, 3, 2]
    profile = sensor_candidate_profile(worlds[0])
    assert profile['distinct'] > 1 and profile['nonzero'] > 0

    adaptive_world = build_long_body_fixture(
        dataclasses.replace(config, sensor_chunk=4), 'cpu', seed=917)
    budget = sensor_tile_work_elements(16, config.maps, config.worms, len(ANGLES), profile['maximum'])
    with adaptive_sensor_tiling(budget) as tile_stats:
        adaptive_observation = adaptive_world.observe()
    assert torch.equal(adaptive_observation, observations[0])
    assert tile_stats.summary()['selected_chunk_counts']['16'] == 1
    assert adaptive_world.active_body_points() == 95
    assert int((~adaptive_world.alive).sum()) == 20


def test_cpu_dense_fixture_sweeps_adaptive_budgets_on_same_geometry():
    config = SimConfig(maps=5, worms=4, foods=32, preys=0, body_points=96,
                       arena_radius=2400)
    profile_world = build_long_body_fixture(config, 'cpu', seed=917)
    candidate_maximum = sensor_candidate_profile(profile_world)['maximum']
    low_budget = 1
    chunk16_budget = sensor_tile_work_elements(
        16, config.maps, config.worms, len(ANGLES), candidate_maximum)

    result = run_long_body_fixture(
        config, seed=917, chunks=(4, 16), device='cpu', repeats=2,
        warmup_repeats=1, adaptive_work_budgets=(low_budget, chunk16_budget))

    assert result['all_geometry_equal'] and result['all_parity_equal']
    assert result['repeats'] == 2 and result['warmup_repeats'] == 1
    fixed = {row['sensor_chunk']: row for row in result['results']
             if row['condition'] == 'fixed'}
    adaptive = {row['work_budget_elements']: row for row in result['results']
                if row['condition'] == 'adaptive'}
    assert {chunk: row['raycast_blocks'] for chunk, row in fixed.items()} == {4: 2, 16: 1}
    assert all(len(row['observe_samples_ms']) == 2 for row in (*fixed.values(), *adaptive.values()))
    assert adaptive[low_budget]['geometry_exact_to_reference']
    assert adaptive[low_budget]['tile_profile']['selected_chunk_counts']['4'] == 1
    assert adaptive[low_budget]['tile_profile']['smallest_tile_over_budget_calls'] == 1
    assert adaptive[chunk16_budget]['tile_profile']['selected_chunk_counts']['16'] == 1
    assert all(row['observation_exact_to_reference'] for row in adaptive.values())


def test_adaptive_policy_sweep_is_opt_in_and_records_exact_provenance():
    random.seed(71)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    config = SimConfig(maps=17, worms=2, foods=32, preys=0, body_points=16,
                       arena_radius=1200)

    result = run_adaptive_tile_sweep(
        config, genome, neat_config, work_budgets=(1, 10**9),
        seeds=(71,), seconds=.2, device='cpu')

    assert result['all_parity_equal']
    assert result['runtime_override'].startswith('opt-in')
    assert len(result['timing_summary']['adaptive']) == 2
    profiles = {row['work_budget_elements']: row['tile_profile']
                for row in result['parity_episodes']}
    assert profiles[1]['selected_chunk_counts']['4'] == 2
    assert profiles[1]['smallest_tile_over_budget_calls'] == 2
    assert profiles[10**9]['selected_chunk_counts']['16'] == 2
    assert all(row['equal'] and row['summary_metrics_equal']
               for row in result['parity_episodes'])


@pytest.mark.skipif(
    os.environ.get('SLITHERAI_RUN_CUDA_FIXTURE') != '1' or not torch.cuda.is_available(),
    reason='Run only inside an approved CUDA benchmark window',
)
def test_cuda_long_body_death_and_crowding_fixture():
    config = SimConfig(maps=16, worms=8, foods=256, preys=0, body_points=96,
                       arena_radius=2400)
    result = run_long_body_fixture(
        config, seed=917, chunks=(4, 8, 16), device='cuda',
        adaptive_work_budgets=(1, 10**9))

    assert result['all_parity_equal']
    assert result['all_geometry_equal']
    assert all(row['active_body_points'] == 95 for row in result['results'])
    assert all(row['alive'] == 108 and row['dead'] == 20 for row in result['results'])
    fixed = {row['sensor_chunk']: row for row in result['results'] if row['condition'] == 'fixed'}
    adaptive = {row['work_budget_elements']: row for row in result['results']
                if row['condition'] == 'adaptive'}
    assert {chunk: row['raycast_blocks'] for chunk, row in fixed.items()} == {4: 4, 8: 2, 16: 1}
    assert adaptive[1]['tile_profile']['selected_chunk_counts']['4'] == 1
    assert adaptive[10**9]['tile_profile']['selected_chunk_counts']['16'] == 1
