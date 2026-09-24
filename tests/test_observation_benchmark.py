import dataclasses
import hashlib
import json
import os
import pickle
import random

import neat
import pytest
import torch

from slitherai.benchmark_observation import (
    _capture_episode_trace, _event_difference, _fixed_actions, build_long_body_fixture,
    load_run_champion, run_adaptive_tile_sweep, run_long_body_fixture, run_paired_policy_episode,
    run_policy_timing_only, run_tiling_trace_diagnostic,
    sensor_version_from_schema,
    sensor_candidate_profile, sensor_ray_block_count,
)
from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.schema import ANGLES, VERSION, contract
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


def test_policy_timing_only_runs_two_untraced_cpu_episodes(monkeypatch):
    import slitherai.benchmark_observation as benchmark

    metrics = {key: [float(index + 1)] for index, key in enumerate((
        'fitness', 'alive', 'food_gain', 'boost_spent', 'kills',
        'border_death', 'collision_death'))}
    calls = []

    def fake_play(genome, neat_config, config, device, seed, seconds, trace=False,
                  collect_tile_stats=True, **kwargs):
        calls.append((config.sensor_chunk, device, seed, seconds, trace, collect_tile_stats))
        return metrics, .125, None

    monkeypatch.setattr(benchmark, '_play', fake_play)
    config = SimConfig(maps=5, worms=2, foods=32, preys=0, body_points=16)
    result = run_policy_timing_only(config, object(), object(), seed=1038282,
                                    seconds=60, device='cpu', chunks=(4, 8))

    assert calls == [(4, 'cpu', 1038282, 60, False, False),
                     (8, 'cpu', 1038282, 60, False, False)]
    assert result['mode'] == 'policy_timing_only'
    assert result['parity'].startswith('not measured')
    assert result['warmup_episodes'] == 0 and result['repetitions_per_chunk'] == 1
    assert [row['sensor_chunk'] for row in result['timed_episodes']] == [4, 8]
    assert all(row['wall_ms'] == 125. and row['peak_allocated_mib'] is None
               and row['peak_delta_mib'] is None for row in result['timed_episodes'])
    assert all(row['metrics']['fitness'] == 1. for row in result['timed_episodes'])
    assert result['wall_ms_ratio_first_chunk_over_second'] == 1.


def test_policy_timing_only_requires_two_distinct_chunks():
    config = SimConfig(maps=1, worms=2, foods=16, preys=0, body_points=16)
    with pytest.raises(ValueError, match='exactly two distinct'):
        run_policy_timing_only(config, object(), object(), device='cpu', chunks=(4, 4))


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
    assert details['floating']['bytewise_equal'] is False
    signed_zero = _event_difference(
        [('floating', torch.tensor([-0.]))], [('floating', torch.tensor([0.]))])
    assert signed_zero['floating']['exact_equal'] is True
    assert signed_zero['floating']['bytewise_equal'] is False
    assert details['alive']['exact_equal']
    assert details['alive']['bytewise_equal'] is True
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


def test_trace_capture_retains_only_selected_steps_under_payload_cap():
    config = SimConfig(maps=1, worms=2, foods=16, preys=0, body_points=16,
                       arena_radius=1200)
    world = WorldBatch(config, 'cpu', seed=83)
    selected = {'observations': {2}, 'actions': {2}, 'states': {1}, 'rewards': {1}}

    with _capture_episode_trace(retain_steps=selected,
                                max_retained_snapshot_bytes=1024 * 1024) as trace:
        for _ in range(3):
            world.observe()
            world.step(_fixed_actions(world))

    assert len(trace['observations']) == len(trace['actions']) == 3
    assert set(trace['snapshots']['observations']) == {2}
    assert set(trace['snapshots']['actions']) == {2}
    assert set(trace['snapshots']['states']) == {1}
    assert set(trace['snapshots']['rewards']) == {1}
    assert 0 < trace['retained_snapshot_tensor_bytes'] < 1024 * 1024

    too_small_world = WorldBatch(config, 'cpu', seed=84)
    with pytest.raises(MemoryError, match='tensor payload'):
        with _capture_episode_trace(retain_steps={'observations': {1}},
                                    max_retained_snapshot_bytes=1):
            too_small_world.observe()


def test_bounded_tiling_diagnostic_reports_numeric_deltas_and_rng(monkeypatch):
    import slitherai.benchmark_observation as benchmark

    metrics = {key: [1.] for key in (
        'fitness', 'alive', 'food_gain', 'boost_spent', 'kills',
        'border_death', 'collision_death')}

    def fake_play(genome, neat_config, config, device, seed, seconds, trace=False,
                  trace_references=(), retain_trace_values=False,
                  retain_trace_steps=None, max_retained_snapshot_bytes=128 * 1024 * 1024,
                  adaptive_work_budget_elements=None, collect_tile_stats=True,
                  adaptive_tile_choices=(4, 8, 16)):
        adaptive = adaptive_work_budget_elements is not None
        traces = {name: ['same-1', 'same-2'] for name in ('observations', 'actions', 'states', 'rewards')}
        if adaptive:
            traces['observations'][0] = 'adaptive-observation'
            traces['states'][0] = 'adaptive-state'
        snapshots = None
        retained_bytes = 0
        if retain_trace_steps is not None:
            snapshots = {name: {} for name in traces}
            for group, steps in retain_trace_steps.items():
                for step in steps:
                    if group == 'observations':
                        values = [('observation', torch.tensor([.25 if adaptive else 0.]))]
                    elif group == 'states':
                        values = [('head', torch.tensor([.5 if adaptive else 0.])),
                                  ('alive', torch.tensor([True])),
                                  ('rng_state', torch.tensor([4, 5], dtype=torch.uint8))]
                    else:
                        values = [(group, torch.tensor([0.]))]
                    snapshots[group][step] = values
                    retained_bytes += sum(value.numel() * value.element_size()
                                          for _, value in values if isinstance(value, torch.Tensor))
            assert retained_bytes <= max_retained_snapshot_bytes
        trace_result = dict(traces, snapshots=snapshots, retained_snapshot_tensor_bytes=retained_bytes)
        profile = ({'selected_chunk_counts': {'4': 0, '8': 0, '16': 2}}
                   if adaptive else None)
        return metrics, trace_result, profile

    monkeypatch.setattr(benchmark, '_play', fake_play)
    config = SimConfig(maps=1, worms=2, foods=16, preys=0, body_points=16)
    result = run_tiling_trace_diagnostic(config, object(), object(), seed=1038282,
                                         seconds=.2, adaptive_work_budget=4_000_000,
                                         device='cpu')

    assert [row['name'] for row in result['comparisons']] == [
        'same_chunk_4_repeatability', 'same_chunk_16_repeatability', 'fixed4_vs_adaptive']
    assert result['comparisons'][0]['equal'] and result['comparisons'][1]['equal']
    adaptive = result['comparisons'][2]
    assert not adaptive['equal']
    assert adaptive['numeric_diagnostics']['observations']['first_field'] == 'observation'
    assert adaptive['numeric_diagnostics']['observations']['fields']['observation']['max_abs_difference'] == .25
    state = adaptive['numeric_diagnostics']['states']
    assert state['first_field'] == 'head' and state['rng_state_equal'] is True
    assert state['fields']['head']['different_count'] == 1
    assert state['fields']['head']['max_abs_difference'] == .5
    assert state['non_float_state_equal']['alive'] is True
    assert result['peak_pair_snapshot_tensor_bytes'] <= result['snapshot_tensor_byte_cap']
    assert not result['all_parity_equal']


def test_tiling_trace_diagnostic_runs_bounded_cpu_matrix():
    random.seed(71)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    config = SimConfig(maps=5, worms=2, foods=32, preys=0, body_points=16,
                       arena_radius=1200)

    result = run_tiling_trace_diagnostic(
        config, genome, neat_config, seed=71, seconds=.2,
        adaptive_work_budget=10**9, device='cpu')

    assert result['simulated_step_limit'] == 2
    assert result['all_parity_equal']
    assert len(result['comparisons']) == 3
    assert all(row['equal'] for row in result['comparisons'])
    assert result['peak_pair_snapshot_tensor_bytes'] == 0
    assert result['adaptive_tile_profile']['selected_chunk_counts']['16'] == 2


def test_load_run_champion_selects_explicit_payload_and_validates_schema(tmp_path):
    config = SimConfig(maps=2, worms=2, foods=16, preys=0, body_points=16)
    (tmp_path/'settings.json').write_text(
        json.dumps({'config': dataclasses.asdict(config)}), encoding='utf-8')
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genomes = list(population.population.values())
    default_payload = dict(genome=genomes[0], config=neat_config,
                           schema=VERSION, generation=10)
    selected_payload = dict(genome=genomes[1], config=neat_config,
                            schema=VERSION, generation=9)
    (tmp_path/'champion.pkl').write_bytes(pickle.dumps(default_payload))
    model_path = tmp_path/'checkpoint9-genome1065.pkl'
    model_path.write_bytes(pickle.dumps(selected_payload))

    sim_config, genome, loaded_config, source = load_run_champion(tmp_path, model_path)

    assert sim_config.maps == 2
    assert genome.key == genomes[1].key
    assert loaded_config.genome_config.input_keys == neat_config.genome_config.input_keys
    assert source['genome_generation'] == 9
    assert source['genome_id'] == genomes[1].key
    assert source['genome_file'] == str(model_path.resolve())
    assert source['genome_file_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    settings_path = tmp_path/'settings.json'
    assert source['settings_file_sha256'] == hashlib.sha256(settings_path.read_bytes()).hexdigest()
    assert len(source['genome_gene_sha256']) == 64

    incompatible = dict(selected_payload, schema='wrong-schema')
    bad_path = tmp_path/'wrong-schema.pkl'
    bad_path.write_bytes(pickle.dumps(incompatible))
    with pytest.raises(ValueError, match='schema'):
        load_run_champion(tmp_path, bad_path)

    wrong_dimensions = load_config(4)
    wrong_dimensions.genome_config.input_keys = wrong_dimensions.genome_config.input_keys[:-1]
    incompatible = dict(selected_payload, config=wrong_dimensions)
    bad_path = tmp_path/'wrong-dimensions.pkl'
    bad_path.write_bytes(pickle.dumps(incompatible))
    with pytest.raises(ValueError, match='530 inputs and 2 outputs'):
        load_run_champion(tmp_path, bad_path)

    wrong_key_order = load_config(4)
    wrong_key_order.genome_config.input_keys.reverse()
    incompatible = dict(selected_payload, config=wrong_key_order)
    bad_path = tmp_path/'wrong-key-order.pkl'
    bad_path.write_bytes(pickle.dumps(incompatible))
    with pytest.raises(ValueError, match='input/output keys'):
        load_run_champion(tmp_path, bad_path)

    wrong_activation = pickle.loads(pickle.dumps(selected_payload))
    wrong_activation['genome'].nodes[0].activation = 'tanh'
    bad_path = tmp_path/'wrong-activation.pkl'
    bad_path.write_bytes(pickle.dumps(wrong_activation))
    with pytest.raises(ValueError, match='sigmoid activation and sum aggregation'):
        load_run_champion(tmp_path, bad_path)


def test_sensor_version_schema_normalizer_accepts_legacy_and_export_contracts():
    assert sensor_version_from_schema(VERSION) == 'legacy-v1'
    assert sensor_version_from_schema(contract()) == 'legacy-v1'
    legacy_contract = dict(contract())
    legacy_contract.update(
        version=VERSION, sensor_version='legacy-v1',
        self_body_model='sim_physical_radius_along_body_3r')
    assert sensor_version_from_schema(legacy_contract) == 'legacy-v1'
    pre_version_contract = dict(legacy_contract)
    pre_version_contract.pop('sensor_version')
    pre_version_contract.pop('self_body_model')
    assert sensor_version_from_schema(pre_version_contract) == 'legacy-v1'
    assert sensor_version_from_schema({'version': VERSION, 'sensor_version': 'export-v1'}) is None

    export_contract = dict(contract())
    export_contract.update(
        version='slither-neat-530-export-v1', sensor_version='export-v1',
        self_body_model='extension_0_6_0_radius_2r_plus_3_spatial_cutoff_2r')
    assert sensor_version_from_schema(export_contract) == 'export-v1'
    export_contract['version'] = VERSION
    assert sensor_version_from_schema(export_contract) is None
    assert sensor_version_from_schema({'sensor_version':'unknown-v9'}) is None


def test_load_run_champion_accepts_matching_export_schema_and_rejects_cross_mode(tmp_path, monkeypatch):
    import slitherai.benchmark_observation as benchmark

    class SensorAwareConfig:
        def __init__(self, sensor_version):
            self.sensor_version = sensor_version

        @classmethod
        def from_dict(cls, data):
            return cls(data.get('sensor_version', 'legacy-v1'))

        def validate(self):
            return self

    monkeypatch.setattr(benchmark, 'SimConfig', SensorAwareConfig)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome = next(iter(population.population.values()))
    export_contract = dict(contract())
    export_contract.update(
        version='slither-neat-530-export-v1', sensor_version='export-v1',
        self_body_model='extension_0_6_0_radius_2r_plus_3_spatial_cutoff_2r')
    model_path = tmp_path/'export-model.pkl'
    model_path.write_bytes(pickle.dumps(dict(
        genome=genome, config=neat_config, schema=export_contract, generation=1)))

    (tmp_path/'settings.json').write_text(
        json.dumps({'config': {'sensor_version':'export-v1'}}), encoding='utf-8')
    sim_config, selected, _, source = load_run_champion(tmp_path, model_path)
    assert sim_config.sensor_version == 'export-v1'
    assert selected.key == genome.key
    assert source['sensor_version'] == 'export-v1'

    (tmp_path/'settings.json').write_text(
        json.dumps({'config': {'sensor_version':'legacy-v1'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match run settings'):
        load_run_champion(tmp_path, model_path)


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
