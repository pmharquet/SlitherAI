import dataclasses
import json
from types import SimpleNamespace

import neat
import numpy as np
import pytest
import torch

from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.protocol import protocol_settings
from slitherai.train import (Trainer, aggregate_selfplay_scores,
                             midrank_percentiles, mixed_reference_scenarios,
                             remap_mixed_slot_values, selfplay_scenarios,
                             validate_mixed_reference_layout, validate_selfplay_layout)


def selfplay_config(**changes):
    base = SimConfig(maps=2, worms=4, foods=8, body_points=8, preys=0,
                     arena_radius=400., arena_variation=0.)
    return dataclasses.replace(base, **changes)


def test_selfplay_schedule_is_deterministic_balanced_and_rotates_seats():
    games = selfplay_scenarios(71, 4, population_size=24, worms=4, training_games=4)
    repeated = selfplay_scenarios(71, 4, population_size=24, worms=4, training_games=4)
    later = selfplay_scenarios(71, 5, population_size=24, worms=4, training_games=4)
    assert games == repeated
    assert [game['seed'] for game in games] != [game['seed'] for game in later]

    cohort_membership = []
    seats_by_genome = {genome: [] for genome in range(24)}
    partners_by_genome = {genome: [] for genome in range(24)}
    for game in games:
        assignment = np.asarray(game['assignment']).reshape(6, 4)
        assert sorted(assignment.reshape(-1).tolist()) == list(range(24))
        cohort_membership.append({frozenset(row) for row in assignment.tolist()})
        for arena, row in enumerate(assignment):
            for genome in row:
                seats_by_genome[int(genome)].append(int(np.flatnonzero(row == genome)[0]))
                partners_by_genome[int(genome)].append(frozenset(int(other) for other in row if other != genome))
    assert all(len(set(seats)) == len(games) for seats in seats_by_genome.values())
    assert all(len(set(partners)) > 1 for partners in partners_by_genome.values())
    assert any(cohort_membership[0] != cohorts for cohorts in cohort_membership[1:])


def test_selfplay_score_uses_equal_mean_and_median_weight():
    scores = aggregate_selfplay_scores([[0., 0., 100.], [2., 4., 6.]])
    np.testing.assert_allclose(scores, [50. / 3, 4.])
    with pytest.raises(ValueError, match='finite genome-by-game'):
        aggregate_selfplay_scores([[1., np.nan]])


def test_midranks_are_tie_aware_scale_invariant_and_finite():
    scores = np.asarray([1., 2., 2., 4.])
    expected = np.asarray([.125, .5, .5, .875])
    np.testing.assert_allclose(midrank_percentiles(scores), expected)
    np.testing.assert_allclose(midrank_percentiles(scores * 17 + 300), expected)
    np.testing.assert_allclose(midrank_percentiles(np.ones(4)), np.full(4, .5))
    with pytest.raises(ValueError, match='finite score vector'):
        midrank_percentiles([1., np.inf])


def test_mixed_reference_v2_schedule_keeps_historical_assignments():
    games = mixed_reference_scenarios(81, 7, mixed_version=2)
    assert games == mixed_reference_scenarios(81, 7, mixed_version=2)
    assert games[0]['seed'] != games[1]['seed']
    assert games[0]['candidate_seats'] == list(range(8))
    assert games[1]['candidate_seats'] == list(range(8, 16))
    cohorts = []
    for game in games:
        assignment = np.asarray(game['assignment']).reshape(32, 16)
        assert np.all((assignment >= 0).sum(axis=1) == 8)
        assert np.all((assignment == -1).sum(axis=1) == 8)
        assert sorted(assignment[assignment >= 0].tolist()) == list(range(256))
        cohorts.append({frozenset(row[row >= 0].tolist()) for row in assignment})
    assert cohorts[0] != cohorts[1]
    for genome in range(256):
        seats = [int(np.flatnonzero(row == genome)[0])
                 for game in games
                 for row in np.asarray(game['assignment']).reshape(32, 16)
                 if genome in row]
        assert len(seats) == 2 and seats[0] != seats[1]


def test_mixed_reference_v3_schedule_is_balanced_and_rotates_seats_across_generations():
    games = mixed_reference_scenarios(81, 7)
    assert games == mixed_reference_scenarios(81, 7)
    assert games[0]['seed'] != games[1]['seed']
    assert games[0]['candidate_seats'] == [12, 13, 14, 15]
    assert games[1]['candidate_seats'] == [4, 5, 6, 7]
    cohorts = []
    seats_by_genome = {genome: [] for genome in range(256)}
    for game in games:
        assignment = np.asarray(game['assignment']).reshape(64, 16)
        assert np.all((assignment >= 0).sum(axis=1) == 4)
        assert np.all((assignment == -1).sum(axis=1) == 12)
        assert sorted(assignment[assignment >= 0].tolist()) == list(range(256))
        cohorts.append({frozenset(row[row >= 0].tolist()) for row in assignment})
        for row in assignment:
            for genome in row[row >= 0]:
                seats_by_genome[int(genome)].append(int(np.flatnonzero(row == genome)[0]))
    assert cohorts[0] != cohorts[1]
    assert all(len(seats) == 2 and seats[0] != seats[1]
               for seats in seats_by_genome.values())
    next_generation = mixed_reference_scenarios(81, 8)
    assert next_generation[0]['candidate_seats'] == [0, 1, 2, 3]
    assert next_generation[1]['candidate_seats'] == [8, 9, 10, 11]


def test_mixed_reference_metrics_remap_to_candidate_genomes_only():
    assignment = np.asarray([0, -1, 1, -1, 2, -1, 3, -1])
    slot_values = np.asarray([10., 101., 20., 102., 30., 103., 40., 104.])
    np.testing.assert_array_equal(remap_mixed_slot_values(slot_values, assignment, 4),
                                  [10., 20., 30., 40.])
    with pytest.raises(ValueError, match='exactly once'):
        remap_mixed_slot_values(slot_values, [0, -1, 0, -1, 2, -1, 3, -1], 4)
    with pytest.raises(ValueError, match='flat vectors'):
        remap_mixed_slot_values(slot_values, [0., -1., 1., -1., 2., -1., 3., -1], 4)


def test_mixed_reference_evaluation_scores_each_candidate_from_two_games(tmp_path, monkeypatch):
    from slitherai import train as train_module

    torch.set_num_threads(2)
    neat_config = load_config(256)
    genomes = list(neat.Population(neat_config).population.items())
    config = SimConfig(maps=64, worms=16, foods=8, body_points=8, preys=0, sensor_chunk=8)
    trainer = Trainer(config, tmp_path, 'cpu', seed=81, validation_every=5,
                      opponent_mode='mixed-reference')
    assert trainer.episode_seconds == 90
    assert trainer.stagnation_metric == 'within_generation_midrank_percentile'
    trainer.episode_seconds = .1
    trainer.controls = lambda: {}
    trainer.publish_preview = lambda control: None
    trainer.status = lambda phase, **extra: None
    calls = []
    score_transform = [1., 0.]

    def fake_mixed_episode(candidates, neat_config, sim_config, device, seed, assignment,
                           seconds, shared_random=True, on_tick=None, controls=None):
        assignment = np.asarray(assignment, dtype=np.int64)
        trial = len(calls)
        calls.append((seed, assignment.copy()))
        slot_values = np.full(sim_config.maps * sim_config.worms, 10_000. + trial)
        candidate_values = assignment[assignment >= 0] + trial * 100.
        slot_values[assignment >= 0] = (score_transform[0] * candidate_values
                                        + score_transform[1])
        if on_tick:
            world = SimpleNamespace(c=sim_config, steps=1, elapsed=sim_config.dt,
                alive=torch.ones((sim_config.maps, sim_config.worms), dtype=torch.bool))
            on_tick(world, SimpleNamespace(), torch.zeros((len(assignment), 530)), assignment, 0)
        return dict(fitness=slot_values.tolist(), food_gain=slot_values.tolist(),
                    alive=np.ones_like(slot_values).tolist())

    monkeypatch.setattr(train_module, 'play_mixed_population_episode', fake_mixed_episode)
    trainer.evaluate(genomes, neat_config)
    schedule = mixed_reference_scenarios(81, 0)
    assert [seed for seed, _ in calls] == [game['seed'] for game in schedule]
    for (_, observed), game in zip(calls, schedule):
        np.testing.assert_array_equal(observed, game['assignment'])

    expected = np.arange(256, dtype=np.float64) + 50.
    expected_ranks = midrank_percentiles(expected)
    np.testing.assert_allclose([genome.fitness for _, genome in genomes], expected)
    np.testing.assert_allclose([genome.stagnation_fitness for _, genome in genomes], expected_ranks)
    assert all(genome.stagnation_fitness is not None for _, genome in genomes)
    assert all(genome.behavior['food_gain'] == pytest.approx(expected[index])
               for index, (_, genome) in enumerate(genomes))
    episode = json.loads((tmp_path / 'episodes' / 'generation-0000.json').read_text())
    assert episode['opponent_mode'] == 'mixed-reference'
    assert episode['protocol'] == 'mixed-reference-v3'
    assert episode['aggregation'] == 'arithmetic_mean_two_games'
    assert trainer.evaluation_metrics['candidates_per_arena'] == 4
    assert trainer.evaluation_metrics['references_per_arena'] == 12
    assert all(row == [float(index), float(index + 100)]
               for index, row in enumerate(episode['game_scores']))

    score_transform[:] = [9., 71.]
    calls.clear()
    trainer.evaluate(genomes, neat_config)
    np.testing.assert_allclose([genome.stagnation_fitness for _, genome in genomes], expected_ranks)


def test_mixed_preview_uses_compact_network_slots_and_marks_heuristics(tmp_path):
    config = SimConfig(maps=64, worms=16, foods=8, body_points=8, preys=0, sensor_chunk=8)
    trainer = Trainer(config, tmp_path, 'cpu', opponent_mode='mixed-reference')
    schedule = mixed_reference_scenarios(91, 0)
    assignment = np.asarray(schedule[0]['assignment'], dtype=np.int64)
    world_slots = np.flatnonzero(assignment >= 0).tolist()

    class PreviewNetwork:
        def __init__(self):
            self.world_slots = world_slots
            self.world_to_network_slot = [-1] * len(assignment)
            for network_slot, world_slot in enumerate(world_slots):
                self.world_to_network_slot[world_slot] = network_slot
            self.assignment = [int(assignment[slot]) for slot in world_slots]
            self.genomes = [SimpleNamespace(key=index) for index in range(256)]
            self.described = None

        def describe(self, slot, inputs, sensor_version):
            self.described = (slot, float(inputs[slot, 0]))
            return dict(genome_id=self.assignment[slot], nodes=[], connections=[])

    class PreviewWorld:
        c = config
        elapsed = .8
        steps = 8

        def __init__(self):
            self.alive = torch.ones((64, 16), dtype=torch.bool)
            self.mass = torch.zeros((64, 16))
            self.mass[0, 2] = 99.

        def snapshot(self, arena):
            return dict(arena=arena, radius=400., elapsed=.8, config=dataclasses.asdict(config),
                worms=[dict(id=i, alive=True, body=[[float(i), 0.]]) for i in range(16)],
                food=[], maps=[])

    network = PreviewNetwork()
    observations = torch.zeros((64 * 16, 530))
    observations[:, 0] = torch.arange(64 * 16)
    trainer.generation = 3
    trainer.progress = {}
    trainer.live_world = PreviewWorld()
    trainer.live_network = network
    trainer.live_observations = observations
    trainer.live_assignment = assignment
    trainer.live_focal = None
    trainer.species_tracker = SimpleNamespace(population=SimpleNamespace(
        species=SimpleNamespace(genome_to_species={index: index % 7 for index in range(256)})))

    trainer.publish_preview({'arena': 0, 'network_worm': 15})
    snapshot = json.loads((tmp_path / 'preview.json').read_text())
    assert network.described == (2, 2.)
    assert [worm['controller'] for worm in snapshot['worms']] == [
        'neat' if value >= 0 else 'reference' for value in assignment[:16]]
    assert snapshot['species_ids'] == [int(assignment[index]) % 7 if assignment[index] >= 0 else None
                                       for index in range(16)]
    assert snapshot['network']['worm'] == 2


def test_selfplay_evaluation_maps_slot_results_back_to_each_genome(tmp_path, monkeypatch):
    from slitherai import train as train_module

    torch.set_num_threads(2)
    neat_config = load_config(8)
    genomes = list(neat.Population(neat_config).population.items())
    trainer = Trainer(selfplay_config(), tmp_path, 'cpu', seed=71,
                      validation_every=0, opponent_mode='selfplay', training_games=3)
    assert trainer.active_protocol['version'] == 'population-selfplay-v2'
    assert trainer.stagnation_metric == 'within_generation_midrank_percentile'
    trainer.generation = 2
    trainer.episode_seconds = .2
    trainer.controls = lambda: {}
    trainer.publish_preview = lambda control: None
    trainer.status = lambda phase, **extra: None
    calls = []
    score_transform = [1., 0.]

    def fake_population_episode(candidates, config, sim_config, device, seed, assignment,
                                seconds, shared_random=True, on_tick=None, controls=None):
        assignment = np.asarray(assignment, dtype=np.int64)
        calls.append((sim_config.maps, seed, assignment.copy()))
        base_values = assignment.astype(np.float64) * 10 + seed % 97
        values = score_transform[0] * base_values + score_transform[1]
        if on_tick:
            world = SimpleNamespace(c=sim_config, steps=1, elapsed=sim_config.dt,
                                    alive=torch.ones((sim_config.maps, sim_config.worms), dtype=torch.bool))
            on_tick(world, None, torch.zeros((len(assignment), 530)), assignment, 0)
        return dict(fitness=values.tolist(), food_gain=values.tolist(),
                    alive=np.ones_like(values).tolist())

    monkeypatch.setattr(train_module, 'play_population_episode', fake_population_episode)
    trainer.evaluate(genomes, neat_config)

    schedule = selfplay_scenarios(71, 2, 8, 4, 3)
    assert [(maps, seed) for maps, seed, _ in calls] == [(2, game['seed']) for game in schedule]
    for (_, _, observed), game in zip(calls, schedule):
        np.testing.assert_array_equal(observed, game['assignment'])
    game_scores = np.empty((8, 3))
    for trial, (_, seed, assignment) in enumerate(calls):
        game_scores[assignment, trial] = assignment * 10 + seed % 97
    expected = aggregate_selfplay_scores(game_scores)
    np.testing.assert_allclose([genome.fitness for _, genome in genomes], expected)
    np.testing.assert_allclose([genome.anchor_fitness for _, genome in genomes], expected)
    expected_ranks = midrank_percentiles(expected)
    np.testing.assert_allclose([genome.stagnation_fitness for _, genome in genomes], expected_ranks)
    np.testing.assert_allclose(midrank_percentiles(expected * 9 + 71), expected_ranks)
    for index, (_, genome) in enumerate(genomes):
        assert genome.behavior['food_gain'] == pytest.approx(game_scores[index].mean())
    assert trainer.evaluation_metrics['games_per_genome'] == 3
    assert trainer.evaluation_metrics['anchor_fitness'] is None
    assert trainer.evaluation_metrics['anchor_fitness_compatibility_alias'] == pytest.approx(expected.mean())
    assert trainer.evaluation_metrics['stagnation_metric'] == 'within_generation_midrank_percentile'
    assert trainer.evaluation_metrics['stagnation_score_mean'] == pytest.approx(expected_ranks.mean())

    episode = json.loads((tmp_path / 'episodes' / 'generation-0002.json').read_text())
    assert episode['opponent_mode'] == 'selfplay'
    assert 'anchor_score' not in episode
    assert episode['assignments'] == [game['assignment'] for game in schedule]
    assert len(episode['game_scores']) == 8
    assert all(len(row) == 3 for row in episode['game_scores'])
    np.testing.assert_allclose(episode['stagnation_score'], expected_ranks)

    score_transform[:] = [9., 71.]
    calls.clear()
    trainer.evaluate(genomes, neat_config)
    assert all(genome.stagnation_fitness is not None for _, genome in genomes)
    np.testing.assert_allclose([genome.stagnation_fitness for _, genome in genomes], expected_ranks)


def test_selfplay_layout_must_match_population_and_arena_count():
    config = selfplay_config(maps=6)
    assert validate_selfplay_layout(config, population_size=24, training_games=4) == 6
    with pytest.raises(ValueError, match='maps=population/worms'):
        validate_selfplay_layout(config, population_size=32, training_games=4)
    with pytest.raises(ValueError, match='divisible by worms'):
        validate_selfplay_layout(config, population_size=23, training_games=4)
    with pytest.raises(ValueError, match='between 1 and worms'):
        validate_selfplay_layout(config, population_size=24, training_games=8)


def test_mixed_reference_layout_is_fixed_to_the_pilot_protocol():
    legacy_config = SimConfig(maps=32, worms=16)
    config = SimConfig(maps=64, worms=16, sensor_chunk=8)
    assert validate_mixed_reference_layout(legacy_config, 256, 2, mixed_version=1) == 32
    assert validate_mixed_reference_layout(legacy_config, 256, 2, mixed_version=2) == 32
    assert validate_mixed_reference_layout(config, 256, 2, mixed_version=3) == 64
    with pytest.raises(ValueError, match='Mixed-reference v3 requires'):
        validate_mixed_reference_layout(dataclasses.replace(config, maps=32), 256, 2, mixed_version=3)
    with pytest.raises(ValueError, match='Mixed-reference v3 requires'):
        validate_mixed_reference_layout(config, 128, 2, mixed_version=3)
    with pytest.raises(ValueError, match='sensor_chunk=8'):
        validate_mixed_reference_layout(dataclasses.replace(config, sensor_chunk=4), 256, 2,
                                        mixed_version=3)


def test_selfplay_protocol_refuses_reference_resume_but_supports_same_mode_resume(
        tmp_path, monkeypatch):
    from slitherai import train as train_module

    torch.set_num_threads(2)
    config = selfplay_config(maps=1)

    def fake_metrics(config, focal=None):
        return dict(fitness=np.ones(config.maps).tolist(),
                    food_gain=np.ones(config.maps).tolist(),
                    alive=np.ones(config.maps).tolist())

    def fake_reference(genomes, neat_config, sim_config, device, seed, focal, seconds,
                       shared_random=False, on_tick=None, controls=None, policy='neat'):
        if on_tick:
            world = SimpleNamespace(c=sim_config, steps=1, elapsed=sim_config.dt,
                alive=torch.ones((sim_config.maps, sim_config.worms), dtype=torch.bool))
            on_tick(world, None, torch.zeros((sim_config.maps, 530)),
                    torch.as_tensor(focal), 0)
        return fake_metrics(sim_config)

    def fake_population(genomes, neat_config, sim_config, device, seed, assignment, seconds,
                        shared_random=True, on_tick=None, controls=None):
        assignment = np.asarray(assignment, dtype=np.int64)
        if on_tick:
            world = SimpleNamespace(c=sim_config, steps=1, elapsed=sim_config.dt,
                alive=torch.ones((sim_config.maps, sim_config.worms), dtype=torch.bool))
            on_tick(world, None, torch.zeros((len(assignment), 530)), assignment, 0)
        return dict(fitness=np.ones(len(assignment)).tolist(),
                    food_gain=np.ones(len(assignment)).tolist(),
                    alive=np.ones(len(assignment)).tolist())

    monkeypatch.setattr(train_module, 'play_episode', fake_reference)
    monkeypatch.setattr(train_module, 'play_population_episode', fake_population)
    monkeypatch.setattr(Trainer, 'publish_preview', lambda self, control: None)

    reference_run = tmp_path / 'reference'
    Trainer(config, reference_run, 'cpu', seed=17, validation_every=0).train(4, 1, .1)
    assert json.loads((reference_run / 'settings.json').read_text())['protocol'] == protocol_settings()
    with pytest.raises(ValueError, match='Evaluation protocol changed'):
        Trainer(config, reference_run, 'cpu', seed=17, validation_every=0,
                opponent_mode='selfplay', training_games=2).train(
                    4, 1, .1, resume=reference_run / 'checkpoint-1')

    selfplay_run = tmp_path / 'selfplay'
    trainer = Trainer(config, selfplay_run, 'cpu', seed=17, validation_every=0,
                      opponent_mode='selfplay', training_games=2)
    trainer.train(4, 1, .1)
    settings = json.loads((selfplay_run / 'settings.json').read_text())
    assert settings['protocol'] == protocol_settings('selfplay', 2)
    assert settings['opponent_mode'] == 'selfplay'
    assert settings['training_games'] == 2 and settings['maps_per_game'] == 1

    resumed = Trainer(config, selfplay_run, 'cpu', seed=17, validation_every=0,
                      opponent_mode='selfplay', training_games=2)
    resumed.train(4, 1, .1, resume=selfplay_run / 'checkpoint-1')
    assert (selfplay_run / 'checkpoint-2').exists()


def test_selfplay_v1_checkpoint_resumes_with_historical_raw_stagnation(tmp_path, monkeypatch):
    from slitherai import train as train_module

    base_protocol_settings = train_module.protocol_settings
    monkeypatch.setattr(train_module, 'protocol_settings',
        lambda mode='reference', training_games=5: base_protocol_settings(
            mode, training_games, selfplay_version=1) if mode == 'selfplay'
            else base_protocol_settings(mode, training_games))

    def fake_population(genomes, neat_config, sim_config, device, seed, assignment,
                        seconds, shared_random=True, on_tick=None, controls=None):
        assignment = np.asarray(assignment, dtype=np.int64)
        if on_tick:
            world = SimpleNamespace(c=sim_config, steps=1, elapsed=sim_config.dt,
                alive=torch.ones((sim_config.maps, sim_config.worms), dtype=torch.bool))
            on_tick(world, None, torch.zeros((len(assignment), 530)), assignment, 0)
        values = assignment.astype(float) + seed % 5
        return dict(fitness=values.tolist(), food_gain=values.tolist(),
                    alive=np.ones(len(assignment)).tolist())

    monkeypatch.setattr(train_module, 'play_population_episode', fake_population)
    monkeypatch.setattr(Trainer, 'publish_preview', lambda self, control: None)
    config = selfplay_config(maps=1)
    run = tmp_path / 'selfplay-v1'

    Trainer(config, run, 'cpu', seed=91, validation_every=0,
            opponent_mode='selfplay', training_games=2).train(4, 1, .1)
    saved = json.loads((run / 'settings.json').read_text())['protocol']
    assert saved == protocol_settings('selfplay', 2, selfplay_version=1)

    resumed = Trainer(config, run, 'cpu', seed=91, validation_every=0,
                      opponent_mode='selfplay', training_games=2)
    resumed.train(4, 1, .1, resume=run / 'checkpoint-1')
    assert resumed.active_protocol == saved
    assert resumed.stagnation_metric == 'raw'
    population = neat.Checkpointer.restore_checkpoint(str(run / 'checkpoint-2'))
    assert population.config.stagnation_config.progress_metric == 'raw'


def test_mixed_reference_v1_resume_keeps_45_seconds_under_v2_default(tmp_path, monkeypatch):
    from slitherai import train as train_module

    base_protocol_settings = train_module.protocol_settings
    monkeypatch.setattr(train_module, 'protocol_settings',
        lambda mode='reference', training_games=5, mixed_version=None: base_protocol_settings(
            mode, training_games, mixed_version=1 if mixed_version is None else mixed_version)
            if mode == 'mixed-reference' else base_protocol_settings(mode, training_games))

    def fake_evaluate(self, genomes, neat_config):
        raw_scores = np.asarray([float(key % 17) for key, _ in genomes])
        ranks = midrank_percentiles(raw_scores)
        for index, (_, genome) in enumerate(genomes):
            genome.fitness = float(raw_scores[index])
            genome.stagnation_fitness = float(ranks[index])
            genome.behavior = {}
        self.evaluation_metrics = {'fitness': float(raw_scores.mean())}

    monkeypatch.setattr(Trainer, 'evaluate', fake_evaluate)
    monkeypatch.setattr(Trainer, 'publish_preview', lambda self, control: None)
    monkeypatch.setattr(train_module, 'fixed_validation',
        lambda *args, **kwargs: {'fitness': 1., 'per_map': []})
    config = SimConfig(maps=32, worms=16, foods=8, body_points=8, preys=0)
    run = tmp_path / 'mixed-v1-resume'
    first = Trainer(config, run, 'cpu', seed=37, validation_every=5,
                    opponent_mode='mixed-reference')
    first.train(256, 1, seconds=None)
    saved = json.loads((run / 'settings.json').read_text())
    assert saved['protocol'] == base_protocol_settings(
        'mixed-reference', 2, mixed_version=1)
    assert saved['seconds'] == 45

    monkeypatch.setattr(train_module, 'protocol_settings', base_protocol_settings)
    resumed = Trainer(config, run, 'cpu', seed=37, validation_every=5,
                      opponent_mode='mixed-reference')
    assert resumed.active_protocol['version'] == 'mixed-reference-v3'
    resumed.train(256, 1, seconds=None, resume=run / 'checkpoint-1')
    assert resumed.active_protocol == saved['protocol']
    assert resumed.episode_seconds == 45
    assert json.loads((run / 'settings.json').read_text())['seconds'] == 45
