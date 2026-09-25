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
                             selfplay_scenarios, validate_selfplay_layout)


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


def test_selfplay_evaluation_maps_slot_results_back_to_each_genome(tmp_path, monkeypatch):
    from slitherai import train as train_module

    torch.set_num_threads(2)
    neat_config = load_config(8)
    genomes = list(neat.Population(neat_config).population.items())
    trainer = Trainer(selfplay_config(), tmp_path, 'cpu', seed=71,
                      validation_every=0, opponent_mode='selfplay', training_games=3)
    trainer.generation = 2
    trainer.episode_seconds = .2
    trainer.controls = lambda: {}
    trainer.publish_preview = lambda control: None
    trainer.status = lambda phase, **extra: None
    calls = []

    def fake_population_episode(candidates, config, sim_config, device, seed, assignment,
                                seconds, shared_random=True, on_tick=None, controls=None):
        assignment = np.asarray(assignment, dtype=np.int64)
        calls.append((sim_config.maps, seed, assignment.copy()))
        values = assignment.astype(np.float64) * 10 + seed % 97
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
    for index, (_, genome) in enumerate(genomes):
        assert genome.behavior['food_gain'] == pytest.approx(game_scores[index].mean())
    assert trainer.evaluation_metrics['games_per_genome'] == 3
    assert trainer.evaluation_metrics['anchor_fitness'] is None
    assert trainer.evaluation_metrics['anchor_fitness_compatibility_alias'] == pytest.approx(expected.mean())

    episode = json.loads((tmp_path / 'episodes' / 'generation-0002.json').read_text())
    assert episode['opponent_mode'] == 'selfplay'
    assert 'anchor_score' not in episode
    assert episode['assignments'] == [game['assignment'] for game in schedule]
    assert len(episode['game_scores']) == 8
    assert all(len(row) == 3 for row in episode['game_scores'])


def test_selfplay_layout_must_match_population_and_arena_count():
    config = selfplay_config(maps=6)
    assert validate_selfplay_layout(config, population_size=24, training_games=4) == 6
    with pytest.raises(ValueError, match='maps=population/worms'):
        validate_selfplay_layout(config, population_size=32, training_games=4)
    with pytest.raises(ValueError, match='divisible by worms'):
        validate_selfplay_layout(config, population_size=23, training_games=4)
    with pytest.raises(ValueError, match='between 1 and worms'):
        validate_selfplay_layout(config, population_size=24, training_games=8)


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
