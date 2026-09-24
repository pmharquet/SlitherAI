import dataclasses
import json
import pickle
import random
from types import SimpleNamespace
import neat
import numpy as np
import pytest
import torch
from slitherai.config import SimConfig
from slitherai.evaluation import aggregate_scores, play_episode, scenarios
from slitherai.evolution import WindowedStagnation
from slitherai.network import load_config
from slitherai.rewards import reward_score
from slitherai.sim import WorldBatch
from slitherai.train import Trainer


def small_config(**changes):
    return dataclasses.replace(SimConfig(maps=2, worms=4, foods=32, body_points=16, preys=0), **changes)


def test_reward_rewards_growth_and_does_not_pay_for_mutual_suicide_or_recycled_boost():
    food = torch.tensor([0., 0., 50., 20.])
    spent = torch.tensor([0., 0., 0., 20.])
    age = torch.tensor([0., 90., 90., 0.])
    kills = torch.tensor([1., 0., 0., 0.])
    alive = torch.tensor([False, True, True, True])
    scores = reward_score(food, spent, age, kills, alive)
    assert scores[0] < 0
    assert scores[2] > 10*scores[1]
    assert scores[3] == 0
    assert reward_score(food, spent, age, kills, alive, 'legacy-v1')[0] == 10


def test_legacy_simulation_configs_keep_their_original_reward():
    assert SimConfig.from_dict({'maps':2}).reward_version == 'legacy-v1'
    old = small_config()
    del old.__dict__['reward_version']
    assert pickle.loads(pickle.dumps(old)).reward_version == 'legacy-v1'


def test_fixed_cases_do_not_drift_and_extreme_scores_have_less_influence():
    a, b = scenarios(5, 0, 16), scenarios(5, 12, 16)
    assert a[:4] == b[:4] and a[4] != b[4]
    score, anchor = aggregate_scores([[0,0,0,100,0], [20,20,20,20,20]])
    assert score[1] > score[0]
    assert anchor.tolist() == [12.5,20]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_private_scenarios_are_identical_across_candidate_slots_and_batch_sizes(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('No CUDA')
    torch.set_num_threads(2)
    random.seed(22)
    config = load_config(4)
    genomes = list(neat.Population(config).population.values())
    a = play_episode(genomes[:2], config, small_config(), device, 937, 1, 1., shared_random=True)
    b = play_episode([genomes[1]], config, small_config(maps=1), device, 937, 1, 1., shared_random=True)
    c = play_episode(genomes[1::-1], config, small_config(), device, 937, 1, 1., shared_random=True)
    for key in a:
        assert a[key][1] == pytest.approx(b[key][0], abs=1e-5)
        np.testing.assert_allclose(a[key], c[key][::-1], atol=1e-5)


def test_old_training_run_is_not_resumed_with_incompatible_fitness(tmp_path):
    from fastapi.testclient import TestClient
    from slitherai import server
    from unittest.mock import patch
    (tmp_path/'settings.json').write_text(json.dumps({'config':{}}))
    (tmp_path/'checkpoint-50').write_bytes(b'not deserialized')
    with patch.object(server, 'current', tmp_path), patch.object(server, 'process', None):
        with patch.object(server.subprocess, 'Popen', side_effect=AssertionError('Must not start')):
            response = TestClient(server.app).post('/api/start', json={'resume':True})
    assert response.status_code == 400 and 'barème' in response.json()['detail']


def test_truncating_inactive_body_segments_preserves_sensors_and_physics():
    config = small_config(body_points=96)
    short, full = WorldBatch(config, seed=891), WorldBatch(config, seed=891)
    full.active_body_points = lambda: config.body_points-1
    short.mass[:, 2] = full.mass[:, 2] = 420
    short.alive[0, 3] = full.alive[0, 3] = False
    rng = np.random.default_rng(4)
    for _ in range(5):
        torch.testing.assert_close(short.observe(), full.observe(), atol=1e-6, rtol=1e-6)
        actions = torch.tensor(rng.random((8,2)), dtype=torch.float32)
        short.step(actions); full.step(actions)
    for key in ('head','body','alive','gained','kills','mass','age'):
        torch.testing.assert_close(getattr(short,key),getattr(full,key))


def species_set(scores, history=None, last_improved=0):
    result = SimpleNamespace(species={})
    for sid, values in enumerate(scores, 1):
        species = neat.species.Species(sid, 0)
        species.members = {i:SimpleNamespace(fitness=value, anchor_fitness=value) for i,value in enumerate(values)}
        species.fitness_history = list(history or [])
        species.last_improved = last_improved
        result.species[sid] = species
    return result


def test_stagnation_tracks_recent_progress_despite_an_old_lucky_record():
    config = load_config(4).stagnation_config
    tracker = WindowedStagnation(config, neat.reporting.ReporterSet())
    species = species_set([[0]])
    for gen,value in enumerate([100,1,1,1,1,2,2,2,2,2]):
        species.species[1].members[0].anchor_fitness = value
        tracker.update(species, gen)
    assert species.species[1].last_improved == 9
    assert species.species[1].recent_score == 2


def test_best_mean_species_is_protected_and_removals_are_limited():
    config = load_config(4).stagnation_config
    config.species_elitism = 1
    tracker = WindowedStagnation(config, neat.reporting.ReporterSet())
    species = species_set([[100,0], [70,70], [10,10]], history=[1000]*10)
    result = tracker.update(species, 40)
    assert species.species[2].protected
    assert sum(stagnant for _,_,stagnant in result) == 1
    assert next(stagnant for sid,_,stagnant in result if sid == 2) is False


def test_adaptive_speciation_targets_initial_diversity_and_respects_reproduction_capacity():
    random.seed(125)
    config = load_config(64)
    population = neat.Population(config)
    assert 8 <= len(population.species.species) <= 12
    before = config.species_set_config.compatibility_threshold
    config.species_set_config.target_min = 13
    config.species_set_config.target_max = 16
    population.species.speciate(config, population.population, 1)
    assert config.species_set_config.compatibility_threshold < before
    config.species_set_config.compatibility_threshold = .001
    config.species_set_config.threshold_min = .001
    population.species.speciate(config, population.population, 2)
    assert len(population.species.species)*config.reproduction_config.min_species_size <= 64
    assert set(population.species.genome_to_species) == set(population.population)


def test_v2_training_resume_matches_uninterrupted_evolution(tmp_path):
    config = small_config(foods=8)
    whole, split = tmp_path/'whole', tmp_path/'split'
    Trainer(config, whole, 'cpu', seed=123, validation_every=0).train(4, 2, .2)
    Trainer(config, split, 'cpu', seed=123, validation_every=0).train(4, 1, .2)
    Trainer(config, split, 'cpu', seed=123, validation_every=0).train(4, 1, .2, split/'checkpoint-1')
    a = neat.Checkpointer.restore_checkpoint(str(whole/'checkpoint-2'))
    b = neat.Checkpointer.restore_checkpoint(str(split/'checkpoint-2'))
    assert a.species.genome_to_species == b.species.genome_to_species
    assert a.config.species_set_config.compatibility_threshold == b.config.species_set_config.compatibility_threshold
    for key in a.population:
        ga, gb = a.population[key], b.population[key]
        assert ga.fitness == gb.fitness
        assert [(k,c.weight,c.enabled) for k,c in ga.connections.items()] == [(k,c.weight,c.enabled) for k,c in gb.connections.items()]
    episodes = json.loads((split/'episodes'/'generation-0001.json').read_text())
    assert len(episodes['metrics']['fitness']) == 4 and len(episodes['metrics']['fitness'][0]) == 5
    preview = json.loads((split/'preview.json').read_text())
    assert sum(w['controller']=='neat' for w in preview['worms']) == 1
    assert preview['network']['genome_id'] in episodes['genome_ids']
