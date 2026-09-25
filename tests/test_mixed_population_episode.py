import dataclasses
import random

import neat
import numpy as np
import pytest

from slitherai.config import SimConfig
from slitherai.evaluation import play_episode, play_mixed_population_episode
from slitherai.network import load_config


def small_config(**changes):
    base = SimConfig(maps=2, worms=3, foods=12, body_points=8, preys=0,
                     arena_radius=400., arena_variation=0.)
    return dataclasses.replace(base, **changes)


def small_population():
    random.seed(481516)
    neat_config = load_config(3)
    population = neat.Population(neat_config)
    return neat_config, list(population.population.values())


def test_mixed_episode_matches_private_focal_games_against_the_same_heuristic():
    neat_config, genomes = small_population()
    config = small_config()
    focal = [1, 2]
    assignment = [-1] * (config.maps * config.worms)
    for arena, slot in enumerate(focal):
        assignment[arena * config.worms + slot] = arena

    mixed = play_mixed_population_episode(genomes[:2], neat_config, config, 'cpu', 1937,
                                          assignment, .3, shared_random=True)
    private = play_episode(genomes[:2], neat_config, config, 'cpu', 1937, focal, .3,
                           shared_random=True)

    for key, values in private.items():
        candidate_values = [mixed[key][arena * config.worms + slot]
                            for arena, slot in enumerate(focal)]
        np.testing.assert_allclose(candidate_values, values, rtol=1e-6, atol=1e-6)
    assert all(len(values) == config.maps * config.worms for values in mixed.values())


def test_mixed_assignment_compacts_candidate_slots_and_preserves_genome_mapping():
    neat_config, genomes = small_population()
    config = small_config()
    assignment = [0, -1, 1, 2, -1, 0]
    callbacks = []

    def on_tick(world, network, observations_flat, callback_assignment, step):
        callbacks.append(step)
        assert callback_assignment == assignment
        assert observations_flat.shape == (config.maps * config.worms, 530)
        assert network.world_slots == [0, 2, 3, 5]
        assert network.world_to_network_slot == [0, -1, 1, 2, -1, 3]
        assert network.assignment == [0, 1, 2, 0]
        candidate_observations = observations_flat[network.world_slots]
        for network_slot, genome_index in enumerate(network.assignment):
            assert network.describe(network_slot, candidate_observations)['genome_id'] == genomes[genome_index].key
        assert world.c.maps == config.maps

    result = play_mixed_population_episode(genomes, neat_config, config, 'cpu', 7821,
                                           assignment, .2, on_tick=on_tick)

    assert callbacks
    assert set(result) == {
        'fitness', 'food_gain', 'boost_spent', 'alive', 'age', 'kills',
        'border_death', 'collision_death', 'boost_fraction', 'turn_degrees',
        'reward_growth', 'reward_survival', 'reward_kills', 'reward_death',
    }
    assert all(np.isfinite(values).all() for values in map(np.asarray, result.values()))


def test_permuting_genome_list_and_remapping_indices_preserves_each_slot_result():
    neat_config, genomes = small_population()
    config = small_config()
    assignment = [0, -1, 1, 2, -1, 0]
    order = [2, 0, 1]
    new_index_by_old = {old_index: new_index for new_index, old_index in enumerate(order)}
    permuted_assignment = [-1 if index == -1 else new_index_by_old[index]
                           for index in assignment]

    original = play_mixed_population_episode(genomes, neat_config, config, 'cpu', 2718,
                                             assignment, .3, shared_random=True)
    permuted = play_mixed_population_episode([genomes[index] for index in order], neat_config,
                                             config, 'cpu', 2718, permuted_assignment, .3,
                                             shared_random=True)

    for key in original:
        np.testing.assert_allclose(original[key], permuted[key], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(('assignment', 'message'), [
    ([-1, -1, -1, -1, 0], 'one entry per'),
    ([0, -1, 0, -1, 0, 0.0], 'integer genome indices'),
    ([0, -1, 0, -1, 0, True], 'integer genome indices'),
    ([0, -1, 0, -1, 0, -2], 'refer to an existing genome'),
    ([0, -1, 0, -1, 0, 3], 'refer to an existing genome'),
    ([[0, -1, 0], [-1, 0, 0]], 'one-dimensional'),
    ([-1, -1, -1, -1, -1, -1], 'At least one slot'),
])
def test_mixed_episode_rejects_invalid_assignment(assignment, message):
    neat_config, genomes = small_population()
    with pytest.raises(ValueError, match=message):
        play_mixed_population_episode(genomes, neat_config, small_config(), 'cpu', 9,
                                      assignment, .1)


def test_mixed_episode_requires_a_candidate_genome():
    neat_config, _ = small_population()
    with pytest.raises(ValueError, match='At least one NEAT genome'):
        play_mixed_population_episode([], neat_config, small_config(), 'cpu', 9,
                                      [0] * 6, .1)
