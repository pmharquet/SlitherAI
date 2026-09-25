import dataclasses
import math
import random

import neat
import numpy as np
import pytest
import torch

from slitherai.config import SimConfig
from slitherai.evaluation import play_population_episode
from slitherai.network import BatchedNetwork, load_config
from slitherai.sim import WorldBatch


def small_config(**changes):
    base = SimConfig(maps=2, worms=3, foods=12, body_points=8, preys=0,
                     arena_radius=400., arena_variation=0.)
    return dataclasses.replace(base, **changes)


def small_population():
    random.seed(481516)
    neat_config = load_config(3)
    population = neat.Population(neat_config)
    return neat_config, list(population.population.values())


def manual_population_episode(genomes, neat_config, config, seed, assignment, seconds):
    """Reference loop that spells out the all-cell observation/action mapping."""
    world = WorldBatch(config, 'cpu', seed, shared_random=True)
    network = BatchedNetwork(genomes, neat_config, assignment=assignment, device='cpu')
    slots = config.maps * config.worms
    with torch.inference_mode():
        for _ in range(round(seconds / config.dt)):
            observations = world.observe().reshape(slots, -1)
            world.step(network.activate(observations))
            if not bool(world.alive.any()):
                break
    values = dict(fitness=world.fitness(), food_gain=world.gained, boost_spent=world.spent,
                  alive=world.alive, age=world.age, kills=world.kills,
                  border_death=world.border_deaths, collision_death=world.collision_deaths,
                  boost_fraction=world.boost_steps / world.decisions.clamp_min(1),
                  turn_degrees=world.turn_sum / world.decisions.clamp_min(1) * (180 / math.pi))
    values.update({f'reward_{key}': value for key, value in world.fitness_terms().items()})
    return {key: value.float().reshape(-1).tolist() for key, value in values.items()}


def test_population_episode_maps_all_cells_and_returns_flat_metrics_and_preview_state():
    neat_config, genomes = small_population()
    config = small_config()
    assignment = [2, 0, 1, 1, 2, 0]
    callbacks = []

    def on_tick(world, network, observations_flat, callback_assignment, step):
        callbacks.append((observations_flat.shape, list(callback_assignment), step))
        assert network.assignment == assignment
        assert len(callback_assignment) == config.maps * config.worms
        for slot, genome_index in enumerate(assignment):
            preview = network.describe(slot, observations_flat)
            assert preview['genome_id'] == genomes[genome_index].key
        assert world.c.maps == config.maps
        assert observations_flat.shape == (config.maps * config.worms, 530)

    result = play_population_episode(genomes, neat_config, config, 'cpu', 7821,
                                     assignment, .2, on_tick=on_tick)

    assert callbacks
    assert set(result) == {
        'fitness', 'food_gain', 'boost_spent', 'alive', 'age', 'kills',
        'border_death', 'collision_death', 'boost_fraction', 'turn_degrees',
        'reward_growth', 'reward_survival', 'reward_kills', 'reward_death',
    }
    assert all(len(values) == config.maps * config.worms for values in result.values())
    assert all(np.isfinite(values).all() for values in map(np.asarray, result.values()))


def test_population_episode_matches_manual_world_network_loop():
    neat_config, genomes = small_population()
    config = small_config()
    assignment = [2, 0, 1, 1, 2, 0]

    actual = play_population_episode(genomes, neat_config, config, 'cpu', 1937,
                                     assignment, .3)
    expected = manual_population_episode(genomes, neat_config, config, 1937,
                                         assignment, .3)

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(('assignment', 'message'), [
    ([0, 0, 0, 0, 0], 'one index per'),
    ([0, 0, 0, 0, 0, -1], 'existing genome'),
    ([0, 0, 0, 0, 0, 3], 'existing genome'),
    ([0, 0, 0, 0, 0, 0.0], 'integer genome indices'),
    ([[0, 0, 0], [0, 0, 0]], 'one-dimensional'),
])
def test_population_episode_rejects_invalid_assignment(assignment, message):
    neat_config, genomes = small_population()
    with pytest.raises(ValueError, match=message):
        play_population_episode(genomes, neat_config, small_config(), 'cpu', 9,
                                assignment, .1)


def test_population_episode_requires_at_least_one_genome():
    neat_config, _ = small_population()
    with pytest.raises(ValueError, match='At least one NEAT genome'):
        play_population_episode([], neat_config, small_config(), 'cpu', 9,
                                [0] * 6, .1)
