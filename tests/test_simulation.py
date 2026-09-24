import math
import random
import numpy as np
import pytest
import torch
import neat
from slitherai.config import SimConfig
from slitherai.geometry import ray_capsule, ray_circle
from slitherai.network import BatchedNetwork, load_config
from slitherai.schema import INPUTS, ANGLES, record_inputs, record_outputs
from slitherai.sim import WorldBatch

torch.set_num_threads(2)

def world(device='cpu', seed=1):
    return WorldBatch(SimConfig(maps=2, worms=4, foods=32, body_points=16, preys=0, arena_radius=1200), device, seed)

def separate(e):
    for worm in range(e.c.worms):
        e.head[:, worm] = torch.tensor([worm * 220., 0.], device=e.device)
        e.body[:, worm, :, 0] = worm * 220. - torch.arange(e.c.body_points, device=e.device) * 9
        e.body[:, worm, :, 1] = 0
    e.heading.zero_()
    e.target.zero_()
    e.food_mass.zero_()

def test_analytic_geometry():
    direction = torch.tensor([1., 0.])
    assert float(ray_circle(direction, torch.tensor([10., 0.]), torch.tensor(2.))) == pytest.approx(8)
    assert float(ray_capsule(direction, torch.tensor([10., -5.]), torch.tensor([10., 5.]), torch.tensor(2.))) == pytest.approx(8)
    assert torch.isinf(ray_capsule(direction, torch.tensor([-10., -5.]), torch.tensor([-10., 5.]), torch.tensor(2.)))

def test_shapes_finite_seed_and_independent_maps():
    a, b = world(), world()
    assert torch.equal(a.head, b.head)
    x = a.observe()
    assert x.shape == (8, 530) and torch.isfinite(x).all()
    before = x[4:].clone()
    a.food[0] += 90
    a.head[0] += 10
    assert torch.allclose(a.observe()[4:], before)

def test_food_is_awarded_once_and_grows_the_winner():
    e = world(); separate(e)
    e.head[:, 0] = torch.tensor([0., 0.])
    e.head[:, 1] = torch.tensor([20., 0.])
    e.food[:, 0] = torch.tensor([10., 0.])
    e.food_mass[:, 0] = 3
    before = e.mass.clone()
    e._eat()
    assert torch.allclose((e.mass - before).sum(1), torch.full((2,), 3.))
    assert torch.allclose(e.gained.sum(1), torch.full((2,), 3.))

def test_self_body_is_safe_enemy_body_and_border_kill():
    e = world(); separate(e)
    e.body[:, 0, 3] = e.head[:, 0]
    e._collisions()
    assert e.alive.all()
    e.head[:, 0] = e.body[:, 1, 3]
    e._collisions()
    assert not e.alive[:, 0].any()
    assert (e.kills[:, 1] == 1).all()
    assert (e.food_mass[:, e.c.foods:e.c.foods+e.c.body_points].sum(1) > 0).all()
    e.head[:, 2, 0] = e.arena
    e._collisions()
    assert not e.alive[:, 2].any()

def test_boost_cost_turn_limit_and_wrapped_direction():
    a, b = world(), world(); separate(a); separate(b)
    initial = a.mass.clone()
    action = torch.zeros((8, 2)); action[:, 0] = 1
    a.step(action)
    action[:, 1] = 1
    b.step(action)
    assert torch.allclose(a.head, b.head)
    assert torch.all(a.mass < initial)
    assert (a.speed >= a.c.boost_speed).all()
    action[:, 1] = .5
    old = a.heading.clone(); a.step(action)
    assert torch.all((a.heading-old).abs() <= a.c.turn_rate * a.c.dt + 1e-5)

def test_blind_cone_excludes_food_but_front_food_is_encoded():
    e = world(); separate(e)
    e.food[:, 0] = torch.tensor([-100., 0.]); e.food_mass[:, 0] = 2; e.food_size[:, 0] = 6
    assert float(e.observe()[0, :522].reshape(87, 6)[:, 5].max()) == 0
    e.food[:, 0, 0] = 100
    obs = e.observe()[0, :522].reshape(87, 6)
    assert float(obs[43, 5]) == pytest.approx(.3 / (1 + .94), abs=1e-6)
    assert torch.count_nonzero(obs[:, 5]) == 1

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_neat_recurrent_cuda_matches_reference_with_hidden_nodes(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('No CUDA')
    random.seed(123)
    config = load_config(4)
    population = neat.Population(config)
    genomes = list(population.population.values())
    for genome in genomes:
        genome.mutate_add_node(config.genome_config)
        genome.mutate_add_node(config.genome_config)
    reference = [neat.nn.RecurrentNetwork.create(g, config) for g in genomes]
    network = BatchedNetwork(genomes, config, device=device)
    rng = np.random.default_rng(321)
    for _ in range(8):
        values = rng.uniform(-1, 1, (4, INPUTS)).astype(np.float32)
        expected = np.array([n.activate(v) for n, v in zip(reference, values)])
        actual = network.activate(torch.tensor(values, device=device)).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6)

def test_jsonl_converter_contract_and_absolute_output():
    sample = dict(lidar=dict(rays=[dict(relativeAngle=float(a), range=500., returns=[]) for a in ANGLES],
                            player=dict(heading=math.pi/2, wantedHeading=math.pi/2, speedRaw=6, bodyRadiusEstimate=15, segmentCount=4, segmentFraction=.2)),
                  action=dict(boost=True, steeringAngle=math.pi/2))
    sample['lidar']['rays'][43]['returns'] = [dict(kind='food', size=10, distance=100)]
    vector = record_inputs(sample, previous_boost=False)
    assert vector.shape == (INPUTS,)
    assert vector[43*6+5] == pytest.approx(.25)
    assert vector[-1] == 0
    np.testing.assert_allclose(record_outputs(sample), [1., .5])

def test_checkpoint_restores_rng_population_and_innovation_tracker(tmp_path):
    random.seed(44)
    config = load_config(4)
    population = neat.Population(config)
    reporter = neat.Checkpointer(1, filename_prefix=str(tmp_path / 'checkpoint-'))
    reporter.save_checkpoint(config, population.population, population.species, 0)
    expected_random = random.random()
    restored = neat.Checkpointer.restore_checkpoint(str(tmp_path / 'checkpoint-0'))
    assert random.random() == expected_random
    assert set(restored.population) == set(population.population)
    assert restored.config.genome_config.innovation_tracker is restored.reproduction.innovation_tracker

def test_network_inspector_reports_selected_genome_and_actual_recurrent_state():
    config = load_config(4)
    population = neat.Population(config)
    genomes = list(population.population.values())
    genomes[2].mutate_add_node(config.genome_config)
    network = BatchedNetwork(genomes, config, assignment=[2, 0, 1, 2])
    inputs = torch.rand(4, INPUTS)
    network.activate(inputs)
    previous = network.state.clone()
    network.activate(inputs * .5)
    data = network.describe(0, inputs * .5)
    assert data['genome_id'] == genomes[2].key
    assert data['inputs'] == (inputs[0] * .5).tolist()
    for i, node in enumerate(data['nodes']):
        assert node['value'] == pytest.approx(float(network.state[0, i]))
        assert node['previous_value'] == pytest.approx(float(previous[0, i]))
    assert all(e['delayed'] == (e['source'] >= 0) for e in data['connections'])

def test_live_preview_selection_stays_aligned_with_arena_and_assignment(tmp_path):
    import json
    from slitherai.train import Trainer
    config = load_config(4)
    population = neat.Population(config)
    genomes = list(population.population.values())
    env = world()
    assignment = np.array([0,1,2,3,3,2,1,0])
    network = BatchedNetwork(genomes, config, assignment)
    observations = env.observe()
    env.step(network.activate(observations))
    trainer = Trainer(env.c, tmp_path, 'cpu')
    trainer.live_world, trainer.live_network = env, network
    trainer.live_observations, trainer.live_assignment = observations, assignment
    trainer.publish_preview({'arena': 1, 'network_worm': 1})
    data = json.loads((tmp_path / 'preview.json').read_text())
    assert data['network']['worm'] == 1 and data['network']['arena'] == data['arena'] == 1
    assert data['network']['genome_id'] == genomes[2].key
    assert len(data['network']['inputs']) == 530
    assert data['network']['observation_time'] == 0

def test_dashboard_get_does_not_start_a_process(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from slitherai import server
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    monkeypatch.setattr(server, 'current', None)
    monkeypatch.setattr(server, 'process', None)
    def forbidden(*args, **kwargs): raise AssertionError('No training process should be started')
    monkeypatch.setattr(server.subprocess, 'Popen', forbidden)
    client = TestClient(server.app)
    assert client.get('/').status_code == 200
    assert client.get('/assets/network-view.js').status_code == 200
    response = client.get('/api/state').json()
    assert not response['active'] and not response['can_resume']
    assert client.post('/api/start', json=dict(maps=1, worms=1, population=4)).status_code == 422
