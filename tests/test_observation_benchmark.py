import torch

from slitherai.config import SimConfig
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
