import math

import pytest
import torch

from slitherai.config import SimConfig
from slitherai.schema import ANGLES_DEG
from slitherai.sim import WorldBatch


def make_world(view=3000.0):
    config = SimConfig(maps=1, worms=2, foods=8, preys=0, body_points=16,
                       arena_radius=1000.0, arena_variation=0.0,
                       view_half_width=view, view_half_height=view)
    world = WorldBatch(config, device="cpu", seed=53)
    world.mass[0, 0] = 35.0
    world.alive[0, 1] = False
    world.food_mass.zero_()
    return world


def body_radius_for_mass(mass):
    """Evaluate the documented local radius curve independently of WorldBatch.radius."""
    return 14.5 * (1.0 + mass / 300.0) ** 0.4


def border_channel(world):
    return world.observe()[0, :522].reshape(87, 6)[:, 4].clone()


def center_ray_index():
    return ANGLES_DEG.index(0)


def test_center_border_distance_is_proximity_normalized_and_rotation_invariant():
    world = make_world()
    radius = body_radius_for_mass(35.0)
    sensor_circle_radius = 1000.0 - radius - 3.0
    expected = 1.0 / (1.0 + sensor_circle_radius / 100.0)

    world.head[0, 0] = torch.tensor([0.0, 0.0])
    world.heading[0, 0] = 0.0
    east_oriented = border_channel(world)
    world.heading[0, 0] = math.pi / 2
    north_oriented = border_channel(world)

    torch.testing.assert_close(east_oriented, torch.full((87,), expected), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(north_oriented, east_oriented, atol=2e-6, rtol=2e-6)


def test_near_edge_ray_reports_three_units_inside_the_lethal_circle():
    world = make_world()
    radius = float(world.radius[0, 0])
    sensor_circle_radius = 1000.0 - radius - 3.0
    start_x = sensor_circle_radius - 20.0
    world.head[0, 0] = torch.tensor([start_x, 0.0])
    world.heading[0, 0] = 0.0

    values = border_channel(world)
    center = center_ray_index()
    assert ANGLES_DEG[center] == 0
    assert float(values[center]) == pytest.approx(1.0 / 1.2, abs=2e-6)

    actual_outward_distance = 1000.0 - radius - float(world.head[0, 0, 0])
    sensed_outward_distance = sensor_circle_radius - float(world.head[0, 0, 0])
    assert sensed_outward_distance == pytest.approx(20.0, abs=2e-5)
    assert actual_outward_distance - sensed_outward_distance == pytest.approx(3.0, abs=2e-5)

    # Rotating both the head position and heading preserves each relative-ray value.
    world.head[0, 0] = torch.tensor([0.0, start_x])
    world.heading[0, 0] = math.pi / 2
    rotated_values = border_channel(world)
    torch.testing.assert_close(rotated_values, values, atol=2e-6, rtol=2e-6)

    # The local collision rule kills at the larger circle, three units beyond the sensor circle.
    world.head[0, 0] = torch.tensor([1000.0 - radius, 0.0])
    world._collisions()
    assert not bool(world.alive[0, 0])
    assert bool(world.border_deaths[0, 0])


def test_view_cutoff_censors_far_border_and_away_facing_ray():
    world = make_world(view=100.0)
    radius = float(world.radius[0, 0])
    sensor_circle_radius = 1000.0 - radius - 3.0
    center = center_ray_index()

    world.head[0, 0] = torch.tensor([0.0, 0.0])
    world.heading[0, 0] = 0.0
    center_values = border_channel(world)
    assert float(center_values[center]) == 0.0  # The wall is outside this ray's view range.

    start_x = sensor_circle_radius - 20.0
    world.head[0, 0] = torch.tensor([start_x, 0.0])
    world.heading[0, 0] = 0.0
    toward_edge = border_channel(world)
    assert float(toward_edge[center]) == pytest.approx(1.0 / 1.2, abs=2e-6)

    # With the same near-edge head, the forward ray now faces across the arena.
    world.heading[0, 0] = math.pi
    away_from_edge = border_channel(world)
    away_distance = sensor_circle_radius + start_x
    assert away_distance > 100.0 * math.sqrt(radius / 14.5)
    assert float(away_from_edge[center]) == 0.0

