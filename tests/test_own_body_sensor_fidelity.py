"""CPU characterization of own-body sensing in the extension and simulator."""
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from slitherai.config import SimConfig
from slitherai.schema import INPUTS
from slitherai.sim import WorldBatch


ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name('own_body_sensor_probe.cjs')
SPACING_INDEX_RAY = 0  # schema ray 0 is relative angle -166 degrees


def _path_point(case, distance):
    if case == 'straight':
        return (-distance, 0.0)
    if case == 'tight_coil':
        if distance <= 27.0:
            return (-distance, 0.0)
        angle = math.pi + (distance - 27.0) / 2.0
        return (-25.0 + 2.0 * math.cos(angle), 2.0 * math.sin(angle))
    raise AssertionError(case)


def _synthetic_observations(case):
    config = SimConfig(
        maps=1, worms=2, foods=1, preys=0, body_points=96,
        arena_radius=100000.0, arena_variation=0.0,
        view_half_width=2000.0, view_half_height=2000.0,
        initial_mass=35.0,
    )
    world = WorldBatch(config, device='cpu', seed=17)
    world.mass.fill_(35.0)
    world.head[0, 0] = torch.tensor([0.0, 0.0])
    world.head[0, 1] = torch.tensor([10000.0, 0.0])
    world.heading.zero_()
    world.food_mass.zero_()
    radius = float(world.radius[0, 0])
    spacing = float(world.spacing[0, 0])
    points = [_path_point(case, index * spacing) for index in range(config.body_points)]
    world.body[0, 0].copy_(torch.tensor(points, dtype=world.body.dtype))
    world.body[0, 1, :, 0] = 10000.0 - torch.arange(config.body_points) * spacing
    world.body[0, 1, :, 1] = 0.0

    output = world.observe()[0]
    python_value = float(output[SPACING_INDEX_RAY * 6 + 3])
    assert output.numel() == INPUTS
    scenario = {
        'radius': radius,
        # Extension pts excludes the head, matching the simulator's body[1:].
        'points': points[1:],
    }
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required to execute the extension sensor')
    completed = subprocess.run(
        [node, str(PROBE)], input=json.dumps(scenario), text=True,
        capture_output=True, cwd=ROOT, check=True,
    )
    js = json.loads(completed.stdout)
    assert js['status'] == 'ok'
    return radius, spacing, python_value, js['ray']['selfBodyDistance']


def _endpoint_circle_entry(distance, ray_radius):
    # Ray angle -166°; the center is on the negative x axis.
    forward = distance * math.cos(math.radians(14.0))
    sideways = distance * math.sin(math.radians(14.0))
    return forward - math.sqrt(ray_radius**2 - sideways**2)


def test_straight_body_uses_different_radii_and_cutoffs():
    radius, spacing, python_value, js_distance = _synthetic_observations('straight')

    # JS accepts the sample at 4*spacing (head distance >= 2r), with a 2r+3
    # circle. Python first accepts a capsule whose arc index is >= 3r.
    js_expected = _endpoint_circle_entry(4 * spacing, 2 * radius + 3)
    python_first_segment = math.ceil(3 * radius / spacing)
    python_expected = _endpoint_circle_entry(python_first_segment * spacing, radius)

    assert js_distance == pytest.approx(js_expected, abs=1e-5)
    assert python_value == pytest.approx(1 / (1 + python_expected / 100), abs=2e-5)
    assert js_distance < 5.0
    assert python_expected > 40.0


def test_tight_coil_distinguishes_head_distance_from_arc_distance():
    radius, _spacing, python_value, js_distance = _synthetic_observations('tight_coil')

    # Every synthetic centerline point is < 2r from the head, so the extension
    # suppresses all own-body geometry. The simulator includes the late-arc
    # capsules once their sample index reaches 3r.
    assert 27.0 < 2 * radius
    assert js_distance is None
    assert python_value > 0.0
