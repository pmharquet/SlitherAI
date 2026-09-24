"""One input/output contract for simulation and recorded games."""
import math
import numpy as np

VERSION = 'slither-neat-530-v1'
ANGLES_DEG = list(range(-166, -93, 8)) + list(range(-88, -47, 4)) + list(range(-44, 45, 2)) + list(range(48, 89, 4)) + list(range(94, 167, 8))
ANGLES = np.deg2rad(ANGLES_DEG).astype(np.float32)
CHANNELS = ['visible_range', 'enemy_head', 'enemy_body', 'self_body', 'border', 'food_value']
GLOBALS = ['speed', 'radius', 'length', 'heading_sin', 'heading_cos', 'turn_error_sin', 'turn_error_cos', 'previous_boost']
INPUTS = len(ANGLES) * len(CHANNELS) + len(GLOBALS)
OUTPUTS = ['boost_probability', 'absolute_direction_turns']
DISTANCE_SCALE = 100.0

def proximity(distance):
    return 0.0 if distance is None else 1.0 / (1.0 + max(0.0, distance) / DISTANCE_SCALE)

def food_value(size, distance):
    return min(max(size or 0.0, 0.0) / 20.0, 1.0) * proximity(distance)

def record_inputs(sample, previous_boost=False):
    """Convert a real JSONL sample. Missing physical state is rejected, never guessed."""
    lidar = sample['lidar']
    rays = lidar['rays']
    if len(rays) != len(ANGLES) or not np.allclose([r['relativeAngle'] for r in rays], ANGLES, atol=1e-5):
        raise ValueError('Expected the 87-ray layout from extension 0.6.0')
    channels = []
    for ray in rays:
        values = [ray['range'] / (ray['range'] + 1000.0), 0., 0., 0., 0., 0.]
        for hit in ray['returns']:
            kind = hit['kind']
            if kind in CHANNELS[1:5]:
                slot = CHANNELS.index(kind)
                values[slot] = max(values[slot], proximity(hit['distance']))
            elif kind in ('food', 'prey'):
                values[5] = max(values[5], food_value(hit['size'], hit['distance']))
        channels.extend(values)
    p = lidar['player']
    required = ['heading', 'wantedHeading', 'speedRaw', 'bodyRadiusEstimate', 'segmentCount', 'segmentFraction']
    if any(p.get(k) is None for k in required):
        raise ValueError('Missing player fields for input contract')
    heading = p['heading']
    delta = p['wantedHeading'] - heading
    channels.extend([min(p['speedRaw'] / 12., 1.), min(p['bodyRadiusEstimate'] / 80., 1.),
                     min((p['segmentCount'] + p['segmentFraction']) / 400., 1.),
                     math.sin(heading), math.cos(heading), math.sin(delta), math.cos(delta), float(previous_boost)])
    return np.asarray(channels, dtype=np.float32)

def record_outputs(sample):
    angle = sample['lidar']['player']['heading'] + sample['action']['steeringAngle']
    return np.asarray([float(sample['action']['boost']), (angle / math.tau) % 1.], dtype=np.float32)

def contract():
    return dict(version=VERSION, inputs=INPUTS, angles_degrees=ANGLES_DEG,
                ray_channels=CHANNELS, global_channels=GLOBALS, outputs=OUTPUTS,
                direction='0=east, 0.25=south, 0.5=west, 0.75=north; 1 wraps to 0',
                proximity='1/(1+distance/100), absent=0', food='min(size/20,1)*proximity(distance)')
