"""Comparable private games against fixed opponents, batched on CUDA."""
import dataclasses
import math
import time
import numpy as np
import torch
from .network import BatchedNetwork
from .schema import ANGLES
from .sim import WorldBatch
from .protocol import PROTOCOL, protocol_settings


def scenarios(seed, generation, worms):
    anchors = [dict(seed=seed+offset, focal=position % worms, anchor=True)
               for offset, position in zip((100003, 200003, 300007, 400009), (0, 5, 10, 15))]
    return anchors + [dict(seed=seed+10000019+generation*1009, focal=generation % worms, anchor=False)]


def aggregate_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 5 or not np.isfinite(scores).all():
        raise ValueError('Expected five finite game scores per genome')
    anchor = .5*scores[:, :4].mean(1)+.5*np.median(scores[:, :4], axis=1)
    return .8*anchor+.2*scores[:, 4], anchor


def heuristic(observations):
    x = observations[:, :522].reshape(-1, 87, 6)
    danger = torch.maximum(torch.maximum(x[..., 1], x[..., 2]), x[..., 4])
    angles = torch.as_tensor(ANGLES, device=x.device)
    choice = (x[..., 5]*2-danger*5+angles.cos()*.15).argmax(-1)
    heading = torch.atan2(observations[:, 525], observations[:, 526])
    action = torch.zeros((len(x), 2), device=x.device)
    action[:, 1] = ((heading+angles[choice])/math.tau).remainder(1)
    return action


def episode_metrics(world, focal):
    rows = torch.arange(world.c.maps, device=world.device)
    def take(values): return values[rows, focal].float()
    values = _episode_metric_values(world)
    array = torch.stack([take(value) for value in values.values()], -1).cpu().numpy()
    return {key:array[:, i].tolist() for i,key in enumerate(values)}


def _episode_metric_values(world):
    values = dict(fitness=world.fitness(), food_gain=world.gained, boost_spent=world.spent,
                  alive=world.alive, age=world.age, kills=world.kills,
                  border_death=world.border_deaths, collision_death=world.collision_deaths,
                  boost_fraction=world.boost_steps/world.decisions.clamp_min(1),
                  turn_degrees=world.turn_sum/world.decisions.clamp_min(1)*(180/math.pi))
    values.update({f'reward_{key}':value for key,value in world.fitness_terms().items()})
    return values


def population_episode_metrics(world):
    """Return one metric value per map/worm cell, in map-major order."""
    values = _episode_metric_values(world)
    array = torch.stack([value.float().reshape(-1) for value in values.values()], -1).cpu().numpy()
    return {key:array[:, i].tolist() for i,key in enumerate(values)}


@torch.inference_mode()
def play_episode(genomes, neat_config, config, device, seed, focal, seconds,
                 shared_random=False, on_tick=None, controls=None, policy='neat'):
    world = WorldBatch(config, device, seed, shared_random=shared_random)
    rows = torch.arange(config.maps, device=device)
    focal = torch.as_tensor(focal, device=device, dtype=torch.long).expand(config.maps)
    network = BatchedNetwork(genomes, neat_config, device=device) if genomes else None
    if network and len(genomes) != config.maps:
        raise ValueError('Exactly one candidate network per private arena is required')
    published = -math.inf
    for step in range(round(seconds/config.dt)):
        observations = world.observe().reshape(config.maps, config.worms, -1)
        inputs = observations[rows, focal]
        actions = heuristic(observations.flatten(0, 1)).reshape(config.maps, config.worms, 2)
        if network:
            actions[rows, focal] = network.activate(inputs)
        elif policy == 'circle':
            actions[rows, focal, 0] = 0
            actions[rows, focal, 1] = ((world.heading[rows, focal]+.8)/math.tau).remainder(1)
        world.step(actions)
        now = time.perf_counter()
        if now-published > 1.:
            if controls: controls()
            if on_tick: on_tick(world, network, inputs, focal, step)
            published = now
        if not bool(world.alive[rows, focal].any()): break
    if on_tick: on_tick(world, network, inputs, focal, step)
    if controls: controls()
    return episode_metrics(world, focal)


@torch.inference_mode()
def play_population_episode(genomes, neat_config, config, device, seed, assignment, seconds,
                            shared_random=True, on_tick=None, controls=None):
    """Run one arena batch where every map/worm cell is controlled by a NEAT genome.

    This is the population/self-play path; it does not alter the
    anchor/rotating-game behavior of :func:`play_episode`.
    ``assignment`` contains one genome-list index per cell in map-major/worm-minor
    order. Results use that same flattened order. ``on_tick`` follows the live
    preview callback contract: ``(world, network, observations_flat, assignment,
    step)``. ``observations_flat`` has shape ``(maps * worms, inputs)`` and the
    assignment vector is exactly the one used to build the batched network.
    """
    if not genomes:
        raise ValueError('At least one NEAT genome is required')
    config.validate()
    try:
        assignment_values = list(assignment)
    except TypeError as exc:
        raise ValueError('assignment must be a one-dimensional integer vector') from exc
    try:
        assignment_array = np.asarray(assignment_values)
    except (TypeError, ValueError) as exc:
        raise ValueError('assignment must be a one-dimensional integer vector') from exc
    if assignment_array.ndim != 1:
        raise ValueError('assignment must be a one-dimensional integer vector')
    slots = config.maps * config.worms
    if len(assignment_values) != slots:
        raise ValueError(f'assignment must contain one index per map/worm cell ({slots})')
    if (assignment_array.dtype.kind not in ('i', 'u')
            or any(isinstance(index, (bool, np.bool_)) for index in assignment_values)):
        raise ValueError('assignment entries must be integer genome indices')
    if np.any(assignment_array < 0) or np.any(assignment_array >= len(genomes)):
        raise ValueError('assignment indices must refer to an existing genome')
    assignment_values = assignment_array.astype(np.int64, copy=False)
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError('seconds must be finite and positive') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    steps = round(seconds / config.dt)
    if steps < 1:
        raise ValueError('seconds must include at least one simulation step')

    world = WorldBatch(config, device, seed, shared_random=shared_random)
    network = BatchedNetwork(genomes, neat_config, assignment=assignment_values, device=device)
    published = -math.inf
    observations_flat = None
    for step in range(steps):
        observations_flat = world.observe().reshape(slots, -1)
        actions = network.activate(observations_flat)
        world.step(actions)
        now = time.perf_counter()
        if now - published > 1.:
            if controls:
                controls()
            if on_tick:
                on_tick(world, network, observations_flat, network.assignment, step)
            published = now
        if not bool(world.alive.any()):
            break
    if on_tick:
        on_tick(world, network, observations_flat, network.assignment, step)
    if controls:
        controls()
    return population_episode_metrics(world)


@torch.inference_mode()
def play_mixed_population_episode(genomes, neat_config, config, device, seed, assignment, seconds,
                                  shared_random=True, on_tick=None, controls=None):
    """Evaluate candidate genomes in parallel against fixed heuristic opponents.

    ``assignment`` is a flat map-major/worm-minor vector with one entry per
    world slot. ``-1`` selects the fixed :func:`heuristic` policy; every other
    value selects an index in ``genomes``. Candidate networks are compacted in
    ascending world-slot order. Metrics are returned for every world slot in
    map-major order. The preview callback receives full flattened observations
    and the full assignment; ``network.world_slots`` and
    ``network.world_to_network_slot`` map candidate world slots to the compact
    network batch. ``network.assignment`` gives the genome index for each
    compact network slot.

    The game stops once all candidate-controlled slots are dead, matching the
    focal-survival stopping rule of :func:`play_episode`. Heuristic slots may
    therefore have partial-episode metrics after the last candidate dies.
    """
    if not genomes:
        raise ValueError('At least one NEAT genome is required')
    config.validate()
    try:
        assignment_values = list(assignment)
        assignment_array = np.asarray(assignment_values)
    except (TypeError, ValueError) as exc:
        raise ValueError('assignment must be a one-dimensional integer vector') from exc
    if assignment_array.ndim != 1:
        raise ValueError('assignment must be a one-dimensional integer vector')
    slots = config.maps * config.worms
    if len(assignment_values) != slots:
        raise ValueError(f'assignment must contain one entry per map/worm cell ({slots})')
    if any(not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
           for index in assignment_values):
        raise ValueError('assignment entries must be integer genome indices or -1')
    assignment_values = [int(index) for index in assignment_values]
    if any(index < -1 or index >= len(genomes) for index in assignment_values):
        raise ValueError('assignment entries must be -1 or refer to an existing genome')
    candidate_slots = [slot for slot, index in enumerate(assignment_values) if index >= 0]
    if not candidate_slots:
        raise ValueError('At least one slot must be assigned to a candidate genome')
    candidate_assignment = [assignment_values[slot] for slot in candidate_slots]
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError('seconds must be finite and positive') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    steps = round(seconds / config.dt)
    if steps < 1:
        raise ValueError('seconds must include at least one simulation step')

    world = WorldBatch(config, device, seed, shared_random=shared_random)
    network = BatchedNetwork(genomes, neat_config, assignment=candidate_assignment, device=device)
    network.world_slots = candidate_slots
    world_to_network_slot = [-1] * slots
    for network_slot, world_slot in enumerate(candidate_slots):
        world_to_network_slot[world_slot] = network_slot
    network.world_to_network_slot = world_to_network_slot
    candidate_world_slots = torch.as_tensor(candidate_slots, dtype=torch.long, device=device)
    heuristic_slots = torch.as_tensor(
        [slot for slot, index in enumerate(assignment_values) if index == -1],
        dtype=torch.long, device=device)
    candidate_alive = torch.zeros(slots, dtype=torch.bool, device=device)
    candidate_alive[candidate_world_slots] = True

    published = -math.inf
    observations_flat = None
    for step in range(steps):
        observations_flat = world.observe().reshape(slots, -1)
        actions_flat = torch.empty((slots, 2), dtype=observations_flat.dtype,
                                   device=observations_flat.device)
        if len(heuristic_slots):
            actions_flat[heuristic_slots] = heuristic(observations_flat[heuristic_slots])
        candidate_actions = network.activate(observations_flat[candidate_world_slots])
        actions_flat[candidate_world_slots] = candidate_actions
        world.step(actions_flat.reshape(config.maps, config.worms, 2))
        now = time.perf_counter()
        if now - published > 1.:
            if controls:
                controls()
            if on_tick:
                on_tick(world, network, observations_flat, assignment_values, step)
            published = now
        if not bool((world.alive.reshape(-1) & candidate_alive).any()):
            break
    if on_tick:
        on_tick(world, network, observations_flat, assignment_values, step)
    if controls:
        controls()
    return population_episode_metrics(world)


def fixed_validation(genome, neat_config, config, device, seconds=90., controls=None, maps=32, policy='neat'):
    c = dataclasses.replace(config, maps=maps)
    result = play_episode([genome]*maps if genome else None, neat_config, c, device,
                          seed=938271, focal=np.arange(maps)*7 % c.worms, seconds=seconds,
                          controls=controls, policy=policy)
    return dict(fitness=float(np.mean(result['fitness'])), alive=float(np.mean(result['alive'])),
                food_gain=float(np.mean(result['food_gain'])), kills=float(np.mean(result['kills'])),
                metrics={key:float(np.mean(value)) for key,value in result.items()},
                standard_error=float(np.std(result['fitness'], ddof=1)/math.sqrt(maps)) if maps > 1 else 0.,
                seconds=seconds, maps=maps, seed=938271, per_map=result)
