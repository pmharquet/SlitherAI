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
    values = dict(fitness=world.fitness(), food_gain=world.gained, boost_spent=world.spent,
                  alive=world.alive, age=world.age, kills=world.kills,
                  border_death=world.border_deaths, collision_death=world.collision_deaths,
                  boost_fraction=world.boost_steps/world.decisions.clamp_min(1),
                  turn_degrees=world.turn_sum/world.decisions.clamp_min(1)*(180/math.pi))
    values.update({f'reward_{key}':value for key,value in world.fitness_terms().items()})
    array = torch.stack([take(value) for value in values.values()], -1).cpu().numpy()
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
