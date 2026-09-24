"""Offline diagnostics: saved history and frozen-policy trials, without training."""
import argparse
import dataclasses
import gzip
import json
import math
import pickle
import statistics
import sys
from pathlib import Path
import numpy as np
import torch
from .config import SimConfig
from .io import read_json, write_json
from .network import BatchedNetwork
from .sim import WorldBatch
from .train import Trainer, TrainingReporter, AtomicCheckpointer, heuristic


def load_checkpoint(path):
    # Older local checkpoints included reporters serialized by python -m slitherai.train.
    for cls in (Trainer, TrainingReporter, AtomicCheckpointer):
        setattr(sys.modules['__main__'], cls.__name__, cls)
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def topology(config, population, species):
    genomes = list(population.values())
    rng = np.random.default_rng(8451)
    pairs = rng.integers(0, len(genomes), size=(512, 2))
    within, between = [], []
    for i, j in pairs:
        if i == j:
            continue
        a, b = genomes[i], genomes[j]
        same = species.genome_to_species[a.key] == species.genome_to_species[b.key]
        (within if same else between).append(a.distance(b, config.genome_config))
    def stats(values):
        return dict(mean=float(np.mean(values)), p10=float(np.quantile(values, .1)),
                    p90=float(np.quantile(values, .9))) if values else None
    counts = []
    for genome in genomes:
        used = {c.key[0] for c in genome.connections.values() if c.enabled and c.key[0] < 0}
        food = [k for k in used if (-k-1) < 522 and (-k-1) % 6 == 5]
        counts.append(dict(inputs=len(used), food_inputs=len(food), hidden=len(genome.nodes)-2,
                           enabled=sum(c.enabled for c in genome.connections.values()),
                           total=len(genome.connections)))
    return dict(mean={key:statistics.mean(r[key] for r in counts) for key in counts[0]},
                species=len(species.species), within_distance=stats(within), between_distance=stats(between))


@torch.inference_mode()
def trial(name, sim_config, genome=None, config=None, maps=16, seconds=90., device='cuda', seed=20260924):
    c = dataclasses.replace(sim_config, maps=maps, worms=16, foods=1024)
    world = WorldBatch(c, device, seed)
    row = torch.arange(maps, device=device)
    focal = row.remainder(c.worms)
    network = BatchedNetwork([genome], config, assignment=[0]*maps, device=device) if genome else None
    decisions = torch.zeros(maps, device=device)
    boosts, saturated, turning, observed_food = (torch.zeros_like(decisions) for _ in range(4))
    initial = world.head[row, focal].clone()
    displacement = torch.zeros_like(decisions)
    previous_direction = None
    for step in range(round(seconds/c.dt)):
        obs = world.observe().reshape(maps, c.worms, -1)
        actions = heuristic(obs.flatten(0, 1)).reshape(maps, c.worms, 2)
        inputs = obs[row, focal].clone()
        live = world.alive[row, focal]
        observed_food += (inputs[:, :522].reshape(maps, 87, 6)[:, :, 5].amax(-1) > 0) * live
        if name == 'final_sensors_masked':
            inputs[:, :522].reshape(maps, 87, 6)[:, :, 1:] = 0
        if network:
            actions[row, focal] = network.activate(inputs)
        elif name == 'circle':
            actions[row, focal, 0] = 0
            actions[row, focal, 1] = ((world.heading[row, focal] + .8)/math.tau).remainder(1)
        chosen = actions[row, focal]
        decisions += live
        boosts += (chosen[:, 0] >= .5) * live
        saturated += ((chosen[:, 1] < .01) | (chosen[:, 1] > .99)) * live
        if previous_direction is not None:
            delta = (chosen[:, 1] - previous_direction).abs()
            turning += torch.minimum(delta, 1-delta) * 360 * live
        previous_direction = chosen[:, 1].clone()
        world.step(actions)
        displacement = torch.maximum(displacement, (world.head[row, focal]-initial).norm(dim=-1))
        if step % 300 == 0:
            print(json.dumps(dict(policy=name, simulated_seconds=round(world.elapsed, 1))), flush=True)
        if not bool(world.alive[row, focal].any()):
            break
    arrays = dict(fitness=world.fitness()[row, focal], alive=world.alive[row, focal].float(),
        age=world.age[row, focal], food=world.gained[row, focal], spent=world.spent[row, focal],
        kills=world.kills[row, focal], boost_fraction=boosts/decisions.clamp_min(1),
        direction_saturation=saturated/decisions.clamp_min(1),
        direction_change_deg=turning/decisions.clamp_min(1), max_displacement=displacement,
        food_visible_fraction=observed_food/decisions.clamp_min(1))
    values = {key:value.cpu().tolist() for key,value in arrays.items()}
    return dict(name=name, maps=maps, worms=c.worms, foods=c.foods, seconds=seconds, seed=seed,
        means={key:statistics.mean(value) for key,value in values.items()}, per_map=values,
        fitness_se=statistics.stdev(values['fitness'])/math.sqrt(maps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    species_history = read_json(run/'species-history.json')
    settings = read_json(run/'settings.json')
    config = SimConfig.from_dict(settings['config'])
    summary = dict(windows=[], validation=[dict(generation=r['generation']+1, **r['validation']) for r in history if 'validation' in r], extinctions=[])
    for start in range(0, len(history), 10):
        window = history[start:start+10]
        summary['windows'].append(dict(generations=[window[0]['generation']+1,window[-1]['generation']+1],
            best_mean=statistics.mean(r['best'] for r in window), population_mean=statistics.mean(r['mean'] for r in window)))
    for row in species_history:
        for species in row.get('rows', []):
            if species.get('removed'):
                summary['extinctions'].append(dict(generation=row['generation']+1, **species))
    summary['topology'] = {}
    for gen in (0, 10, 25, 50):
        _, nc, pop, species, _ = load_checkpoint(run/f'checkpoint-{gen}')
        summary['topology'][str(gen)] = topology(nc, pop, species)
    write_json(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    _, nc, pop, _, _ = load_checkpoint(run/'checkpoint-1')
    initial = max((g for g in pop.values() if g.fitness is not None), key=lambda g:g.fitness)
    best = pickle.loads((run/'best-validation.pkl').read_bytes())
    final = pickle.loads((run/'champion.pkl').read_bytes())
    policies = [('initial', initial, nc), ('best_validation_g10', best['genome'], best['config']),
                ('final', final['genome'], final['config']), ('heuristic', None, None),
                ('circle', None, None), ('final_sensors_masked', final['genome'], final['config'])]
    trials = []
    for name, genome, nc in policies:
        result = trial(name, config, genome, nc)
        trials.append(result)
        write_json(out/'trials.json', trials)
        print(json.dumps(dict(name=name, means=result['means'], fitness_se=result['fitness_se'])), flush=True)


if __name__ == '__main__':
    main()
