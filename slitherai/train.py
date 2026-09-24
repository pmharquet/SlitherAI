import argparse
import copy
import dataclasses
import gzip
import json
import math
import pickle
import random
import time
from pathlib import Path
import neat
import numpy as np
import torch
from .config import SimConfig
from .io import write_json, read_json
from .network import BatchedNetwork, load_config
from .schema import contract, sensor_version_from_schema
from .sim import WorldBatch
from .species_metrics import SpeciesReporter
from .evaluation import PROTOCOL, protocol_settings, scenarios, aggregate_scores, play_episode, heuristic, fixed_validation
from .rewards import REWARD_VERSION

class Cancelled(Exception): pass


def _saved_sensor_version(run, saved_config):
    """Resolve a run's observation semantics without guessing from bad metadata.

    Runs predating ``sensor_version`` can only be migrated as legacy when their
    settings also predate that field and no schema file was written. A present
    schema is authoritative and must be recognized before resume rewrites any
    run metadata.
    """
    schema_path = Path(run) / 'schema.json'
    if schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError('Saved observation schema is unreadable; refusing to overwrite it on resume') from exc
        version = sensor_version_from_schema(schema)
        if version is None:
            raise ValueError('Saved observation schema is unrecognized; refusing to overwrite it on resume')
        return version

    if isinstance(saved_config, dict) and 'sensor_version' not in saved_config:
        return 'legacy-v1'
    raise ValueError('Resume requires schema.json when settings specify sensor_version')

def _config_pickle_snapshot(config):
    """Copy the config shell and node counter before pickle observes it.

    neat-python 1.1.0's DefaultGenomeConfig.__getstate__ peeks at its counter by
    calling next(node_indexer), which advances the live config. Clone both config
    objects without copy/deepcopy so that __getstate__ runs only on the snapshot.
    """
    snapshot = object.__new__(type(config))
    snapshot.__dict__ = config.__dict__.copy()

    genome_config = config.genome_config
    genome_snapshot = object.__new__(type(genome_config))
    genome_snapshot.__dict__ = genome_config.__dict__.copy()
    if genome_config.node_indexer is not None:
        genome_snapshot.node_indexer = copy.copy(genome_config.node_indexer)
    snapshot.genome_config = genome_snapshot
    return snapshot


class AtomicCheckpointer(neat.Checkpointer):
    """An interrupted write cannot replace a complete checkpoint."""
    def save_checkpoint(self, config, population, species_set, generation):
        final = Path(f'{self.filename_prefix}{generation}')
        temporary = final.with_name('.' + final.name + '.tmp')
        # Runtime reporters reference the live CUDA world; keep them out of checkpoints.
        saved_species = object.__new__(type(species_set))
        saved_species.__dict__ = species_set.__dict__.copy()
        saved_species.indexer = copy.copy(species_set.indexer)
        saved_species.reporters = neat.reporting.ReporterSet()
        saved_config = _config_pickle_snapshot(config)
        with gzip.open(temporary, 'wb', compresslevel=5) as stream:
            pickle.dump((generation, saved_config, population, saved_species, random.getstate()), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(final)
        print(f'Checkpoint: {final.name}', flush=True)

class TrainingReporter(neat.reporting.BaseReporter):
    def __init__(self, trainer): self.trainer = trainer
    def start_generation(self, generation): self.trainer.generation = generation
    def post_evaluate(self, config, population, species, best_genome):
        t = self.trainer
        genomes = list(population.values())
        row = dict(generation=t.generation, best=float(best_genome.fitness), mean=float(np.mean([g.fitness for g in genomes])),
                   species=len(species.species), nodes=float(np.mean([len(g.nodes) for g in genomes])),
                   connections=float(np.mean([sum(c.enabled for c in g.connections.values()) for g in genomes])),
                   seconds=round(time.perf_counter() - t.generation_started, 2))
        row['evaluation'] = t.evaluation_metrics
        t.save_genome(best_genome, config, 'champion')
        if t.validation_every and (t.generation == 0 or (t.generation + 1) % t.validation_every == 0):
            t.status('validating')
            validation = fixed_validation(best_genome, config, t.config, t.device, controls=t.controls)
            write_json(t.run / 'validation' / f'generation-{t.generation:04d}.json', validation)
            row['validation'] = {key:value for key,value in validation.items() if key != 'per_map'}
            previous = read_json(t.run / 'best-validation.json', {})
            if row['validation']['fitness'] > previous.get('fitness', -math.inf):
                t.save_genome(best_genome, config, 'best-validation')
                write_json(t.run / 'best-validation.json', dict(row['validation'], generation=t.generation))
            if not (t.run / 'baselines.json').exists():
                baselines = {}
                for policy in ('heuristic', 'circle'):
                    t.status('validating', message=f'Calibration du contrôleur {policy}')
                    baseline = fixed_validation(None, None, t.config, t.device, controls=t.controls, policy=policy)
                    baselines[policy] = {key:value for key,value in baseline.items() if key != 'per_map'}
                write_json(t.run / 'baselines.json', baselines)
        with (t.run / 'history.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row) + '\n')
        t.last_metrics = row
        print(json.dumps(row), flush=True)

class Trainer:
    def __init__(self, config, run, device='auto', seed=1, validation_every=5):
        self.config = config.validate()
        self.run = Path(run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(4)
        self.seed, self.validation_every = seed, validation_every
        self.generation = 0
        self.last_metrics = None
        self.last_preview = 0
        self.started = time.perf_counter()
        self.generation_started = self.started
        self.episode_seconds = 90.
        self.population_size = 256
        self.generations = 50
        self.base_generation = 0
        self.live_world = self.live_network = self.live_observations = self.live_assignment = None
        self.species_tracker = None
        self.live_focal = None
        self.evaluation_metrics = {}
        self.progress = {}

    def publish_preview(self, control):
        if self.live_world is None or self.live_observations is None:
            return
        world = self.live_world
        arena = int(control.get('arena', 0)) % world.c.maps
        species_ids = self.species_tracker.population.species.genome_to_species if self.species_tracker else {}
        if self.live_focal is not None:
            chosen = int(self.live_focal[arena])
            slot = arena
        else:
            chosen = int(control.get('network_worm', -1))
            if not 0 <= chosen < world.c.worms:
                candidates = torch.where(world.alive[arena], world.mass[arena], -torch.inf)
                chosen = int(candidates.argmax()) if bool(world.alive[arena].any()) else int(world.mass[arena].argmax())
            slot = arena*world.c.worms+chosen
        network = self.live_network.describe(slot, self.live_observations,
                                             sensor_version=self.config.sensor_version)
        network.update(species_id=species_ids.get(network['genome_id']), arena=arena, worm=chosen,
                       alive=bool(world.alive[arena, chosen]), generation=self.generation,
                       observation_time=round(max(0., world.elapsed-world.c.dt), 3), captured_at=time.time())
        snapshot = world.snapshot(arena)
        if self.live_focal is not None:
            assignment = [-1]*world.c.worms
            assignment[chosen] = arena
            ids = [None]*world.c.worms
            ids[chosen] = network['species_id']
            for worm in snapshot['worms']:
                worm['controller'] = 'neat' if worm['id'] == chosen else 'reference'
        else:
            assignment = self.live_assignment.reshape(world.c.maps, world.c.worms)[arena].tolist()
            ids = [species_ids.get(self.live_network.genomes[int(index)].key) for index in assignment]
        snapshot.update(generation=self.generation, assignment=assignment, species_ids=ids,
                        network=network, evaluation=self.progress)
        write_json(self.run / 'preview.json', snapshot)

    def status(self, phase, **extra):
        self.phase = phase
        state = dict(phase=phase, generation=self.generation, target_generation=self.base_generation + self.generations,
                     population=self.population_size, device=self.device,
                     gpu=torch.cuda.get_device_name(0) if self.device == 'cuda' else None,
                     elapsed_wall=round(time.perf_counter()-self.started, 1),
                     inputs=530, outputs=2, sensor_version=self.config.sensor_version,
                     schema=contract(self.config.sensor_version),
                     config=dataclasses.asdict(self.config), last_metrics=self.last_metrics,
                     run=str(self.run), protocol=protocol_settings(), progress=self.progress, **extra)
        write_json(self.run / 'status.json', state)

    def save_genome(self, genome, config, name):
        payload = dict(genome=genome, config=_config_pickle_snapshot(config),
                       schema=contract(self.config.sensor_version), generation=self.generation)
        path = self.run / f'{name}.pkl'
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(payload))
        tmp.replace(path)
        write_json(self.run / f'{name}-network.json', dict(schema=contract(self.config.sensor_version), generation=self.generation,
            nodes=[dict(id=g.key, bias=g.bias, response=g.response, activation=g.activation) for g in genome.nodes.values()],
            connections=[dict(source=g.key[0], target=g.key[1], weight=g.weight, enabled=g.enabled, innovation=g.innovation) for g in genome.connections.values()]))

    def controls(self):
        state = read_json(self.run / 'control.json', {})
        if state.get('stop'): raise Cancelled()
        paused, previous_phase = bool(state.get('pause')), getattr(self, 'phase', 'training')
        while state.get('pause'):
            self.status('paused')
            self.publish_preview(state)
            time.sleep(.25)
            state = read_json(self.run / 'control.json', {})
            if state.get('stop'): raise Cancelled()
        if paused: self.status(previous_phase)
        return state

    def evaluate(self, genomes, neat_config):
        self.generation_started = time.perf_counter()
        games = scenarios(self.seed, self.generation, self.config.worms)
        total_batches = math.ceil(len(genomes)/self.config.maps)
        values = {}
        agent_steps = 0
        game_scores = np.zeros((len(genomes), len(games)), dtype=np.float64)
        for trial, scenario in enumerate(games):
            for batch, start in enumerate(range(0, len(genomes), self.config.maps)):
                self.controls()
                candidates = genomes[start:start+self.config.maps]
                c = dataclasses.replace(self.config, maps=len(candidates))
                self.progress = dict(game=trial+1, games=len(games), batch=batch+1, batches=total_batches,
                    anchor=scenario['anchor'], completed_episodes=trial*len(genomes)+start,
                    total_episodes=len(games)*len(genomes), candidates=len(candidates))
                self.status('training', episode_seconds=0)
                def tick(world, network, inputs, focal, step):
                    self.live_world, self.live_network = world, network
                    self.live_observations, self.live_focal = inputs, focal
                    self.publish_preview(self.controls())
                    self.status('training', episode_seconds=round(world.elapsed, 1),
                        agent_steps_per_second=round((agent_steps+c.maps*c.worms*(step+1))/max(.001,time.perf_counter()-self.generation_started)),
                        alive=int(world.alive.sum()), candidates_alive=int(world.alive[torch.arange(c.maps, device=self.device), focal].sum()),
                        gpu_memory_mb=round(torch.cuda.memory_allocated()/1e6) if self.device == 'cuda' else None)
                result = play_episode([g for _,g in candidates], neat_config, c, self.device,
                    seed=scenario['seed'], focal=scenario['focal'], seconds=self.episode_seconds,
                    shared_random=True, on_tick=tick, controls=self.controls)
                agent_steps += self.live_world.steps*c.maps*c.worms
                for key, outcomes in result.items():
                    if key not in values: values[key] = np.zeros_like(game_scores)
                    values[key][start:start+len(candidates), trial] = outcomes
                game_scores[start:start+len(candidates), trial] = result['fitness']
        scores, anchors = aggregate_scores(game_scores)
        for i, (_, genome) in enumerate(genomes):
            genome.fitness, genome.anchor_fitness = float(scores[i]), float(anchors[i])
            genome.behavior = {key:float(value[i].mean()) for key,value in values.items()}
        self.evaluation_metrics = {key:float(value.mean()) for key,value in values.items()}
        self.evaluation_metrics.update(games_per_genome=len(games), anchor_fitness=float(anchors.mean()),
            episode_score_sd=float(game_scores.std(axis=1).mean()), episodes=int(game_scores.size),
            champion=genomes[int(scores.argmax())][1].behavior)
        self.progress['completed_episodes'] = self.progress['total_episodes']
        write_json(self.run / 'episodes' / f'generation-{self.generation:04d}.json',
                   dict(generation=self.generation, genome_ids=[key for key,_ in genomes], scenarios=games,
                        score=scores.tolist(), anchor_score=anchors.tolist(),
                        metrics={key:value.tolist() for key,value in values.items()}))

    def train(self, population=256, generations=50, seconds=90., resume=None):
        if population < 4 or generations < 1 or seconds <= 0:
            raise ValueError('Invalid training limits')
        self.population_size, self.generations, self.episode_seconds = population, generations, seconds
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.config.reward_version != REWARD_VERSION:
            raise ValueError('This trainer requires growth-v2; preserve legacy runs and start a new session')
        if resume:
            settings = read_json(self.run / 'settings.json', {})
            if settings.get('protocol') != protocol_settings():
                raise ValueError('Evaluation protocol changed: start a new session instead of mixing fitness histories')
            pop = neat.Checkpointer.restore_checkpoint(str(resume))
            self.population_size = pop.config.pop_size
            settings = read_json(self.run / 'settings.json')
            if settings and (dataclasses.asdict(SimConfig.from_dict(settings['config'])) != dataclasses.asdict(self.config)
                             or settings['seed'] != self.seed):
                raise ValueError('Resume requires the same simulation config and seed')
            saved_config = settings.get('config', {})
            saved_schema_version = _saved_sensor_version(self.run, saved_config)
            if saved_schema_version != self.config.sensor_version:
                raise ValueError('Resume requires the same observation schema; preserve the saved sensor version')
        else:
            if (self.run / 'history.jsonl').exists(): raise ValueError('Run already exists; resume it or select a new directory')
            pop = neat.Population(load_config(population))
        self.base_generation = pop.generation
        write_json(self.run / 'schema.json', contract(self.config.sensor_version))
        write_json(self.run / 'settings.json', dict(config=dataclasses.asdict(self.config), seed=self.seed, population=self.population_size,
                   generations=generations, seconds=seconds, validation_every=self.validation_every, device=self.device, protocol=protocol_settings()))
        self.species_tracker = SpeciesReporter(self.run, pop)
        pop.add_reporter(self.species_tracker)
        pop.add_reporter(TrainingReporter(self))
        checkpointer = AtomicCheckpointer(1, filename_prefix=str(self.run / 'checkpoint-'))
        pop.add_reporter(checkpointer)
        # Even an interruption during the first episode has a valid restart point.
        if not resume: checkpointer.save_checkpoint(pop.config, pop.population, pop.species, pop.generation)
        try:
            pop.run(self.evaluate, generations)
            self.status('completed')
        except (Cancelled, KeyboardInterrupt):
            self.status('stopped', message='Resume from latest complete checkpoint; partial generation discarded.')
        except Exception as e:
            self.status('error', message=str(e))
            raise

def main():
    p = argparse.ArgumentParser(description='Train recurrent NEAT Slither policies in batched CUDA arenas')
    p.add_argument('--run', default='runs/main')
    p.add_argument('--maps', type=int, default=64)
    p.add_argument('--worms', type=int, default=16)
    p.add_argument('--population', type=int, default=256)
    p.add_argument('--generations', type=int, default=50)
    p.add_argument('--seconds', type=float, default=90)
    p.add_argument('--foods', type=int, default=1024)
    p.add_argument('--body-points', type=int, default=96)
    p.add_argument('--arena-radius', type=float, default=2400)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--sensor-version', choices=['legacy-v1', 'export-v1'], default='legacy-v1')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--validation-every', type=int, default=5)
    p.add_argument('--resume')
    args = p.parse_args()
    config = SimConfig(maps=args.maps, worms=args.worms, foods=args.foods,
                       body_points=args.body_points, arena_radius=args.arena_radius,
                       sensor_version=args.sensor_version)
    trainer = Trainer(config, args.run, args.device, args.seed, args.validation_every)
    trainer.train(args.population, args.generations, args.seconds, args.resume)

if __name__ == '__main__': main()
