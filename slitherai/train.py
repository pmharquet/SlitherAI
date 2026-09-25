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
from .evaluation import (PROTOCOL, protocol_settings, scenarios, aggregate_scores,
                         play_episode, play_population_episode, play_mixed_population_episode,
                         heuristic, fixed_validation)
from .protocol import mixed_reference_protocol_version, selfplay_protocol_version
from .rewards import REWARD_VERSION
from .warmstart import initialize_from_checkpoint

class Cancelled(Exception): pass


def selfplay_scenarios(seed, generation, population_size, worms, training_games):
    """Build deterministic balanced cohorts and rotate every genome's seat.

    Each game uses one candidate in every arena slot. Cohorts are remixed by
    independently shuffling each base-seat group; rotating the seat mapping
    guarantees distinct worm slots for every candidate across the requested
    games (which the protocol bounds to at most ``worms``).
    """
    if population_size < 1 or worms < 1 or population_size % worms:
        raise ValueError('Self-play population must be divisible by worms')
    if not 1 <= training_games <= worms:
        raise ValueError('Self-play training_games must be between 1 and worms')
    maps = population_size // worms
    partition_rng = random.Random(seed + generation * 1_000_003 + 0x51F1)
    ordered = list(range(population_size))
    partition_rng.shuffle(ordered)
    base_seats = [ordered[seat::worms] for seat in range(worms)]

    games = []
    for trial in range(training_games):
        rng = random.Random(seed + generation * 10_000_019 + trial * 200_003 + 0xBEEF)
        groups = [members.copy() for members in base_seats]
        for members in groups:
            rng.shuffle(members)
        assignment = np.full((maps, worms), -1, dtype=np.int64)
        for base_seat, members in enumerate(groups):
            seat = (base_seat + trial) % worms
            assignment[:, seat] = members
        if np.any(assignment < 0):
            raise RuntimeError('Self-play matchmaking did not fill every arena slot')
        games.append(dict(
            seed=seed + 10_000_019 + generation * 1_000_003 + trial * 200_003,
            trial=trial, anchor=False,
            assignment=assignment.reshape(-1).tolist()))
    return games


def mixed_reference_scenarios(seed, generation, population_size=256,
                              maps=32, worms=16, training_games=2):
    """Assign every genome once against eight fixed heuristics per arena.

    The two games reshuffle cohort membership and rotate candidate seats from
    slots 0–7 to slots 8–15. A value of -1 marks a heuristic-controlled slot.
    """
    if (population_size, maps, worms, training_games) != (256, 32, 16, 2):
        raise ValueError('Mixed-reference requires population=256, maps=32, worms=16, and two games')
    games = []
    for trial in range(training_games):
        rng = random.Random(seed + generation * 1_000_003 + 0xA71C + trial * 200_003)
        ordered = list(range(population_size))
        rng.shuffle(ordered)
        assignment = np.full((maps, worms), -1, dtype=np.int64)
        candidate_seats = list(range(trial * 8, (trial + 1) * 8))
        for arena in range(maps):
            group = ordered[arena * 8:(arena + 1) * 8]
            assignment[arena, candidate_seats] = group
        flat = assignment.reshape(-1).tolist()
        candidates = [index for index in flat if index >= 0]
        if sorted(candidates) != list(range(population_size)):
            raise RuntimeError('Mixed-reference matchmaking must assign every candidate exactly once')
        games.append(dict(
            seed=seed + 30_000_019 + generation * 1_000_003 + trial * 200_003,
            trial=trial, candidate_seats=candidate_seats, reference_seats=[s for s in range(worms)
                                                                           if s not in candidate_seats],
            assignment=flat))
    return games


def remap_mixed_slot_values(slot_values, assignment, population_size=256):
    """Return candidate-only slot values in genome-list order."""
    assignment = np.asarray(assignment)
    outcomes = np.asarray(slot_values, dtype=np.float64)
    if (assignment.ndim != 1 or assignment.dtype.kind not in ('i', 'u')
            or outcomes.shape != assignment.shape):
        raise ValueError('Mixed-reference values and assignment must be matching flat vectors')
    assignment = assignment.astype(np.int64, copy=False)
    if not np.isfinite(outcomes).all():
        raise ValueError('Mixed-reference slot values must be finite')
    candidate_slots = np.flatnonzero(assignment >= 0)
    candidate_ids = assignment[candidate_slots]
    if (np.any(assignment < -1) or np.any(assignment >= population_size)
            or len(candidate_ids) != population_size
            or sorted(candidate_ids.tolist()) != list(range(population_size))):
        raise ValueError('Mixed-reference assignment must place each genome exactly once')
    remapped = np.empty(population_size, dtype=np.float64)
    remapped[candidate_ids] = outcomes[candidate_slots]
    return remapped


def aggregate_selfplay_scores(game_scores):
    """Aggregate each candidate's co-evolution games using the v1 formula."""
    scores = np.asarray(game_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 1 or not np.isfinite(scores).all():
        raise ValueError('Expected a non-empty finite genome-by-game score matrix')
    mean = scores.mean(axis=1)
    median = np.median(scores, axis=1)
    return .5 * mean + .5 * median


def midrank_percentiles(scores):
    """Map finite scores to deterministic population midranks in (0, 1)."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError('Expected a non-empty finite score vector')
    order = np.argsort(values, kind='mergesort')
    result = np.empty(values.shape, dtype=np.float64)
    start, count = 0, values.size
    while start < count:
        end = start + 1
        while end < count and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end) / (2.0 * count)
        start = end
    return result


def configure_stagnation_protocol(population, protocol):
    """Bind checkpointed stagnation behavior to the saved selection protocol."""
    config = population.config.stagnation_config
    if protocol.get('stagnation_metric') == 'within_generation_midrank_percentile':
        config.progress_metric = 'rank_percentile'
        config.progress_window = protocol['stagnation_window']
        config.progress_delta = protocol['stagnation_delta']
    else:
        # Preserve reference and self-play-v1 behavior on fresh starts and resume.
        config.progress_metric = 'raw'


def validate_selfplay_layout(config, population_size, training_games):
    if population_size < 1 or population_size % config.worms:
        raise ValueError('Self-play population must be divisible by worms')
    if not 1 <= training_games <= config.worms:
        raise ValueError('Self-play training_games must be between 1 and worms')
    expected_maps = population_size // config.worms
    if config.maps != expected_maps:
        raise ValueError(f'Self-play requires maps=population/worms ({expected_maps}); got {config.maps}')
    return expected_maps


def validate_mixed_reference_layout(config, population_size, training_games,
                                    validation_every=5):
    if (population_size != 256 or config.maps != 32 or config.worms != 16
            or training_games != 2 or validation_every != 5):
        raise ValueError('Mixed-reference requires population=256, maps=32, worms=16, '
                         'two 45-second games per genome, and validation every five generations')
    return config.maps


def mixed_reference_scenarios_for_config(seed, generation, config,
                                         population_size, training_games):
    validate_mixed_reference_layout(config, population_size, training_games)
    return mixed_reference_scenarios(seed, generation, population_size,
                                     config.maps, config.worms, training_games)


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
    def __init__(self, config, run, device='auto', seed=1, validation_every=5,
                 opponent_mode='reference', training_games=None):
        self.config = config.validate()
        self.run = Path(run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(4)
        self.seed, self.validation_every = seed, validation_every
        if training_games is None:
            training_games = 2 if opponent_mode == 'mixed-reference' else 5
        self.opponent_mode, self.training_games = opponent_mode, training_games
        self.active_protocol = protocol_settings(opponent_mode, training_games)
        self.stagnation_metric = self.active_protocol.get('stagnation_metric', 'raw')
        self.generation = 0
        self.last_metrics = None
        self.last_preview = 0
        self.started = time.perf_counter()
        self.generation_started = self.started
        self.episode_seconds = self.active_protocol.get('seconds_per_game', 90.)
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
            network_slot = slot
            network_observations = self.live_observations
        else:
            frame_assignment = self.live_assignment.reshape(world.c.maps, world.c.worms)[arena]
            requested = int(control.get('network_worm', -1))
            if self.opponent_mode == 'mixed-reference':
                candidate_worms = np.flatnonzero(frame_assignment >= 0)
                if 0 <= requested < world.c.worms and frame_assignment[requested] >= 0:
                    chosen = requested
                else:
                    candidate_mask = torch.as_tensor(candidate_worms, device=self.device)
                    alive_candidates = candidate_mask[world.alive[arena, candidate_mask]]
                    eligible = alive_candidates if len(alive_candidates) else candidate_mask
                    chosen = int(eligible[world.mass[arena, eligible].argmax()])
            else:
                chosen = requested
                if not 0 <= chosen < world.c.worms:
                    candidates = torch.where(world.alive[arena], world.mass[arena], -torch.inf)
                    chosen = int(candidates.argmax()) if bool(world.alive[arena].any()) else int(world.mass[arena].argmax())
            slot = arena*world.c.worms+chosen
            if self.opponent_mode == 'mixed-reference':
                network_slot = self.live_network.world_to_network_slot[slot]
                network_observations = self.live_observations[self.live_network.world_slots]
            else:
                network_slot = slot
                network_observations = self.live_observations
        network = self.live_network.describe(network_slot, network_observations,
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
            assignment = frame_assignment.tolist()
            ids = [species_ids.get(self.live_network.genomes[int(index)].key) if index >= 0 else None
                   for index in assignment]
            for worm in snapshot['worms']:
                worm['controller'] = 'neat' if assignment[worm['id']] >= 0 else 'reference'
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
                     run=str(self.run),
                     protocol=self.active_protocol,
                     opponent_mode=self.opponent_mode, training_games=self.training_games,
                     progress=self.progress, **extra)
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
        if self.opponent_mode == 'selfplay':
            return self._evaluate_selfplay(genomes, neat_config)
        if self.opponent_mode == 'mixed-reference':
            return self._evaluate_mixed_reference(genomes, neat_config)
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

    def _evaluate_selfplay(self, genomes, neat_config):
        self.generation_started = time.perf_counter()
        population_size = len(genomes)
        worms = self.config.worms
        maps = validate_selfplay_layout(self.config, population_size, self.training_games)
        games = selfplay_scenarios(self.seed, self.generation, population_size,
                                   worms, self.training_games)
        game_scores = np.zeros((population_size, self.training_games), dtype=np.float64)
        values = {}
        agent_steps = 0

        for trial, scenario in enumerate(games):
            self.controls()
            assignment = np.asarray(scenario['assignment'], dtype=np.int64)
            c = dataclasses.replace(self.config, maps=maps)
            self.progress = dict(game=trial + 1, games=self.training_games,
                batch=1, batches=1, anchor=False, completed_episodes=trial * population_size,
                total_episodes=population_size * self.training_games,
                candidates=population_size, arenas=maps, worms_per_arena=worms)
            self.status('training', episode_seconds=0)
            episode_steps = [0]

            def tick(world, network, observations, slot_assignment, step):
                self.live_world, self.live_network = world, network
                self.live_observations = observations
                self.live_assignment = np.asarray(slot_assignment, dtype=np.int64)
                self.live_focal = None
                episode_steps[0] = int(world.steps)
                self.publish_preview(self.controls())
                alive = int(world.alive.sum())
                self.status('training', episode_seconds=round(world.elapsed, 1),
                    agent_steps_per_second=round((agent_steps + c.maps*c.worms*episode_steps[0]) /
                                                  max(.001, time.perf_counter()-self.generation_started), 1),
                    alive=alive, candidates_alive=alive,
                    gpu_memory_mb=round(torch.cuda.memory_allocated()/1e6) if self.device == 'cuda' else None)

            result = play_population_episode(
                [genome for _, genome in genomes], neat_config, c, self.device,
                seed=scenario['seed'], assignment=assignment, seconds=self.episode_seconds,
                shared_random=True, on_tick=tick, controls=self.controls)
            agent_steps += episode_steps[0] * maps * worms
            if len(np.unique(assignment)) != population_size:
                raise RuntimeError('Each self-play genome must appear exactly once per game')
            for key, outcomes in result.items():
                outcome_array = np.asarray(outcomes, dtype=np.float64)
                if outcome_array.shape != assignment.shape or not np.isfinite(outcome_array).all():
                    raise RuntimeError(f'Self-play metric {key!r} must have one finite value per arena slot')
                if key not in values:
                    values[key] = np.zeros_like(game_scores)
                values[key][assignment, trial] = outcome_array
            if 'fitness' not in result:
                raise RuntimeError('Self-play evaluation did not return fitness')
            game_scores[assignment, trial] = np.asarray(result['fitness'], dtype=np.float64)

        scores = aggregate_selfplay_scores(game_scores)
        stagnation_scores = (midrank_percentiles(scores)
                             if self.stagnation_metric == 'within_generation_midrank_percentile' else None)
        for index, (_, genome) in enumerate(genomes):
            genome.fitness = float(scores[index])
            # Retain the old alias for tooling. Self-play-v2 stagnation reads
            # its separate rank signal; reproduction keeps using raw fitness.
            genome.anchor_fitness = float(scores[index])
            if stagnation_scores is not None:
                genome.stagnation_fitness = float(stagnation_scores[index])
            genome.behavior = {key:float(value[index].mean()) for key,value in values.items()}
        self.evaluation_metrics = {key:float(value.mean()) for key,value in values.items()}
        self.evaluation_metrics.update(
            opponent_mode='selfplay', games_per_genome=self.training_games,
            aggregation='half_mean_half_median_all_games', anchor_fitness=None,
            anchor_fitness_compatibility_alias=float(scores.mean()),
            episode_score_sd=float(game_scores.std(axis=1).mean()),
            episodes=int(game_scores.size), arenas_per_game=maps,
            selection_score=float(scores.mean()),
            stagnation_metric=self.active_protocol.get('stagnation_metric', 'raw'),
            stagnation_score_mean=(float(stagnation_scores.mean())
                                   if stagnation_scores is not None else None),
            champion=genomes[int(scores.argmax())][1].behavior)
        self.progress['completed_episodes'] = self.progress['total_episodes']
        write_json(self.run / 'episodes' / f'generation-{self.generation:04d}.json',
            dict(generation=self.generation, opponent_mode='selfplay',
                 aggregation='half_mean_half_median_all_games',
                 genome_ids=[key for key,_ in genomes],
                 scenarios=[{key:value for key,value in game.items() if key != 'assignment'}
                            for game in games],
                 assignments=[game['assignment'] for game in games],
                 score=scores.tolist(), game_scores=game_scores.tolist(),
                 stagnation_score=(stagnation_scores.tolist() if stagnation_scores is not None else None),
                 metrics={key:value.tolist() for key,value in values.items()}))

    def _evaluate_mixed_reference(self, genomes, neat_config):
        self.generation_started = time.perf_counter()
        population_size = len(genomes)
        maps = validate_mixed_reference_layout(
            self.config, population_size, self.training_games, self.validation_every)
        games = mixed_reference_scenarios_for_config(
            self.seed, self.generation, self.config, population_size, self.training_games)
        game_scores = np.zeros((population_size, self.training_games), dtype=np.float64)
        values = {}
        agent_steps = 0
        slots_per_game = maps * self.config.worms

        for trial, scenario in enumerate(games):
            self.controls()
            assignment = np.asarray(scenario['assignment'], dtype=np.int64)
            candidate_slots = np.flatnonzero(assignment >= 0)
            candidate_ids = assignment[candidate_slots]
            if (len(candidate_slots) != population_size
                    or sorted(candidate_ids.tolist()) != list(range(population_size))):
                raise RuntimeError('Each mixed-reference genome must appear exactly once per game')
            if any(np.count_nonzero(assignment.reshape(maps, self.config.worms)[arena] >= 0) != 8
                   for arena in range(maps)):
                raise RuntimeError('Mixed-reference requires eight candidate slots per arena')

            self.progress = dict(game=trial + 1, games=self.training_games,
                batch=1, batches=1, anchor=False,
                completed_episodes=trial * population_size,
                total_episodes=population_size * self.training_games,
                candidates=population_size, arenas=maps, worms_per_arena=self.config.worms,
                candidates_per_arena=8, references_per_arena=8)
            self.status('training', episode_seconds=0)
            episode_steps = [0]

            def tick(world, network, observations, slot_assignment, step):
                self.live_world, self.live_network = world, network
                self.live_observations = observations
                self.live_assignment = np.asarray(slot_assignment, dtype=np.int64)
                self.live_focal = None
                episode_steps[0] = int(world.steps)
                self.publish_preview(self.controls())
                flat_alive = world.alive.reshape(-1)
                candidate_alive = int(flat_alive[candidate_slots].sum())
                self.status('training', episode_seconds=round(world.elapsed, 1),
                    agent_steps_per_second=round(
                        (agent_steps + population_size * episode_steps[0]) /
                        max(.001, time.perf_counter()-self.generation_started), 1),
                    alive=int(world.alive.sum()), candidates_alive=candidate_alive,
                    gpu_memory_mb=round(torch.cuda.memory_allocated()/1e6) if self.device == 'cuda' else None)

            result = play_mixed_population_episode(
                [genome for _, genome in genomes], neat_config, self.config, self.device,
                seed=scenario['seed'], assignment=assignment, seconds=self.episode_seconds,
                shared_random=True, on_tick=tick, controls=self.controls)
            agent_steps += episode_steps[0] * population_size
            if 'fitness' not in result:
                raise RuntimeError('Mixed-reference evaluation did not return fitness')
            for key, outcomes in result.items():
                outcome_array = np.asarray(outcomes, dtype=np.float64)
                if outcome_array.shape != (slots_per_game,) or not np.isfinite(outcome_array).all():
                    raise RuntimeError(f'Mixed-reference metric {key!r} must have one finite value per world slot')
                if key not in values:
                    values[key] = np.zeros_like(game_scores)
                genome_values = remap_mixed_slot_values(outcome_array, assignment, population_size)
                values[key][:, trial] = genome_values
                if key == 'fitness':
                    game_scores[:, trial] = genome_values

        if not np.isfinite(game_scores).all():
            raise RuntimeError('Mixed-reference produced non-finite candidate fitness')
        scores = game_scores.mean(axis=1)
        stagnation_scores = midrank_percentiles(scores)
        for index, (_, genome) in enumerate(genomes):
            genome.fitness = float(scores[index])
            genome.anchor_fitness = float(scores[index])
            genome.stagnation_fitness = float(stagnation_scores[index])
            genome.behavior = {key:float(value[index].mean()) for key,value in values.items()}
        self.evaluation_metrics = {key:float(value.mean()) for key,value in values.items()}
        self.evaluation_metrics.update(
            opponent_mode='mixed-reference', games_per_genome=self.training_games,
            aggregation='arithmetic_mean_two_games', anchor_fitness=None,
            anchor_fitness_compatibility_alias=float(scores.mean()),
            episode_score_sd=float(game_scores.std(axis=1).mean()),
            episodes=int(game_scores.size), arenas_per_game=maps,
            candidates_per_arena=8, references_per_arena=8,
            selection_score=float(scores.mean()),
            stagnation_metric=self.active_protocol['stagnation_metric'],
            stagnation_score_mean=float(stagnation_scores.mean()),
            champion=genomes[int(scores.argmax())][1].behavior)
        self.progress['completed_episodes'] = self.progress['total_episodes']
        write_json(self.run / 'episodes' / f'generation-{self.generation:04d}.json',
            dict(generation=self.generation, opponent_mode='mixed-reference',
                 aggregation='arithmetic_mean_two_games',
                 genome_ids=[key for key,_ in genomes],
                 scenarios=[{key:value for key,value in game.items() if key != 'assignment'}
                            for game in games],
                 assignments=[game['assignment'] for game in games],
                 score=scores.tolist(), game_scores=game_scores.tolist(),
                 stagnation_score=stagnation_scores.tolist(),
                 metrics={key:value.tolist() for key,value in values.items()}))

    def train(self, population=256, generations=50, seconds=None, resume=None,
              initialize_from=None, sensor_version_explicit=False,
              sensor_chunk_explicit=False):
        protocol = protocol_settings(self.opponent_mode, self.training_games)
        if seconds is None:
            seconds = protocol.get('seconds_per_game', 90.)
        if population < 4 or generations < 1 or seconds <= 0:
            raise ValueError('Invalid training limits')
        if resume and initialize_from:
            raise ValueError('--resume and --initialize-from are mutually exclusive')
        if self.opponent_mode == 'selfplay':
            validate_selfplay_layout(self.config, population, self.training_games)
        elif self.opponent_mode == 'mixed-reference':
            validate_mixed_reference_layout(
                self.config, population, self.training_games, self.validation_every)
            if seconds != protocol['seconds_per_game']:
                raise ValueError('Mixed-reference training episodes are fixed at 45 seconds')
        self.population_size, self.generations, self.episode_seconds = population, generations, seconds
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.config.reward_version != REWARD_VERSION:
            raise ValueError('This trainer requires growth-v2; preserve legacy runs and start a new session')
        if resume:
            settings = read_json(self.run / 'settings.json', {})
            saved_protocol = settings.get('protocol')
            if self.opponent_mode == 'selfplay':
                if (selfplay_protocol_version(saved_protocol) is None
                        or saved_protocol.get('games_per_genome') != self.training_games):
                    raise ValueError('Evaluation protocol changed: start a new session instead of mixing fitness histories')
                protocol = saved_protocol
            elif self.opponent_mode == 'mixed-reference':
                if (mixed_reference_protocol_version(saved_protocol) != 1
                        or saved_protocol.get('games_per_genome') != self.training_games
                        or settings.get('seconds') != 45
                        or settings.get('validation_every') != 5):
                    raise ValueError('Evaluation protocol changed: start a new session instead of mixing fitness histories')
                protocol = saved_protocol
            elif saved_protocol != protocol:
                raise ValueError('Evaluation protocol changed: start a new session instead of mixing fitness histories')
            pop = neat.Checkpointer.restore_checkpoint(str(resume))
            self.population_size = pop.config.pop_size
            if self.opponent_mode == 'selfplay':
                validate_selfplay_layout(self.config, self.population_size, self.training_games)
            elif self.opponent_mode == 'mixed-reference':
                validate_mixed_reference_layout(
                    self.config, self.population_size, self.training_games, self.validation_every)
            settings = read_json(self.run / 'settings.json')
            saved_config = SimConfig.from_dict(settings['config']).validate()
            if sensor_chunk_explicit and self.config.sensor_chunk != saved_config.sensor_chunk:
                raise ValueError('Resume requires the saved sensor_chunk; use a new run to change it')
            if not sensor_chunk_explicit and self.config.sensor_chunk != saved_config.sensor_chunk:
                self.config = dataclasses.replace(
                    self.config, sensor_chunk=saved_config.sensor_chunk).validate()
            if (dataclasses.asdict(saved_config) != dataclasses.asdict(self.config)
                    or settings['seed'] != self.seed):
                raise ValueError('Resume requires the same simulation config and seed')
            saved_config = settings.get('config', {})
            saved_schema_version = _saved_sensor_version(self.run, saved_config)
            if saved_schema_version != self.config.sensor_version:
                raise ValueError('Resume requires the same observation schema; preserve the saved sensor version')
        elif initialize_from:
            if any(self.run.iterdir()):
                raise ValueError('Warm-start destination must be a new empty run directory')
            pop, initialization = initialize_from_checkpoint(
                initialize_from, self.config, self.run,
                sensor_version_explicit=sensor_version_explicit,
                sensor_chunk_explicit=sensor_chunk_explicit)
            if not sensor_version_explicit and self.config.sensor_version != initialization['destination_sensor_version']:
                self.config = dataclasses.replace(
                    self.config, sensor_version=initialization['destination_sensor_version']).validate()
            if not sensor_chunk_explicit and self.config.sensor_chunk != initialization['destination_sensor_chunk']:
                self.config = dataclasses.replace(
                    self.config, sensor_chunk=initialization['destination_sensor_chunk']).validate()
            if population != len(pop.population):
                raise ValueError('Warm-start preserves the source population size; set --population to '
                                 f'{len(pop.population)}')
            self.population_size = len(pop.population)
            if self.opponent_mode == 'selfplay':
                validate_selfplay_layout(self.config, self.population_size, self.training_games)
            elif self.opponent_mode == 'mixed-reference':
                validate_mixed_reference_layout(
                    self.config, self.population_size, self.training_games, self.validation_every)
        else:
            if (self.run / 'history.jsonl').exists(): raise ValueError('Run already exists; resume it or select a new directory')
            pop = neat.Population(load_config(population))
        configure_stagnation_protocol(pop, protocol)
        self.active_protocol = protocol
        self.stagnation_metric = protocol.get('stagnation_metric', 'raw')
        self.base_generation = pop.generation
        write_json(self.run / 'schema.json', contract(self.config.sensor_version))
        settings = dict(config=dataclasses.asdict(self.config), seed=self.seed, population=self.population_size,
                        generations=generations, seconds=seconds, validation_every=self.validation_every,
                        device=self.device, opponent_mode=self.opponent_mode,
                        training_games=self.training_games,
                        maps_per_game=(self.population_size // self.config.worms
                                       if self.opponent_mode == 'selfplay' else self.config.maps),
                        protocol=protocol)
        if initialize_from:
            initialization['destination_sensor_version'] = self.config.sensor_version
            write_json(self.run / 'initialization.json', initialization)
            settings['initialization'] = initialization
        write_json(self.run / 'settings.json', settings)
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
    p.add_argument('--maps', type=int)
    p.add_argument('--worms', type=int, default=16)
    p.add_argument('--population', type=int, default=256)
    p.add_argument('--generations', type=int, default=50)
    p.add_argument('--seconds', type=float,
                   help='training episode duration; mixed-reference is fixed at 45 seconds')
    p.add_argument('--foods', type=int, default=1024)
    p.add_argument('--body-points', type=int, default=96)
    p.add_argument('--arena-radius', type=float, default=2400)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--sensor-version', choices=['legacy-v1', 'export-v1'])
    p.add_argument('--sensor-chunk', type=int, choices=[4, 8, 16],
                   help='raycast batch size; default 4, omitted on resume inherits saved settings')
    p.add_argument('--opponent-mode', choices=['reference', 'selfplay', 'mixed-reference'], default='reference',
                   help='selection opponents: fixed focal, full self-play, or 8 NEAT vs 8 heuristic')
    p.add_argument('--training-games', type=int,
                   help='games per genome; reference uses 5 and mixed-reference uses 2')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--validation-every', type=int, default=5)
    start_from = p.add_mutually_exclusive_group()
    start_from.add_argument('--resume')
    start_from.add_argument('--initialize-from', help='trusted local checkpoint-N; starts a new run and resets fitness/species history')
    args = p.parse_args()
    maps = args.maps if args.maps is not None else (32 if args.opponent_mode == 'mixed-reference' else 64)
    seconds = args.seconds if args.seconds is not None else (45 if args.opponent_mode == 'mixed-reference' else 90)
    training_games = (args.training_games if args.training_games is not None else
                      2 if args.opponent_mode == 'mixed-reference' else 5)
    config = SimConfig(maps=maps, worms=args.worms, foods=args.foods,
                       body_points=args.body_points, arena_radius=args.arena_radius,
                       sensor_chunk=args.sensor_chunk or 4,
                       sensor_version=args.sensor_version or 'legacy-v1')
    if args.opponent_mode == 'reference' and training_games == 5:
        trainer = Trainer(config, args.run, args.device, args.seed, args.validation_every)
    else:
        trainer = Trainer(config, args.run, args.device, args.seed, args.validation_every,
                          opponent_mode=args.opponent_mode, training_games=training_games)
    trainer.train(args.population, args.generations, seconds, args.resume,
                  initialize_from=args.initialize_from,
                  sensor_version_explicit=args.sensor_version is not None,
                  sensor_chunk_explicit=args.sensor_chunk is not None)

if __name__ == '__main__': main()
