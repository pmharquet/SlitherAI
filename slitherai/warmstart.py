"""Warm-start a new run from a trusted local NEAT checkpoint.

This imports the live population, not its evaluated fitness or species history.
It deliberately does not copy any files from the source run.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
from pathlib import Path

import neat

from .config import SimConfig
from .protocol import protocol_settings
from .schema import INPUTS, OUTPUTS, contract, sensor_version_from_schema


_OPERATIONAL_CONFIG_FIELDS = {'maps', 'sensor_chunk', 'sensor_version'}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return _sha256_bytes(payload)


def _read_json_object(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode('utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Warm-start source {label} is unreadable: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Warm-start source {label} must be a JSON object: {path}')
    return value, raw


def _validate_neat_population(population) -> None:
    config = population.config
    genome_config = config.genome_config
    expected_inputs = list(range(-1, -INPUTS - 1, -1))
    expected_outputs = list(range(len(OUTPUTS)))
    if genome_config.num_inputs != INPUTS or list(genome_config.input_keys) != expected_inputs:
        raise ValueError('Warm-start checkpoint input keys do not match the 530-input schema')
    if genome_config.num_outputs != len(OUTPUTS) or list(genome_config.output_keys) != expected_outputs:
        raise ValueError('Warm-start checkpoint output keys do not match this trainer')
    if config.pop_size != len(population.population) or not population.population:
        raise ValueError('Warm-start checkpoint population size is inconsistent')
    input_set = set(expected_inputs)
    for genome in population.population.values():
        if not set(expected_outputs).issubset(genome.nodes):
            raise ValueError(f'Warm-start genome {genome.key} is missing an output node')
        for node in genome.nodes.values():
            if node.activation != 'sigmoid' or node.aggregation != 'sum':
                raise ValueError(f'Warm-start genome {genome.key} uses an unsupported node function')
        for connection in genome.connections.values():
            source, target = connection.key
            if source not in input_set and source not in genome.nodes:
                raise ValueError(f'Warm-start genome {genome.key} has an unknown connection source')
            if target not in genome.nodes:
                raise ValueError(f'Warm-start genome {genome.key} has an unknown connection target')


def _validate_simulation_compatibility(source: SimConfig, target: SimConfig) -> None:
    source_values, target_values = dataclasses.asdict(source), dataclasses.asdict(target)
    differences = [name for name in source_values
                   if name not in _OPERATIONAL_CONFIG_FIELDS and source_values[name] != target_values[name]]
    if differences:
        raise ValueError('Warm-start requires matching simulation physics and reward settings; differ: '
                         + ', '.join(differences))


def initialize_from_checkpoint(
    checkpoint: str | Path,
    target_config: SimConfig,
    destination: str | Path,
    *,
    sensor_version_explicit: bool = False,
    sensor_chunk_explicit: bool = False,
) -> tuple[neat.Population, dict]:
    """Load and reset the live population for generation zero of a new run.

    If the caller did not explicitly select a sensor version or raycast chunk,
    the source setting is inherited. An explicit target mode permits a
    deliberate cross-sensor warm-start after compatibility checks.
    """
    checkpoint_path = Path(checkpoint).resolve()
    destination_path = Path(destination).resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f'Warm-start checkpoint does not exist: {checkpoint_path}')
    if not destination_path.is_dir():
        raise ValueError('Warm-start destination must be a new empty run directory')
    match = re.fullmatch(r'checkpoint-(\d+)', checkpoint_path.name)
    if not match:
        raise ValueError('Warm-start source must be a local checkpoint-N file')
    source_generation = int(match.group(1))
    source_run = checkpoint_path.parent
    try:
        destination_path.relative_to(source_run)
    except ValueError:
        pass
    else:
        raise ValueError('Warm-start destination must be outside the source run')
    if any(destination_path.iterdir()):
        raise ValueError('Warm-start destination must be a new empty run directory')

    settings, settings_bytes = _read_json_object(source_run / 'settings.json', 'settings.json')
    schema, schema_bytes = _read_json_object(source_run / 'schema.json', 'schema.json')
    source_config_values = settings.get('config')
    if not isinstance(source_config_values, dict):
        raise ValueError('Warm-start source settings have no simulation config')
    try:
        source_config = SimConfig.from_dict(source_config_values).validate()
    except (TypeError, ValueError) as exc:
        raise ValueError('Warm-start source simulation config is invalid') from exc
    source_sensor_version = sensor_version_from_schema(schema)
    if source_sensor_version is None:
        raise ValueError('Warm-start source schema is unsupported')
    if source_sensor_version != source_config.sensor_version:
        raise ValueError('Warm-start source settings and schema disagree about sensor_version')
    if settings.get('protocol') != protocol_settings():
        raise ValueError('Warm-start source evaluation protocol differs from the active protocol')
    target_config.validate()
    _validate_simulation_compatibility(source_config, target_config)
    if source_config.reward_version != target_config.reward_version:
        raise ValueError('Warm-start requires the same reward version')
    destination_sensor_version = (target_config.sensor_version if sensor_version_explicit
                                  else source_sensor_version)
    destination_sensor_chunk = (target_config.sensor_chunk if sensor_chunk_explicit
                                else source_config.sensor_chunk)
    effective_target_config = dataclasses.replace(
        target_config, sensor_version=destination_sensor_version,
        sensor_chunk=destination_sensor_chunk).validate()

    try:
        population = neat.Checkpointer.restore_checkpoint(str(checkpoint_path))
    except Exception as exc:
        raise ValueError(f'Cannot restore trusted local warm-start checkpoint: {type(exc).__name__}') from exc
    source_rng_state = random.getstate()
    if population.generation != source_generation:
        raise ValueError('Warm-start checkpoint generation does not match its filename')
    saved_population_size = settings.get('population')
    if saved_population_size is not None and saved_population_size != len(population.population):
        raise ValueError('Warm-start checkpoint population size differs from source settings')
    _validate_neat_population(population)
    if not hasattr(population.reproduction, 'innovation_tracker') or population.reproduction.innovation_tracker is None:
        raise ValueError('Warm-start checkpoint has no innovation tracker; cannot guarantee unique innovations')
    tracker = population.reproduction.innovation_tracker
    maximum_gene_innovation = max(
        (connection.innovation for genome in population.population.values()
         for connection in genome.connections.values()), default=0)
    if tracker.global_counter < maximum_gene_innovation:
        raise ValueError('Warm-start innovation counter is behind existing connection genes')
    if population.config.genome_config.innovation_tracker is not tracker:
        raise ValueError('Warm-start genome config lost its innovation tracker')

    for genome in population.population.values():
        genome.fitness = None
        genome.anchor_fitness = None
        genome.__dict__.pop('behavior', None)

    # Rebuild species from the imported genomes. This zeros species ages,
    # fitness history and stagnation annotations while preserving the NEAT
    # config, genome allocator and innovation/node counters.
    reporters = population.reporters
    population.species = population.config.species_set_type(
        population.config.species_set_config, reporters)
    population.species.speciate(population.config, population.population, 0)
    # A fresh generation zero must not reuse the source generation's structural
    # mutation deduplication entries. Keep the monotonic innovation counter.
    tracker.reset_generation()
    population.reproduction.stagnation = population.config.stagnation_type(
        population.config.stagnation_config, reporters)
    population.reproduction.ancestors = {}
    population.generation = 0
    population.best_genome = None
    # Some species-set implementations may sample while clustering. The
    # imported checkpoint RNG is the continuation state for new-run evolution.
    random.setstate(source_rng_state)

    try:
        checkpoint_bytes = checkpoint_path.read_bytes()
    except OSError as exc:
        raise ValueError(f'Cannot hash warm-start checkpoint: {checkpoint_path}') from exc
    metadata = {
        'kind': 'checkpoint_population_warm_start',
        'source_checkpoint': str(checkpoint_path),
        'source_checkpoint_sha256': _sha256_bytes(checkpoint_bytes),
        'source_generation': source_generation,
        'source_run': str(source_run),
        'source_population': len(population.population),
        'source_genome_ids': sorted(population.population),
        'source_sensor_version': source_sensor_version,
        'destination_sensor_version': destination_sensor_version,
        'destination_sensor_version_explicit': bool(sensor_version_explicit),
        'sensor_mode_changed': source_sensor_version != destination_sensor_version,
        'source_sensor_chunk': source_config.sensor_chunk,
        'destination_sensor_chunk': destination_sensor_chunk,
        'destination_sensor_chunk_explicit': bool(sensor_chunk_explicit),
        'sensor_chunk_changed': source_config.sensor_chunk != destination_sensor_chunk,
        'source_settings_sha256': _sha256_bytes(settings_bytes),
        'source_config_sha256': _canonical_sha256(source_config_values),
        'source_schema_sha256': _sha256_bytes(schema_bytes),
        'source_protocol_sha256': _canonical_sha256(settings['protocol']),
        'destination_config_sha256': _canonical_sha256(dataclasses.asdict(effective_target_config)),
        'destination_schema_sha256': _canonical_sha256(contract(destination_sensor_version)),
        'generation_reset_to': 0,
        'fitness_reset': 'fitness, anchor_fitness cleared; every imported genome must be freshly evaluated',
        'behavior_reset': 'cached behavior removed',
        'species_reset': 'species reformed at generation 0; ages, fitness history and stagnation annotations reset',
        'innovation_generation_reset': 'generation deduplication map cleared; monotonic global counter preserved',
        'preserved': [
            'live genome IDs, node genes, connection genes, weights, enabled flags and innovations',
            'NEAT config, node-indexer state, innovation tracker and checkpoint Python RNG state',
            'next genome ID is restored after the largest imported live genome ID',
        ],
        'not_copied': [
            'source history, episode scores, validation, baselines, settings and run artifacts',
            'source species memberships/ages/fitness history, cached genome fitness and behavior',
        ],
        'config_fields_allowed_to_differ': sorted(_OPERATIONAL_CONFIG_FIELDS),
    }
    return population, metadata
