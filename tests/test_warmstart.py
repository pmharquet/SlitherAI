import copy
import dataclasses
import json
import random
import sys

import neat
import numpy as np
import pytest

from slitherai import server, train as train_module
from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.protocol import protocol_settings
from slitherai.schema import contract, sensor_version_from_schema
from slitherai.train import AtomicCheckpointer, Trainer


@pytest.fixture(autouse=True)
def _preserve_test_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _config(sensor_version='legacy-v1', **overrides):
    values = dict(maps=1, worms=2, foods=4, preys=0, body_points=8,
                  arena_radius=400, view_half_width=180, view_half_height=160,
                  sensor_version=sensor_version)
    values.update(overrides)
    return SimConfig(**values)


def _gene_signature(population):
    return {
        key: (
            tuple(sorted((node_key, gene.bias, gene.response, gene.activation, gene.aggregation)
                         for node_key, gene in genome.nodes.items())),
            tuple(sorted((gene.key, gene.weight, gene.enabled, gene.innovation)
                         for gene in genome.connections.values())),
        )
        for key, genome in sorted(population.items())
    }


def _peek_count(counter):
    return None if counter is None else next(copy.copy(counter))


def _source_run(path, *, sensor_version='legacy-v1', generation=12):
    path.mkdir()
    config = _config(sensor_version)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    # Force an inherited hidden node so the source node-index counter is live.
    first = next(iter(population.population.values()))
    first.mutate_add_node(neat_config.genome_config)
    for index, genome in enumerate(population.population.values()):
        genome.fitness = 100. + index
        genome.anchor_fitness = 90. + index
        genome.behavior = {'food_gain': float(index), 'cached': True}
    for species in population.species.species.values():
        species.created = 5
        species.last_improved = 9
        species.fitness = 123.0
        species.adjusted_fitness = 0.75
        species.fitness_history = [100.0, 123.0]
        species.recent_score = 120.0
        species.progress_delta = 10.0
        species.stagnation_eligible = True
        species.protected = True
    settings = dict(config=dataclasses.asdict(config), seed=19, population=4,
                    generations=40, seconds=90.0, validation_every=5,
                    protocol=protocol_settings())
    (path / 'settings.json').write_text(json.dumps(settings), encoding='utf-8')
    (path / 'schema.json').write_text(json.dumps(contract(sensor_version)), encoding='utf-8')
    (path / 'history.jsonl').write_text('{"generation":11,"best":777}\n', encoding='utf-8')
    checkpointer = AtomicCheckpointer(1, filename_prefix=str(path / 'checkpoint-'))
    rng_state = random.getstate()
    checkpointer.save_checkpoint(population.config, population.population, population.species, generation)
    return config, population, rng_state, path / f'checkpoint-{generation}'


def _source_bytes(path):
    return {file.relative_to(path).as_posix(): file.read_bytes()
            for file in path.rglob('*') if file.is_file()}


def _species_state(species_set):
    return [(species.created, species.last_improved, species.fitness,
             species.adjusted_fitness, list(species.fitness_history),
             getattr(species, 'recent_score', None), getattr(species, 'progress_delta', None),
             getattr(species, 'stagnation_eligible', None), getattr(species, 'protected', None),
             sorted(species.members))
            for _, species in sorted(species_set.species.items())]


def test_warmstart_preserves_population_resets_evidence_and_records_provenance(tmp_path, monkeypatch):
    source = tmp_path / 'source-run'
    source_config, source_population, source_rng, checkpoint = _source_run(source)
    source_files = _source_bytes(source)
    expected_genes = _gene_signature(source_population.population)
    source_tracker = source_population.reproduction.innovation_tracker
    expected_innovation_counter = source_tracker.global_counter
    expected_node_counter = _peek_count(source_population.config.genome_config.node_indexer)
    expected_ids = sorted(source_population.population)
    caller_rng = random.getstate()
    caller_np_rng = np.random.get_state()
    destination = tmp_path / 'new-run'
    captured = {}
    original_speciate = type(source_population.species).speciate
    speciate_calls = []

    def speciate_with_rng_use(species, config, genomes, generation):
        # Exercise the restoration guard even if species clustering happens to
        # be deterministic in the current implementation.
        random.random()
        speciate_calls.append(generation)
        return original_speciate(species, config, genomes, generation)

    monkeypatch.setattr(type(source_population.species), 'speciate', speciate_with_rng_use)

    def skip_fitness(population, fitness_function, n=None):
        captured['population'] = population
        assert random.getstate() == source_rng
        assert population.generation == 0
        assert sorted(population.population) == expected_ids
        assert _gene_signature(population.population) == expected_genes
        assert all(genome.fitness is None and genome.anchor_fitness is None
                   and not hasattr(genome, 'behavior') for genome in population.population.values())
        assert all(created == 0 and last_improved == 0 and fitness is None
                   and adjusted is None and history == [] and recent is None and progress is None
                   and eligible is None and protected is None
                   for created, last_improved, fitness, adjusted, history, recent, progress,
                       eligible, protected, _members in _species_state(population.species))
        return None

    monkeypatch.setattr(neat.Population, 'run', skip_fitness)
    try:
        target_config = dataclasses.replace(source_config, maps=2, sensor_chunk=8,
                                            sensor_version='export-v1')
        trainer = Trainer(target_config, destination, device='cpu', seed=999, validation_every=0)
        trainer.train(population=4, generations=1, seconds=.1, initialize_from=checkpoint,
                      sensor_version_explicit=True, sensor_chunk_explicit=True)
    finally:
        random.setstate(caller_rng)
        np.random.set_state(caller_np_rng)

    population = captured['population']
    assert speciate_calls == [0]
    metadata = json.loads((destination / 'initialization.json').read_text(encoding='utf-8'))
    settings = json.loads((destination / 'settings.json').read_text(encoding='utf-8'))
    assert random.getstate() == caller_rng
    assert metadata['kind'] == 'checkpoint_population_warm_start'
    assert metadata['source_generation'] == 12 and metadata['generation_reset_to'] == 0
    assert metadata['source_sensor_version'] == 'legacy-v1'
    assert metadata['destination_sensor_version'] == 'export-v1'
    assert metadata['sensor_mode_changed'] is True
    assert len(metadata['source_checkpoint_sha256']) == 64
    assert len(metadata['source_config_sha256']) == len(metadata['source_schema_sha256']) == 64
    assert settings['initialization'] == metadata
    assert settings['config']['sensor_version'] == 'export-v1'
    assert sensor_version_from_schema(json.loads((destination / 'schema.json').read_text(encoding='utf-8'))) == 'export-v1'
    assert (destination / 'checkpoint-0').is_file()
    assert not (destination / 'history.jsonl').exists()
    assert _source_bytes(source) == source_files

    tracker = population.reproduction.innovation_tracker
    assert tracker.global_counter == expected_innovation_counter
    assert _peek_count(population.config.genome_config.node_indexer) == expected_node_counter
    assert population.config.genome_config.innovation_tracker is tracker
    assert _peek_count(population.reproduction.genome_indexer) == max(expected_ids) + 1
    assert tracker.generation_innovations == {}

    # A real reproduction call verifies that imported innovation numbering is
    # above all source innovations. It does not run simulation or scoring.
    for index, genome in enumerate(population.population.values()):
        genome.fitness = float(index)
        genome.anchor_fitness = float(index)
    population.config.genome_config.node_add_prob = 1.0
    population.config.genome_config.node_delete_prob = 0.0
    population.config.genome_config.conn_add_prob = 0.0
    population.config.genome_config.conn_delete_prob = 0.0
    offspring = population.reproduction.reproduce(population.config, population.species,
                                                  population.config.pop_size, generation=0)
    new_genome_ids = set(offspring) - set(expected_ids)
    assert new_genome_ids
    assert min(new_genome_ids) > max(expected_ids)
    new_innovations = {gene.innovation for genome in offspring.values()
                       for gene in genome.connections.values()} - {
        gene.innovation for genome in population.population.values()
        for gene in genome.connections.values()}
    assert new_innovations
    assert min(new_innovations) > expected_innovation_counter
    assert all(len({gene.innovation for gene in genome.connections.values()})
               == len(genome.connections) for genome in offspring.values())


def test_warmstart_inherits_source_mode_and_persists_it_when_not_explicit(tmp_path, monkeypatch):
    source = tmp_path / 'export-source'
    source_config, _source_population, source_rng, checkpoint = _source_run(
        source, sensor_version='export-v1', generation=4)
    destination = tmp_path / 'dest'
    captured = {}

    def skip_fitness(population, *_args, **_kwargs):
        assert random.getstate() == source_rng
        captured['population'] = population
        return None

    monkeypatch.setattr(neat.Population, 'run', skip_fitness)
    caller_rng = random.getstate()
    caller_np_rng = np.random.get_state()
    try:
        Trainer(dataclasses.replace(source_config, sensor_version='legacy-v1'), destination,
                device='cpu', seed=1, validation_every=0).train(
                    population=4, generations=1, seconds=.1, initialize_from=checkpoint)
    finally:
        random.setstate(caller_rng)
        np.random.set_state(caller_np_rng)
    population = captured['population']
    provenance = json.loads((destination / 'initialization.json').read_text(encoding='utf-8'))
    settings = json.loads((destination / 'settings.json').read_text(encoding='utf-8'))
    assert random.getstate() == caller_rng
    assert provenance['destination_sensor_version'] == settings['config']['sensor_version'] == 'export-v1'
    assert provenance['destination_sensor_version_explicit'] is False
    assert population.generation == 0
    assert sensor_version_from_schema(json.loads((destination / 'schema.json').read_text(encoding='utf-8'))) == 'export-v1'
    assert _source_bytes(source)['checkpoint-4']


def test_warmstart_rejects_physics_mismatch_and_nonempty_destination_without_writes(tmp_path):
    source = tmp_path / 'source-run'
    source_config, _population, _rng, checkpoint = _source_run(source)
    source_files = _source_bytes(source)
    destination = tmp_path / 'bad-destination'
    destination.mkdir()
    trainer = Trainer(dataclasses.replace(source_config, arena_radius=410), destination,
                      device='cpu', seed=19)
    with pytest.raises(ValueError, match='matching simulation physics'):
        trainer.train(population=4, generations=1, seconds=.1, initialize_from=checkpoint)
    assert list(destination.iterdir()) == []
    assert _source_bytes(source) == source_files

    (destination / 'sentinel.txt').write_text('keep', encoding='utf-8')
    trainer = Trainer(source_config, destination, device='cpu', seed=19)
    with pytest.raises(ValueError, match='new empty run directory'):
        trainer.train(population=4, generations=1, seconds=.1, initialize_from=checkpoint)
    assert (destination / 'sentinel.txt').read_text(encoding='utf-8') == 'keep'
    assert _source_bytes(source) == source_files


@pytest.mark.parametrize('invalid_metadata', ['schema', 'protocol'])
def test_warmstart_rejects_unverified_source_metadata_without_creating_run(tmp_path, invalid_metadata):
    source = tmp_path / invalid_metadata
    config, _, _, checkpoint = _source_run(source)
    schema_path, settings_path = source / 'schema.json', source / 'settings.json'
    if invalid_metadata == 'schema':
        schema_path.write_text('[]', encoding='utf-8')
        expected_error = 'must be a JSON object'
    else:
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
        settings['protocol']['version'] = 'stale-protocol'
        settings_path.write_text(json.dumps(settings), encoding='utf-8')
        expected_error = 'evaluation protocol differs'
    source_files = _source_bytes(source)
    destination = tmp_path / f'{invalid_metadata}-destination'
    trainer = Trainer(config, destination, device='cpu', seed=19)

    with pytest.raises(ValueError, match=expected_error):
        trainer.train(population=4, generations=1, seconds=.1, initialize_from=checkpoint)

    assert list(destination.iterdir()) == []
    assert _source_bytes(source) == source_files


def test_resume_and_initialize_from_are_mutually_exclusive(tmp_path):
    source = tmp_path / 'source-run'
    config, _, _, checkpoint = _source_run(source)
    destination = tmp_path / 'dest'
    trainer = Trainer(config, destination, device='cpu', seed=19)
    with pytest.raises(ValueError, match='mutually exclusive'):
        trainer.train(population=4, generations=1, seconds=.1,
                      resume=checkpoint, initialize_from=checkpoint)
    assert list(destination.iterdir()) == []


def test_cli_rejects_both_resume_modes_before_starting(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train', '--resume', 'checkpoint-1',
                                      '--initialize-from', 'checkpoint-1'])
    with pytest.raises(SystemExit):
        train_module.main()


def _api_run(path, sensor_version='legacy-v1'):
    path.mkdir()
    config = dataclasses.asdict(_config(sensor_version))
    (path / 'settings.json').write_text(json.dumps({
        'config': config, 'seed': 19, 'population': 4, 'seconds': 90,
        'validation_every': 5, 'protocol': protocol_settings(),
    }), encoding='utf-8')
    (path / 'checkpoint-0').write_bytes(b'test-only placeholder')


def test_api_resume_rejects_explicit_sensor_mismatch_but_omission_uses_saved_mode(tmp_path, monkeypatch):
    mismatch_run = tmp_path / 'mismatch'
    _api_run(mismatch_run, 'legacy-v1')
    before = _source_bytes(mismatch_run)
    monkeypatch.setattr(server, 'run_dir', lambda: mismatch_run)
    monkeypatch.setattr(server, 'RUNS', tmp_path / 'api-runs')
    monkeypatch.setattr(server, 'process', None)
    launches = []
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *args, **kwargs: launches.append(args))

    with pytest.raises(server.fastapi.HTTPException) as error:
        server.start(server.StartOptions(resume=True, sensor_version='export-v1', device='cpu'))
    assert error.value.status_code == 400
    assert launches == []
    assert _source_bytes(mismatch_run) == before

    saved_export_run = tmp_path / 'omitted-mode'
    _api_run(saved_export_run, 'export-v1')
    monkeypatch.setattr(server, 'run_dir', lambda: saved_export_run)

    class FakeProcess:
        def poll(self):
            return 0

    command = {}
    def fake_popen(args, **kwargs):
        command['args'] = args
        return FakeProcess()

    monkeypatch.setattr(server.subprocess, 'Popen', fake_popen)
    server.start(server.StartOptions(resume=True, device='cpu'))
    args = command['args']
    assert args[args.index('--sensor-version') + 1] == 'export-v1'
    assert server.StartOptions().sensor_version is None
