import dataclasses
import json
import pickle
import random
import sys

import neat
import pytest

from slitherai.config import SimConfig
from slitherai.evaluate_holdout import _load_simulation_config, run_holdout
from slitherai.network import load_config
from slitherai.protocol import protocol_settings
from slitherai.schema import (
    INPUTS,
    VERSION,
    contract,
    schema_id,
    sensor_version_from_schema,
)
from slitherai import server, train as train_module
from slitherai.train import AtomicCheckpointer, Trainer, _saved_sensor_version


def _small_config(sensor_version='legacy-v1'):
    return SimConfig(maps=1, worms=2, foods=4, preys=0, body_points=8,
                     arena_radius=400, view_half_width=180, view_half_height=160,
                     sensor_version=sensor_version)


def _make_run(path):
    config = _small_config()
    path.mkdir()
    (path / 'episodes').mkdir()
    (path / 'settings.json').write_text(json.dumps({
        'config': dataclasses.asdict(config), 'protocol': protocol_settings(), 'seed': 7,
    }), encoding='utf-8')
    (path / 'schema.json').write_text(json.dumps(contract(config.sensor_version)), encoding='utf-8')
    random.seed(51)
    neat_config = load_config(4)
    population = neat.Population(neat_config)
    genome_ids = list(population.population)
    for index, genome in enumerate(population.population.values()):
        genome.fitness = 1000. - index
    checkpointer = AtomicCheckpointer(1, filename_prefix=str(path / 'checkpoint-'))
    checkpointer.save_checkpoint(population.config, population.population, population.species, 0)
    (path / 'episodes' / 'generation-0000.json').write_text(json.dumps({
        'generation': 0, 'genome_ids': genome_ids, 'score': [float(len(genome_ids)-i) for i in range(len(genome_ids))],
    }), encoding='utf-8')
    return config, neat_config, population


def test_sensor_version_defaults_and_migrates_old_config_to_legacy():
    original = _small_config()
    saved = dataclasses.asdict(original)
    saved.pop('sensor_version')
    migrated = SimConfig.from_dict(saved)

    assert migrated.sensor_version == 'legacy-v1'
    assert dataclasses.asdict(migrated) == dataclasses.asdict(original)
    assert SimConfig.from_dict(dataclasses.asdict(_small_config('export-v1'))).sensor_version == 'export-v1'
    with pytest.raises(ValueError, match='Unknown sensor version'):
        SimConfig(sensor_version='future-v9').validate()


def test_sensor_schemas_distinguish_semantics_without_changing_530_inputs():
    legacy = contract('legacy-v1')
    exported = contract('export-v1')

    assert INPUTS == legacy['inputs'] == exported['inputs'] == 530
    assert legacy['ray_channels'] == exported['ray_channels']
    assert legacy['version'] == VERSION
    assert exported['version'] == schema_id('export-v1')
    assert legacy['sensor_version'] == 'legacy-v1'
    assert exported['sensor_version'] == 'export-v1'
    assert sensor_version_from_schema(legacy) == 'legacy-v1'
    assert sensor_version_from_schema(exported) == 'export-v1'
    assert sensor_version_from_schema(VERSION) == 'legacy-v1'

    historical_contract = dict(legacy)
    historical_contract.pop('sensor_version')
    historical_contract.pop('self_body_model')
    assert sensor_version_from_schema(historical_contract) == 'legacy-v1'


def test_old_settings_and_schema_load_as_legacy(tmp_path):
    run = tmp_path / 'old-run'
    run.mkdir()
    old_config = dataclasses.asdict(_small_config())
    old_config.pop('sensor_version')
    (run / 'settings.json').write_text(json.dumps({
        'config': old_config, 'protocol': protocol_settings(), 'seed': 7,
    }), encoding='utf-8')
    historical_schema = dict(contract('legacy-v1'))
    historical_schema.pop('sensor_version')
    historical_schema.pop('self_body_model')
    (run / 'schema.json').write_text(json.dumps(historical_schema), encoding='utf-8')

    config, metadata = _load_simulation_config(run, None, maps=1)

    assert config.sensor_version == 'legacy-v1'
    assert metadata['training_sensor_version'] == metadata['sensor_version'] == 'legacy-v1'


def test_resume_rejects_present_but_unreadable_or_unknown_schema_without_rewriting(tmp_path):
    for name, schema_bytes in (
        ('corrupt', b'{"version":'),
        ('unknown', json.dumps({'version': 'slither-neat-999-future'}).encode('utf-8')),
    ):
        run = tmp_path / name
        config, _, _ = _make_run(run)
        schema_path = run / 'schema.json'
        schema_path.write_bytes(schema_bytes)
        trainer = Trainer(config, run, device='cpu', seed=7)

        with pytest.raises(ValueError, match='Saved observation schema'):
            trainer.train(population=4, generations=1, seconds=.1,
                          resume=run / 'checkpoint-0')

        assert schema_path.read_bytes() == schema_bytes


def test_missing_schema_is_legacy_only_for_historical_settings(tmp_path):
    old_run = tmp_path / 'old-no-schema'
    old_run.mkdir()
    assert _saved_sensor_version(old_run, {'maps': 64}) == 'legacy-v1'

    explicit_run = tmp_path / 'explicit-no-schema'
    explicit_run.mkdir()
    with pytest.raises(ValueError, match='requires schema.json'):
        _saved_sensor_version(explicit_run, {'sensor_version': 'legacy-v1'})


def test_historical_legacy_resume_continues_checkpoint_and_writes_explicit_mode(tmp_path):
    run = tmp_path / 'historical-resume'
    config, _, population = _make_run(run)
    checkpoint = run / 'checkpoint-6'
    AtomicCheckpointer(1, filename_prefix=str(run / 'checkpoint-')).save_checkpoint(
        population.config, population.population, population.species, 6)
    checkpoint_bytes = checkpoint.read_bytes()
    restored_population = neat.Checkpointer.restore_checkpoint(str(checkpoint))
    original_ids = set(population.population)
    assert restored_population.generation == 6
    assert set(restored_population.population) == original_ids
    settings_path = run / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    settings['config'].pop('sensor_version')
    settings_path.write_text(json.dumps(settings), encoding='utf-8')
    historical_schema = dict(contract('legacy-v1'))
    historical_schema.pop('sensor_version')
    historical_schema.pop('self_body_model')
    (run / 'schema.json').write_text(json.dumps(historical_schema), encoding='utf-8')

    Trainer(config, run, device='cpu', seed=7, validation_every=0).train(
        population=4, generations=1, seconds=.1, resume=checkpoint)

    history = [json.loads(line) for line in (run / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['generation'] for row in history] == [6]
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert (run / 'checkpoint-7').is_file()
    assert sensor_version_from_schema(json.loads((run / 'schema.json').read_text(encoding='utf-8'))) == 'legacy-v1'
    saved_settings = json.loads(settings_path.read_text(encoding='utf-8'))
    assert saved_settings['config']['sensor_version'] == 'legacy-v1'
    champion = pickle.loads((run / 'champion.pkl').read_bytes())
    assert sensor_version_from_schema(champion['schema']) == 'legacy-v1'
    assert saved_settings['population'] == 4


def test_export_resume_is_refused_without_changing_legacy_run_files(tmp_path):
    run = tmp_path / 'no-cross-mode-resume'
    config, _, _ = _make_run(run)
    (run / 'history.jsonl').write_text('{"generation":0}\n', encoding='utf-8')
    protected = [run / name for name in (
        'settings.json', 'schema.json', 'history.jsonl', 'checkpoint-0',
    )]
    before = {path.name: path.read_bytes() for path in protected}

    trainer = Trainer(dataclasses.replace(config, sensor_version='export-v1'), run,
                      device='cpu', seed=7)
    with pytest.raises(ValueError, match='same simulation config'):
        trainer.train(population=4, generations=1, seconds=.1, resume=run / 'checkpoint-0')

    assert {path.name: path.read_bytes() for path in protected} == before


def test_cli_and_api_forward_explicit_sensor_version(monkeypatch, tmp_path):
    captured = {}

    class FakeTrainer:
        def __init__(self, config, run, device, seed, validation_every):
            captured['cli_sensor_version'] = config.sensor_version

        def train(self, *args, **kwargs):
            captured['cli_train_called'] = True

    monkeypatch.setattr(train_module, 'Trainer', FakeTrainer)
    monkeypatch.setattr(sys, 'argv', [
        'train', '--run', str(tmp_path / 'cli-run'), '--maps', '1', '--worms', '2',
        '--foods', '4', '--body-points', '8', '--arena-radius', '400',
        '--device', 'cpu', '--sensor-version', 'export-v1',
    ])
    train_module.main()
    assert captured['cli_sensor_version'] == 'export-v1'
    assert captured['cli_train_called'] is True

    class FakeProcess:
        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured['api_command'] = command
        return FakeProcess()

    monkeypatch.setattr(server, 'RUNS', tmp_path / 'api-runs')
    monkeypatch.setattr(server, 'current', None)
    monkeypatch.setattr(server, 'process', None)
    monkeypatch.setattr(server.subprocess, 'Popen', fake_popen)
    server.start(server.StartOptions(device='cpu', maps=1, worms=2, population=4,
                                     generations=1, seconds=5, sensor_version='export-v1'))
    command = captured['api_command']
    assert command[command.index('--sensor-version') + 1] == 'export-v1'


def test_holdout_refuses_implicit_cross_sensor_comparison_but_allows_explicit_rescore(tmp_path):
    run = tmp_path / 'run'
    config, neat_config, population = _make_run(run)
    genome = next(iter(population.population.values()))
    candidate = run / 'export-candidate.pkl'
    candidate.write_bytes(pickle.dumps({
        'genome': genome, 'config': neat_config,
        'schema': contract('export-v1'), 'generation': 12,
    }))

    with pytest.raises(ValueError, match='sensor_version differs'):
        run_holdout(run, [], [('export-candidate', candidate),], seed=32,
                    maps=1, seconds=.1, device='cpu')

    comparison = tmp_path / 'comparison.json'
    export_config = dataclasses.asdict(config)
    export_config['sensor_version'] = 'export-v1'
    comparison.write_text(json.dumps({
        'config': export_config, 'protocol': protocol_settings(),
    }), encoding='utf-8')
    result = run_holdout(run, [], [('export-candidate', candidate)], seed=32,
                         maps=1, seconds=.1, device='cpu', comparison_path=comparison)

    assert result['comparison']['sensor_version'] == 'export-v1'
    assert result['comparison']['training_sensor_version'] == 'legacy-v1'
    assert result['comparison']['schema']['inputs'] == 530
    assert result['comparison']['all_policies_freshly_rescored'] is True
    assert result['models'][0]['source']['sensor_version'] == 'legacy-v1'
    assert next(row for row in result['models'] if row['name'] == 'export-candidate')['source']['sensor_version'] == 'export-v1'
