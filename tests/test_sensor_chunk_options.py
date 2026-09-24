import dataclasses
import json
import sys

import neat
import pytest
from pydantic import ValidationError

from slitherai import server, train as train_module
from slitherai.config import SimConfig
from slitherai.network import load_config
from slitherai.protocol import protocol_settings
from slitherai.schema import contract
from slitherai.train import AtomicCheckpointer, Trainer


def _config(sensor_chunk=4):
    return SimConfig(maps=1, worms=2, foods=4, preys=0, body_points=8,
                     arena_radius=400, view_half_width=180, view_half_height=160,
                     sensor_chunk=sensor_chunk)


def _training_run(path, sensor_chunk=4, *, include_chunk=True):
    path.mkdir()
    config = _config(sensor_chunk)
    config_data = dataclasses.asdict(config)
    if not include_chunk:
        config_data.pop('sensor_chunk')
    settings = dict(config=config_data, seed=19, population=4,
                    generations=1, seconds=.1, validation_every=0,
                    protocol=protocol_settings())
    (path / 'settings.json').write_text(json.dumps(settings), encoding='utf-8')
    (path / 'schema.json').write_text(json.dumps(contract('legacy-v1')), encoding='utf-8')
    population = neat.Population(load_config(4))
    checkpointer = AtomicCheckpointer(1, filename_prefix=str(path / 'checkpoint-'))
    checkpointer.save_checkpoint(population.config, population.population,
                                 population.species, population.generation)
    return path / 'checkpoint-0'


def _file_bytes(path):
    return {file.relative_to(path).as_posix(): file.read_bytes()
            for file in path.rglob('*') if file.is_file()}


def test_cli_accepts_explicit_chunk_and_keeps_fresh_default_four(tmp_path, monkeypatch):
    seen = []

    class FakeTrainer:
        def __init__(self, config, *args):
            self.config = config

        def train(self, *args, **kwargs):
            seen.append((self.config.sensor_chunk, kwargs['sensor_chunk_explicit'],
                         kwargs['initialize_from'], args[3]))

    monkeypatch.setattr(train_module, 'Trainer', FakeTrainer)
    monkeypatch.setattr(sys, 'argv', ['train', '--run', str(tmp_path / 'chunk-16'),
                                     '--sensor-chunk', '16'])
    train_module.main()
    monkeypatch.setattr(sys, 'argv', ['train', '--run', str(tmp_path / 'warm-chunk-8'),
                                     '--initialize-from', 'runs/source/checkpoint-2',
                                     '--sensor-chunk', '8'])
    train_module.main()
    monkeypatch.setattr(sys, 'argv', ['train', '--run', str(tmp_path / 'resume-chunk-8'),
                                     '--resume', 'runs/source/checkpoint-2', '--sensor-chunk', '8'])
    train_module.main()
    monkeypatch.setattr(sys, 'argv', ['train', '--run', str(tmp_path / 'default')])
    train_module.main()
    assert seen == [
        (16, True, None, None),
        (8, True, 'runs/source/checkpoint-2', None),
        (8, True, None, 'runs/source/checkpoint-2'),
        (4, False, None, None),
    ]


def test_cli_rejects_unsupported_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train', '--run', str(tmp_path / 'bad'),
                                     '--sensor-chunk', '12'])
    with pytest.raises(SystemExit):
        train_module.main()
    with pytest.raises(ValueError, match='sensor_chunk must be one of 4, 8, or 16'):
        _config(12).validate()


def test_resume_inherits_saved_chunk_and_historical_absence_means_four(tmp_path, monkeypatch):
    monkeypatch.setattr(neat.Population, 'run', lambda self, fitness_function, n=None: None)
    for name, saved_chunk, include_chunk in (('saved-eight', 8, True), ('historical', 4, False)):
        run = tmp_path / name
        checkpoint = _training_run(run, saved_chunk, include_chunk=include_chunk)
        trainer = Trainer(_config(4), run, device='cpu', seed=19, validation_every=0)
        trainer.train(population=4, generations=1, seconds=.1, resume=checkpoint)
        saved = json.loads((run / 'settings.json').read_text(encoding='utf-8'))
        assert saved['config']['sensor_chunk'] == saved_chunk


def test_resume_rejects_explicit_chunk_mismatch_without_rewriting_run(tmp_path):
    run = tmp_path / 'resume-mismatch'
    checkpoint = _training_run(run, 8)
    before = _file_bytes(run)
    trainer = Trainer(_config(4), run, device='cpu', seed=19, validation_every=0)
    with pytest.raises(ValueError, match='saved sensor_chunk'):
        trainer.train(population=4, generations=1, seconds=.1, resume=checkpoint,
                      sensor_chunk_explicit=True)
    assert _file_bytes(run) == before


def test_warmstart_inherits_chunk_when_omitted_and_records_explicit_change(tmp_path, monkeypatch):
    monkeypatch.setattr(neat.Population, 'run', lambda self, fitness_function, n=None: None)

    source = tmp_path / 'source-eight'
    checkpoint = _training_run(source, 8)
    destination = tmp_path / 'warm-inherit'
    trainer = Trainer(_config(4), destination, device='cpu', seed=23, validation_every=0)
    trainer.train(population=4, generations=1, seconds=.1, initialize_from=checkpoint)
    settings = json.loads((destination / 'settings.json').read_text(encoding='utf-8'))
    metadata = settings['initialization']
    assert settings['config']['sensor_chunk'] == 8
    assert metadata['source_sensor_chunk'] == metadata['destination_sensor_chunk'] == 8
    assert metadata['destination_sensor_chunk_explicit'] is False

    source_four = tmp_path / 'source-four'
    checkpoint_four = _training_run(source_four, 4)
    destination_sixteen = tmp_path / 'warm-sixteen'
    trainer = Trainer(_config(16), destination_sixteen, device='cpu', seed=23, validation_every=0)
    trainer.train(population=4, generations=1, seconds=.1, initialize_from=checkpoint_four,
                  sensor_chunk_explicit=True)
    settings = json.loads((destination_sixteen / 'settings.json').read_text(encoding='utf-8'))
    metadata = settings['initialization']
    assert settings['config']['sensor_chunk'] == 16
    assert metadata['source_sensor_chunk'] == 4
    assert metadata['destination_sensor_chunk'] == 16
    assert metadata['sensor_chunk_changed'] is True
    assert metadata['destination_sensor_chunk_explicit'] is True


def _api_run(path, sensor_chunk=8, *, include_chunk=True):
    path.mkdir()
    config = dataclasses.asdict(_config(sensor_chunk))
    if not include_chunk:
        config.pop('sensor_chunk')
    (path / 'settings.json').write_text(json.dumps({
        'config': config, 'seed': 19, 'population': 4, 'seconds': 90,
        'validation_every': 5, 'protocol': protocol_settings(),
    }), encoding='utf-8')
    (path / 'checkpoint-0').write_bytes(b'test-only placeholder')


def test_server_resume_inherits_chunk_and_refuses_explicit_mismatch(tmp_path, monkeypatch):
    run = tmp_path / 'resume'
    _api_run(run, 8)
    before = _file_bytes(run)
    monkeypatch.setattr(server, 'run_dir', lambda: run)
    monkeypatch.setattr(server, 'RUNS', tmp_path / 'api-runs')
    monkeypatch.setattr(server, 'process', None)
    launches = []
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *args, **kwargs: launches.append(args))
    with pytest.raises(server.fastapi.HTTPException) as error:
        server.start(server.StartOptions(resume=True, sensor_chunk=4, device='cpu'))
    assert error.value.status_code == 400
    assert launches == []
    assert _file_bytes(run) == before

    class FinishedProcess:
        def poll(self):
            return 0

    commands = []
    monkeypatch.setattr(server.subprocess, 'Popen', lambda args, **kwargs:
                        (commands.append(args) or FinishedProcess()))
    server.start(server.StartOptions(resume=True, device='cpu'))
    command = commands[-1]
    assert command[command.index('--sensor-chunk') + 1] == '8'


def test_server_new_run_passes_explicit_chunk_and_old_settings_resume_as_four(tmp_path, monkeypatch):
    class FinishedProcess:
        def poll(self):
            return 0

    commands = []
    monkeypatch.setattr(server, 'RUNS', tmp_path / 'api-runs')
    monkeypatch.setattr(server, 'current', None)
    monkeypatch.setattr(server, 'process', None)
    monkeypatch.setattr(server.subprocess, 'Popen', lambda args, **kwargs:
                        (commands.append(args) or FinishedProcess()))
    with pytest.raises(ValidationError):
        server.StartOptions(sensor_chunk=12)
    server.start(server.StartOptions(device='cpu', maps=1, worms=2, population=4,
                                     generations=1, seconds=5, sensor_chunk=16))
    assert commands[-1][commands[-1].index('--sensor-chunk') + 1] == '16'

    run = tmp_path / 'old-run'
    _api_run(run, 4, include_chunk=False)
    monkeypatch.setattr(server, 'run_dir', lambda: run)
    monkeypatch.setattr(server, 'current', None)
    server.start(server.StartOptions(resume=True, device='cpu'))
    assert commands[-1][commands[-1].index('--sensor-chunk') + 1] == '4'
