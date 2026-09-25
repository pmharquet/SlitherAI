import pytest
from fastapi import HTTPException

from slitherai.protocol import (mixed_reference_protocol_version, protocol_settings,
                                recognized_protocol)
from slitherai import server
from slitherai.server import saved_protocol_options


def test_reference_protocol_remains_compatible_with_saved_runs():
    expected = dict(version='common-reference-v2', anchor_games=4, rotating_games=1,
                    anchor_weight=.8, aggregate='half_mean_half_median',
                    opponents='fixed_food_and_avoidance', validation_maps=32,
                    validation_seconds=90)
    assert protocol_settings() == expected
    assert saved_protocol_options({'protocol': expected}) == ('reference', 5)


def test_selfplay_protocol_is_separate_and_validated():
    protocol = protocol_settings('selfplay', 3)
    assert protocol['version'] == 'population-selfplay-v2'
    assert protocol['games_per_genome'] == 3
    assert protocol['opponents'] == 'co_evolving_population'
    assert protocol['stagnation_metric'] == 'within_generation_midrank_percentile'
    assert protocol['stagnation_window'] == 5
    assert protocol['stagnation_delta'] == .02
    assert protocol['validation_opponents'] == 'fixed_food_and_avoidance'
    assert saved_protocol_options({'protocol': protocol}) == ('selfplay', 3)
    assert saved_protocol_options({'protocol': dict(protocol, aggregate='mean_only')}) is None
    legacy = protocol_settings('selfplay', 3, selfplay_version=1)
    assert legacy['version'] == 'population-selfplay-v1'
    assert 'stagnation_metric' not in legacy
    assert saved_protocol_options({'protocol': legacy}) == ('selfplay', 3)
    with pytest.raises(ValueError):
        protocol_settings('reference', 3)
    with pytest.raises(ValueError):
        protocol_settings('selfplay', 0)


def test_mixed_reference_protocol_is_versioned_and_strict():
    protocol = protocol_settings('mixed-reference', 2)
    assert protocol['version'] == 'mixed-reference-v3'
    assert protocol['population'] == 256
    assert protocol['maps_per_game'] == 64 and protocol['worms_per_map'] == 16
    assert protocol['candidate_slots_per_map'] == 4 and protocol['reference_slots_per_map'] == 12
    assert protocol['games_per_genome'] == 2 and protocol['seconds_per_game'] == 90
    assert protocol['sensor_chunk'] == 8
    assert protocol['aggregate'] == 'arithmetic_mean_two_games'
    assert protocol['stagnation_metric'] == 'within_generation_midrank_percentile'
    assert protocol['stagnation_delta'] == .02
    assert protocol['validation_maps'] == 32 and protocol['validation_seconds'] == 90
    assert protocol['validation_every'] == 5
    assert saved_protocol_options({'protocol': protocol}) == ('mixed-reference', 2)
    assert mixed_reference_protocol_version(protocol) == 3
    assert saved_protocol_options({'protocol': dict(protocol, candidate_slots_per_map=8)}) is None
    v2 = protocol_settings('mixed-reference', 2, mixed_version=2)
    assert v2['version'] == 'mixed-reference-v2'
    assert v2['maps_per_game'] == 32 and v2['candidate_slots_per_map'] == 8
    assert v2['reference_slots_per_map'] == 8 and v2['seconds_per_game'] == 90
    assert 'sensor_chunk' not in v2
    assert saved_protocol_options({'protocol': v2}) == ('mixed-reference', 2)
    assert mixed_reference_protocol_version(v2) == 2
    v1 = protocol_settings('mixed-reference', 2, mixed_version=1)
    assert v1['version'] == 'mixed-reference-v1'
    assert v1['seconds_per_game'] == 45
    assert saved_protocol_options({'protocol': v1}) == ('mixed-reference', 2)
    assert mixed_reference_protocol_version(v1) == 1
    with pytest.raises(ValueError, match='exactly two'):
        protocol_settings('mixed-reference', 1)


def test_mixed_reference_v4_uses_four_games_and_preserves_v3_record():
    v3 = protocol_settings('mixed-reference', 2)
    assert v3['version'] == 'mixed-reference-v3'
    assert v3['games_per_genome'] == 2
    assert v3['matchmaking'] == 'cohort_shuffle_four_seat_generation_rotation_v1'

    v4 = protocol_settings('mixed-reference', 4)
    assert v4['version'] == 'mixed-reference-v4'
    assert v4['population'] == 256
    assert v4['maps_per_game'] == 64 and v4['worms_per_map'] == 16
    assert v4['candidate_slots_per_map'] == 4 and v4['reference_slots_per_map'] == 12
    assert v4['games_per_genome'] == 4 and v4['seconds_per_game'] == 90
    assert v4['sensor_chunk'] == 8
    assert v4['aggregate'] == 'arithmetic_mean_four_games'
    assert v4['stagnation_metric'] == 'within_generation_midrank_percentile'
    assert v4['validation_maps'] == 32 and v4['validation_seconds'] == 90
    assert v4['validation_every'] == 5
    assert saved_protocol_options({'protocol': v4}) == ('mixed-reference', 4)
    assert mixed_reference_protocol_version(v4) == 4
    assert recognized_protocol(v4) == ('mixed-reference', 4)
    with pytest.raises(ValueError, match='exactly 4'):
        protocol_settings('mixed-reference', 2, mixed_version=4)


def test_server_rejects_selfplay_shape_before_creating_run(monkeypatch):
    monkeypatch.setattr(server, 'process', None)
    with pytest.raises(HTTPException) as error:
        server.start(server.StartOptions(opponent_mode='selfplay'))
    assert error.value.status_code == 400
    assert 'maps × worms = population' in error.value.detail


def test_server_rejects_mixed_reference_with_nonpilot_geometry(monkeypatch):
    monkeypatch.setattr(server, 'process', None)
    with pytest.raises(HTTPException) as error:
        server.start(server.StartOptions(opponent_mode='mixed-reference', maps=32))
    assert error.value.status_code == 400
    assert 'Mixed-reference' in error.value.detail


def test_server_starts_new_mixed_reference_with_v4_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'process', None)
    monkeypatch.setattr(server, 'current', None)
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    captured = {}

    class RunningProcess:
        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured['command'] = command
        return RunningProcess()

    monkeypatch.setattr(server.subprocess, 'Popen', fake_popen)
    result = server.start(server.StartOptions(opponent_mode='mixed-reference'))
    assert result['started'] is True
    command = captured['command']
    assert command[command.index('--seconds') + 1] == '90'
    assert command[command.index('--maps') + 1] == '64'
    assert command[command.index('--sensor-chunk') + 1] == '8'
    assert command[command.index('--training-games') + 1] == '4'


@pytest.mark.parametrize(('version', 'games', 'maps', 'chunk', 'duration'),
                         [(1, 2, 32, 4, 45), (2, 2, 32, 4, 90),
                          (3, 2, 64, 8, 90), (4, 4, 64, 8, 90)])
def test_server_resume_preserves_mixed_protocol_version(
        tmp_path, monkeypatch, version, games, maps, chunk, duration):
    import dataclasses
    from slitherai.config import SimConfig

    monkeypatch.setattr(server, 'process', None)
    run = tmp_path / f'mixed-v{version}'
    run.mkdir()
    (run / 'checkpoint-7').write_bytes(b'checkpoint marker')
    config = dataclasses.asdict(SimConfig(maps=maps, worms=16, sensor_chunk=chunk))
    from slitherai.io import write_json
    write_json(run / 'settings.json', dict(
        protocol=protocol_settings('mixed-reference', games, mixed_version=version),
        config=config, population=256, training_games=games,
        seconds=duration, seed=37, validation_every=5))
    monkeypatch.setattr(server, 'current', run)
    captured = {}

    class RunningProcess:
        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured['command'] = command
        return RunningProcess()

    monkeypatch.setattr(server.subprocess, 'Popen', fake_popen)
    server.start(server.StartOptions(resume=True))
    command = captured['command']
    assert command[command.index('--seconds') + 1] == str(duration)
    assert command[command.index('--maps') + 1] == str(maps)
    assert command[command.index('--sensor-chunk') + 1] == str(chunk)
    assert command[command.index('--training-games') + 1] == str(games)
    assert command[command.index('--opponent-mode') + 1] == 'mixed-reference'


@pytest.mark.parametrize(('version', 'games'), [(1, 2), (2, 2), (3, 2), (4, 4)])
def test_all_mixed_protocol_versions_are_recognized(version, games):
    protocol = protocol_settings('mixed-reference', games, mixed_version=version)
    assert recognized_protocol(protocol) == ('mixed-reference', games)
    assert mixed_reference_protocol_version(protocol) == version
