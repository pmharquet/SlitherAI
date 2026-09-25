import pytest
from fastapi import HTTPException

from slitherai.protocol import protocol_settings
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
    assert protocol['version'] == 'mixed-reference-v1'
    assert protocol['population'] == 256
    assert protocol['maps_per_game'] == 32 and protocol['worms_per_map'] == 16
    assert protocol['candidate_slots_per_map'] == protocol['reference_slots_per_map'] == 8
    assert protocol['games_per_genome'] == 2 and protocol['seconds_per_game'] == 45
    assert protocol['aggregate'] == 'arithmetic_mean_two_games'
    assert protocol['stagnation_metric'] == 'within_generation_midrank_percentile'
    assert protocol['stagnation_delta'] == .02
    assert protocol['validation_maps'] == 32 and protocol['validation_seconds'] == 90
    assert protocol['validation_every'] == 5
    assert saved_protocol_options({'protocol': protocol}) == ('mixed-reference', 2)
    assert saved_protocol_options({'protocol': dict(protocol, maps_per_game=64)}) is None
    with pytest.raises(ValueError, match='exactly two'):
        protocol_settings('mixed-reference', 1)


def test_server_rejects_selfplay_shape_before_creating_run(monkeypatch):
    monkeypatch.setattr(server, 'process', None)
    with pytest.raises(HTTPException) as error:
        server.start(server.StartOptions(opponent_mode='selfplay'))
    assert error.value.status_code == 400
    assert 'maps × worms = population' in error.value.detail


def test_server_rejects_mixed_reference_with_nonpilot_geometry(monkeypatch):
    monkeypatch.setattr(server, 'process', None)
    with pytest.raises(HTTPException) as error:
        server.start(server.StartOptions(opponent_mode='mixed-reference', maps=64))
    assert error.value.status_code == 400
    assert 'Mixed-reference exige' in error.value.detail
