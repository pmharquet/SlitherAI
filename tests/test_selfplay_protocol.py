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
    assert protocol['version'] == 'population-selfplay-v1'
    assert protocol['games_per_genome'] == 3
    assert protocol['opponents'] == 'co_evolving_population'
    assert protocol['validation_opponents'] == 'fixed_food_and_avoidance'
    assert saved_protocol_options({'protocol': protocol}) == ('selfplay', 3)
    assert saved_protocol_options({'protocol': dict(protocol, aggregate='mean_only')}) is None
    with pytest.raises(ValueError):
        protocol_settings('reference', 3)
    with pytest.raises(ValueError):
        protocol_settings('selfplay', 0)


def test_server_rejects_selfplay_shape_before_creating_run(monkeypatch):
    monkeypatch.setattr(server, 'process', None)
    with pytest.raises(HTTPException) as error:
        server.start(server.StartOptions(opponent_mode='selfplay'))
    assert error.value.status_code == 400
    assert 'maps × worms = population' in error.value.detail
