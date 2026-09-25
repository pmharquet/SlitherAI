"""External dashboard views must not take control of an independently trained run."""

import fastapi
import pytest

from slitherai import server
from slitherai.io import write_json


def test_external_run_view_preserves_disk_phase_and_dashboard_selection(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    owned = runs / 'owned'
    owned.mkdir()
    external = runs / 'external'
    external.mkdir()
    write_json(owned / 'settings.json', {'protocol': {'version': 'legacy'}})
    write_json(owned / 'status.json', {'phase': 'stopped'})
    write_json(external / 'settings.json', {'opponent_mode': 'mixed-reference'})
    write_json(external / 'status.json', {'phase': 'training', 'generation': 3})
    write_json(external / 'preview.json', {'arena': 0, 'network': {'genome_id': 7}})
    write_json(runs / 'last-run.json', {'name': 'owned'})
    original_last_run = (runs / 'last-run.json').read_bytes()
    original_status = (external / 'status.json').read_bytes()
    monkeypatch.setattr(server, 'RUNS', runs)
    monkeypatch.setattr(server, 'current', owned)
    monkeypatch.setattr(server, 'process', None)

    listed = server.runs()
    assert {item['name'] for item in listed} == {'owned', 'external'}
    state = server.state(run_id='external')
    assert state['status']['phase'] == 'training'
    assert state['selected_run'] == 'external'
    assert state['read_only'] and not state['active'] and not state['process_attached']
    assert not state['can_resume'] and state['status_age_seconds'] >= 0
    preview = server.preview(run_id='external')
    assert preview['network']['genome_id'] == 7
    assert preview['monitoring']['read_only']
    assert preview['monitoring']['preview_age_seconds'] >= 0
    assert server.current == owned
    assert (runs / 'last-run.json').read_bytes() == original_last_run
    assert (external / 'status.json').read_bytes() == original_status


@pytest.mark.parametrize('run_id', ['..', '../external', '..\\external', 'C:external', 'missing'])
def test_external_run_view_rejects_invalid_paths(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    with pytest.raises(fastapi.HTTPException) as exc:
        server.state(run_id=run_id)
    assert exc.value.status_code in (400, 404)
