"""Loopback-only dashboard. Opening it never launches training."""
import argparse
import atexit
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
import fastapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .io import read_json, write_json
from .protocol import protocol_settings, recognized_protocol

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
app = fastapi.FastAPI(title='SlitherAI Lab')
app.mount('/assets', StaticFiles(directory=ROOT / 'web'), name='assets')
process = None
current = None

class StartOptions(BaseModel):
    maps: int = Field(default=64, ge=1, le=64)
    worms: int = Field(default=16, ge=2, le=64)
    population: int = Field(default=256, ge=4, le=4096)
    generations: int = Field(default=50, ge=1, le=10000)
    seconds: float = Field(default=90, ge=5, le=600)
    device: str = 'auto'
    sensor_version: Literal['legacy-v1', 'export-v1'] | None = None
    sensor_chunk: Literal[4, 8, 16] | None = None
    opponent_mode: Literal['reference', 'selfplay', 'mixed-reference'] | None = None
    training_games: int | None = Field(default=None, ge=1, le=16)
    resume: bool = False

class Control(BaseModel):
    pause: bool | None = None
    stop: bool | None = None
    arena: int | None = Field(default=None, ge=0, le=63)
    network_worm: int | None = Field(default=None, ge=-1, le=63)

def run_dir():
    global current
    if current is None:
        last = read_json(RUNS / 'last-run.json', {})
        name = last.get('name')
        if name and Path(name).name == name:
            candidate = RUNS / name
            if candidate.is_dir(): current = candidate
    return current

def active(): return process is not None and process.poll() is None


def resolve_run_id(run_id: str) -> Path:
    """Resolve an explicit run basename without permitting traversal or symlink escape."""
    if (not isinstance(run_id, str) or not run_id or run_id in ('.', '..')
            or Path(run_id).name != run_id or '/' in run_id or '\\' in run_id or ':' in run_id):
        raise fastapi.HTTPException(400, 'Identifiant de session invalide.')
    runs_root = RUNS.resolve()
    candidate = (RUNS / run_id).resolve()
    if candidate.parent != runs_root:
        raise fastapi.HTTPException(400, 'Identifiant de session invalide.')
    if not candidate.is_dir():
        raise fastapi.HTTPException(404, 'Session introuvable.')
    if not (candidate / 'settings.json').is_file() or not (candidate / 'status.json').is_file():
        raise fastapi.HTTPException(404, 'Session introuvable.')
    return candidate


def _attached_to(run: Path | None) -> bool:
    return bool(run and active() and current and current.resolve() == run.resolve())


def _file_age_seconds(path: Path) -> float | None:
    try:
        return max(0., time.time() - path.stat().st_mtime)
    except OSError:
        return None


def saved_protocol_options(settings):
    """Return a recognized run's selection mode and game count, if any."""
    return recognized_protocol(settings.get('protocol'))

@app.get('/')
def index(): return FileResponse(ROOT / 'web' / 'index.html')

@app.get('/api/runs')
def runs():
    """List local run folders for read-only dashboard selection."""
    if not RUNS.is_dir():
        return []
    result = []
    for child in RUNS.iterdir():
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            run = resolve_run_id(child.name)
        except fastapi.HTTPException:
            continue
        settings = read_json(run / 'settings.json', {})
        status = read_json(run / 'status.json', {})
        if not isinstance(settings, dict) or not settings or not isinstance(status, dict) or not status:
            continue
        protocol = saved_protocol_options(settings)
        result.append(dict(
            name=run.name,
            phase=status.get('phase'),
            generation=status.get('generation'),
            target_generation=status.get('target_generation'),
            opponent_mode=status.get('opponent_mode') or settings.get('opponent_mode')
                          or (protocol[0] if protocol else None),
            status_age_seconds=_file_age_seconds(run / 'status.json'),
            preview_available=(run / 'preview.json').is_file(),
        ))
    result.sort(key=lambda item: (item['status_age_seconds'] is None,
                                  item['status_age_seconds'] if item['status_age_seconds'] is not None else math.inf))
    return result

@app.get('/api/state')
def state(run_id: str | None = None):
    explicit_run = run_id is not None
    run = resolve_run_id(run_id) if explicit_run else run_dir()
    attached = _attached_to(run)
    status = read_json(run / 'status.json', {}) if run else {}
    if not isinstance(status, dict):
        status = {}
    if not explicit_run and not active() and status.get('phase') in ('training', 'validating', 'paused'):
        status['phase'] = 'interrupted'
    history = []
    if run and (run / 'history.jsonl').exists():
        import json
        for line in (run / 'history.jsonl').read_text().splitlines():
            try: history.append(json.loads(line))
            except ValueError: pass
    error = None
    if not explicit_run and process is not None and process.poll() not in (None, 0) and run:
        log = run / 'console.log'
        error = log.read_text(encoding='utf-8', errors='replace')[-5000:] if log.exists() else 'Process failed'
    compatible = bool(run and saved_protocol_options(read_json(run / 'settings.json', {})))
    return dict(active=attached if explicit_run else active(), read_only=explicit_run and not attached,
                process_attached=attached if explicit_run else bool(run and active()),
                selected_run=run.name if run else None,
                status_age_seconds=_file_age_seconds(run / 'status.json') if run else None,
                status=status, history=history[-500:],
                species=read_json(run / 'species.json', {}) if run else {},
                species_history=read_json(run / 'species-history.json', [])[-500:] if run else [],
                baselines=read_json(run / 'baselines.json', {}) if run else {},
                can_resume=bool(not explicit_run and compatible and list(run.glob('checkpoint-*'))), error=error)

@app.get('/api/preview')
def preview(run_id: str | None = None):
    run = resolve_run_id(run_id) if run_id is not None else run_dir()
    value = read_json(run / 'preview.json') if run else None
    if run_id is not None and isinstance(value, dict):
        value = dict(value)
        value['monitoring'] = dict(read_only=not _attached_to(run),
                                   process_attached=_attached_to(run),
                                   preview_age_seconds=_file_age_seconds(run / 'preview.json'),
                                   run_id=run.name)
    return value

@app.post('/api/start')
def start(options: StartOptions):
    global process, current
    if active(): raise fastapi.HTTPException(409, 'Un entraînement est déjà actif.')
    if options.device not in ('auto', 'cpu', 'cuda'):
        raise fastapi.HTTPException(400, 'Périphérique invalide.')
    command = [sys.executable, '-m', 'slitherai.train']
    if options.resume:
        run = run_dir()
        if not run: raise fastapi.HTTPException(404, 'Aucune sauvegarde.')
        checkpoints = sorted(run.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        if not checkpoints: raise fastapi.HTTPException(404, 'Aucune sauvegarde.')
        settings = read_json(run / 'settings.json')
        if not settings: raise fastapi.HTTPException(404, 'Configuration de reprise manquante.')
        saved_options = saved_protocol_options(settings)
        if saved_options is None:
            raise fastapi.HTTPException(400, 'Cette ancienne session utilise un autre barème. Démarrez une nouvelle session ; les anciennes sauvegardes restent conservées.')
        saved_mode, saved_games = saved_options
        if options.opponent_mode is not None and options.opponent_mode != saved_mode:
            raise fastapi.HTTPException(400, 'Le mode d’adversaires demandé diffère de la sauvegarde.')
        if options.training_games is not None and options.training_games != saved_games:
            raise fastapi.HTTPException(400, 'Le nombre de parties par génome diffère de la sauvegarde.')
        cfg = settings['config']
        saved_sensor_version = cfg.get('sensor_version', 'legacy-v1')
        if options.sensor_version is not None and options.sensor_version != saved_sensor_version:
            raise fastapi.HTTPException(400, 'Le mode capteur demandé diffère de la sauvegarde. Reprenez son mode ou démarrez une nouvelle session.')
        saved_sensor_chunk = cfg.get('sensor_chunk', 4)
        if options.sensor_chunk is not None and options.sensor_chunk != saved_sensor_chunk:
            raise fastapi.HTTPException(400, 'Le sensor_chunk demandé diffère de la sauvegarde. Reprenez sa valeur ou démarrez une nouvelle session.')
        command += ['--resume', str(checkpoints[-1]), '--maps', str(cfg['maps']), '--worms', str(cfg['worms']),
                    '--foods', str(cfg['foods']), '--body-points', str(cfg['body_points']), '--arena-radius', str(cfg['arena_radius']),
                    '--population', str(settings['population']), '--seconds', str(settings['seconds']), '--seed', str(settings['seed']),
                    '--validation-every', str(settings['validation_every']),
                    '--sensor-version', str(saved_sensor_version),
                    '--sensor-chunk', str(saved_sensor_chunk)]
        if saved_mode in ('selfplay', 'mixed-reference'):
            command += ['--opponent-mode', saved_mode, '--training-games', str(saved_games)]
    else:
        mode = options.opponent_mode or 'reference'
        games = (options.training_games if options.training_games is not None else
                 4 if mode == 'mixed-reference' else 5)
        maps = (64 if mode == 'mixed-reference' and 'maps' not in options.model_fields_set
                else options.maps)
        seconds = (90 if mode == 'mixed-reference' and 'seconds' not in options.model_fields_set
                   else options.seconds)
        sensor_chunk = (8 if mode == 'mixed-reference' and options.sensor_chunk is None
                        else options.sensor_chunk)
        try:
            protocol_settings(mode, games)
        except ValueError as exc:
            raise fastapi.HTTPException(400, str(exc)) from exc
        if mode == 'selfplay' and options.maps * options.worms != options.population:
            raise fastapi.HTTPException(400, 'Self-play exige maps × worms = population (256 génomes, 16 vers : 16 cartes).')
        if mode == 'mixed-reference' and (options.population != 256 or maps != 64
                or options.worms != 16 or seconds != 90 or sensor_chunk != 8):
            raise fastapi.HTTPException(400, 'Mixed-reference v3/v4 exige 256 génomes, 64 cartes, 16 vers, 4 candidats, 12 références, des parties de 90 secondes et sensor_chunk=8.')
        run = RUNS / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        run.mkdir(parents=True)
        command += ['--maps', str(maps), '--worms', str(options.worms), '--population', str(options.population),
                    '--seconds', str(seconds), '--sensor-version', options.sensor_version or 'legacy-v1']
        if sensor_chunk is not None:
            command += ['--sensor-chunk', str(sensor_chunk)]
        if mode in ('selfplay', 'mixed-reference'):
            command += ['--opponent-mode', mode, '--training-games', str(games)]
    current = run
    command += ['--run', str(run), '--generations', str(options.generations), '--device', options.device]
    write_json(run / 'control.json', dict(pause=False, stop=False, arena=0))
    write_json(run / 'status.json', dict(phase='starting'))
    write_json(RUNS / 'last-run.json', dict(name=run.name))
    with (run / 'console.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return dict(started=True, run=run.name)

@app.post('/api/control')
def control(request: Control):
    run = run_dir()
    if not run: raise fastapi.HTTPException(404, 'Aucune session.')
    value = read_json(run / 'control.json', {})
    value.update(request.model_dump(exclude_none=True))
    write_json(run / 'control.json', value)
    return value

def shutdown():
    if active() and current:
        value = read_json(current / 'control.json', {})
        write_json(current / 'control.json', dict(value, stop=True, pause=False))

atexit.register(shutdown)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port)

if __name__ == '__main__': main()
