(() => {
  'use strict';
  const CHANNEL = 'slitherai:lidar:v1';
  let latest = null;
  let lastPlayable = null;
  let canvas = null;
  let recording = false;
  let sessionId = null;
  let index = 0;
  let pending = false;
  let lastError = null;
  let overlayVisible = true;
  let pointer = null;
  let mouseDown = false;
  let spaceDown = false;

  function send(message) {
    return new Promise((resolve, reject) => chrome.runtime.sendMessage(message, response => {
      if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
      else if (!response?.ok) reject(new Error(response?.error || 'Réponse invalide'));
      else resolve(response.result);
    }));
  }
  function ensureCanvas() {
    if (canvas) return canvas;
    canvas = document.createElement('canvas');
    canvas.id = 'slitherai-lidar-overlay';
    canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483646;';
    document.documentElement.appendChild(canvas);
    return canvas;
  }
  function draw(state) {
    if (!overlayVisible || state.status !== 'ok' || !state.camera?.screenHead) {
      if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const surface = ensureCanvas();
    // Le canvas d'aperçu reste en pixels CSS pour éviter un grand tampon HiDPI à effacer.
    const dpr = 1;
    const width = window.innerWidth, height = window.innerHeight;
    if (surface.width !== Math.round(width * dpr) || surface.height !== Math.round(height * dpr)) {
      surface.width = Math.round(width * dpr);
      surface.height = Math.round(height * dpr);
    }
    const ctx = surface.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const head = state.camera.screenHead;
    const scaleX = state.camera.screenScaleX || state.camera.scale;
    const scaleY = state.camera.screenScaleY || state.camera.scale;
    const color = { enemy_head: '#ff496a', enemy_body: '#ffa643', self_body: '#59a9ff',
      border: '#bd8cff', food: '#ffd84e', prey: '#57f1e9', none: '#64e4ae' };
    for (const ray of state.rays) {
      const angle = state.player.heading + ray.relativeAngle;
      const value = ray.distance === null ? ray.range : ray.distance;
      const pixelsPerWorld = SlitherAIOverlayGeometry.pixelLength(angle, 1, scaleX, scaleY);
      const length = Math.max(0, Math.min(value, 450 / pixelsPerWorld));
      const end = SlitherAIOverlayGeometry.rayPoint(head, angle, length, scaleX, scaleY);
      ctx.beginPath();
      ctx.moveTo(head.x, head.y);
      ctx.lineTo(end.x, end.y);
      ctx.strokeStyle = color[ray.kind] || '#fff';
      ctx.globalAlpha = ray.kind === 'none' ? 0.12 : 0.48;
      ctx.lineWidth = ray.kind === 'none' ? 1 : 1.5;
      ctx.stroke();
      // L'overlay est un aperçu; les retours complets restent dans l'état et l'enregistrement.
      const preview = ray.returns.slice(0, 3);
      const danger = ray.returns.find(hit => ['enemy_head', 'enemy_body', 'border'].includes(hit.kind) && !preview.includes(hit));
      if (danger) preview.push(danger);
      else if (ray.returns.length > 3) preview.push(ray.returns.at(-1));
      for (const hit of preview) {
        const hitLength = hit.distance * pixelsPerWorld;
        if (hitLength > 450) continue;
        const hitPoint = SlitherAIOverlayGeometry.rayPoint(head, angle, hit.distance, scaleX, scaleY);
        ctx.beginPath();
        ctx.arc(hitPoint.x, hitPoint.y,
          hit.kind === 'food' || hit.kind === 'prey' ? Math.max(1.5, Math.min(4, (hit.size || 3) * Math.min(scaleX, scaleY) * 0.35)) : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = color[hit.kind] || '#fff';
        ctx.globalAlpha = hit.sampling === 'sector' ? 0.55 : 0.8;
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(head.x, head.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }
  function currentAction(state) {
    const head = state.camera?.screenHead;
    const steeringAngle = SlitherAIOverlayGeometry.steeringAngle(pointer, head, state.player.heading,
      state.camera?.screenScaleX || state.camera?.scale, state.camera?.screenScaleY || state.camera?.scale);
    return { pointerX: pointer?.x ?? null, pointerY: pointer?.y ?? null, steeringAngle, boost: mouseDown || spaceDown };
  }
  async function storeSample(state) {
    if (!recording || pending || state.status !== 'ok' || !state.rays?.length) return;
    pending = true;
    try {
      const bytes = SlitherAIFormat.pack(state.rays);
      let binary = '';
      for (const byte of bytes) binary += String.fromCharCode(byte);
      await send({
        type: 'ADD_SAMPLE', sessionId, index,
        sample: {
          t: performance.now(), wallTimeMs: Date.now(),
          lidar: {
            status: state.status, timestampMs: state.timestampMs, player: state.player,
            camera: state.camera, telemetry: state.telemetry, leaderboard: state.leaderboard,
            performance: state.performance,
            geometry: state.geometry, counts: state.counts,
            visibleSnakes: state.visibleSnakes, visibleFood: state.visibleFood,
            rayCount: state.rays.length, rayFormat: 'slr2_multireturn_v2',
            raysBase64: btoa(binary)
          },
          action: currentAction(state)
        }
      });
      index++;
      lastError = null;
    } catch (error) { lastError = String(error?.message || error); }
    finally { pending = false; }
  }
  window.addEventListener('message', event => {
    if (event.source !== window || event.data?.channel !== CHANNEL) return;
    const state = event.data.data;
    if (!state || !Array.isArray(state.rays)) return;
    if (state.status === 'ok') lastPlayable = state;
    latest = state;
    draw(state);
    storeSample(state);
    if (recording && state.status === 'waiting_for_player') stop('player_left').catch(error => { lastError = String(error?.message || error); });
  });
  document.addEventListener('pointermove', event => { pointer = { x: event.clientX, y: event.clientY }; }, { capture: true, passive: true });
  document.addEventListener('pointerdown', event => { if (event.button === 0) mouseDown = true; }, { capture: true, passive: true });
  document.addEventListener('pointerup', event => { if (event.button === 0) mouseDown = false; }, { capture: true, passive: true });
  document.addEventListener('keydown', event => { if (event.code === 'Space') spaceDown = true; }, { capture: true, passive: true });
  document.addEventListener('keyup', event => { if (event.code === 'Space') spaceDown = false; }, { capture: true, passive: true });
  window.addEventListener('blur', () => { mouseDown = false; spaceDown = false; });
  window.addEventListener('resize', () => { if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height); });

  async function start() {
    if (recording) return status();
    if (latest?.status !== 'ok') throw new Error('Lidar indisponible : démarrez une partie et vérifiez la caméra du jeu.');
    const session = await send({ type: 'START_SESSION' });
    sessionId = session.id;
    index = 0;
    lastPlayable = latest;
    recording = true;
    lastError = null;
    return status();
  }
  async function stop(reason = 'manual') {
    if (!recording) return status();
    recording = false;
    while (pending) await new Promise(resolve => setTimeout(resolve, 20));
    const id = sessionId;
    sessionId = null;
    await send({ type: 'STOP_SESSION', sessionId: id, reason,
      finalScore: lastPlayable?.player?.score?.value ?? null,
      finalRank: lastPlayable?.player?.rank ?? null });
    return status();
  }
  function status() {
    return {
      recording, sessionId, samples: index, overlayVisible,
      lastError: lastError || (latest?.status === 'read_error' ? latest.detail : null),
      lidarStatus: latest?.status || 'waiting',
      rays: latest?.rays?.length || 0,
      returns: latest?.counts?.returns || 0,
      visibleFood: latest?.counts?.visibleFood || 0,
      visiblePrey: latest?.counts?.visiblePrey || 0,
      visibleSnakes: latest?.counts?.enemies || 0,
      squareSideWorld: Number.isFinite(latest?.camera?.squareHalfExtentWorld) ? latest.camera.squareHalfExtentWorld * 2 : null,
      layoutSource: latest?.camera?.layoutSource ?? null,
      computeMs: latest?.performance?.computeMs ?? null,
      intervalMs: latest?.performance?.intervalMs ?? null,
      score: latest?.player?.score?.value ?? null,
      rank: latest?.player?.rank ?? null,
      playersOnServer: latest?.player?.playersOnServer ?? null,
      topTenThreshold: latest?.leaderboard?.topScores?.at(-1)?.score ?? null,
      nearest: latest?.rays?.reduce((best, ray) => ray.distance !== null && (!best || ray.distance < best.distance) ? { kind: ray.kind, distance: ray.distance } : best, null) || null
    };
  }
  chrome.runtime.onMessage.addListener((message, _sender, respond) => {
    if (message.type === 'REC_STATUS') { respond({ ok: true, result: status() }); return false; }
    if (message.type === 'GET_LIDAR_STATE') { respond({ ok: true, result: latest }); return false; }
    if (message.type === 'TOGGLE_OVERLAY') { overlayVisible = !overlayVisible; if (latest) draw(latest); respond({ ok: true, result: status() }); return false; }
    if (message.type !== 'REC_START' && message.type !== 'REC_STOP') return false;
    (message.type === 'REC_START' ? start() : stop('manual')).then(
      result => respond({ ok: true, result }),
      error => respond({ ok: false, error: String(error?.message || error) })
    );
    return true;
  });
  window.addEventListener('pagehide', () => { if (recording) stop('pagehide').catch(() => {}); });
})();
