const $ = id => document.getElementById(id);
let tabId = null;

function runtime(message) {
  return new Promise((resolve, reject) => chrome.runtime.sendMessage(message, response => {
    if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
    else if (!response?.ok) reject(new Error(response?.error || 'Réponse invalide'));
    else resolve(response.result);
  }));
}
function tab(message) {
  return new Promise((resolve, reject) => chrome.tabs.sendMessage(tabId, message, response => {
    if (chrome.runtime.lastError) reject(new Error('Ouvrez une page de slither.io et rechargez-la après installation.'));
    else if (!response?.ok) reject(new Error(response?.error || 'Réponse invalide'));
    else resolve(response.result);
  }));
}
function showError(error) { $('message').textContent = error ? String(error.message || error) : ''; }

async function refresh() {
  const [active] = await chrome.tabs.query({ active: true, currentWindow: true });
  tabId = active?.id ?? null;
  let state = null;
  try { if (tabId !== null) state = await tab({ type: 'REC_STATUS' }); } catch {}
  $('lidar-status').textContent = state ? `${state.lidarStatus} · ${state.rays} rayons · ${state.returns} retours · calcul ${state.computeMs?.toFixed(0) ?? '?'} ms · ${state.intervalMs > 0 ? (1000 / state.intervalMs).toFixed(1) : '?'} mesures/s` : 'Page de jeu non détectée';
  $('nearest').textContent = state?.nearest ? `Obstacle le plus proche : ${state.nearest.kind}, ${state.nearest.distance.toFixed(1)} unités` : 'Aucun obstacle mesuré';
  $('observed').textContent = state ? `${state.visibleFood} boules · ${state.visiblePrey} proies · ${state.visibleSnakes} vers visibles` : '';
  $('coverage').textContent = state?.squareSideWorld != null
    ? `Carré centré visible : ${state.squareSideWorld.toFixed(0)} unités de côté · ${state.layoutSource === 'game_canvas' ? 'canvas recalé' : 'repère de secours'}`
    : '';
  $('score').textContent = state ? `Score : ${state.score ?? 'inconnu'} · rang : ${state.rank ?? 'inconnu'} / ${state.playersOnServer ?? '?'} · seuil top 10 : ${state.topTenThreshold ?? '?'}` : '';
  $('recording-status').textContent = state?.recording ? `Enregistrement : ${state.samples} vecteurs` : 'Enregistrement arrêté';
  $('toggle-overlay').textContent = state?.overlayVisible ? 'Masquer les rayons' : 'Afficher les rayons';
  $('toggle-overlay').disabled = !state;
  $('start').disabled = !state || state.recording || state.lidarStatus !== 'ok';
  $('stop').disabled = !state?.recording;
  if (state?.lastError) showError(state.lastError);
  const sessions = await runtime({ type: 'LIST_SESSIONS' });
  $('sessions').replaceChildren();
  if (!sessions.length) { $('sessions').textContent = 'Aucune session.'; return; }
  for (const session of sessions) {
    const row = document.createElement('div');
    row.className = 'session';
    row.textContent = `${new Date(session.startedAt).toLocaleString('fr-FR')} · ${session.samples} vecteurs · ${session.status}`;
    if (session.status === 'recording' && session.id !== state?.sessionId) {
      const finish = document.createElement('button');
      finish.textContent = 'Terminer';
      finish.addEventListener('click', async () => {
        try { await runtime({ type: 'STOP_SESSION', sessionId: session.id, reason: 'manual' }); showError(null); await refresh(); }
        catch (error) { showError(error); }
      });
      row.append(finish);
    }
    if (session.samples > 0) {
      const exportButton = document.createElement('button');
      exportButton.textContent = 'Voir / exporter';
      exportButton.addEventListener('click', () => {
        chrome.tabs.create({ url: chrome.runtime.getURL(`export.html?session=${encodeURIComponent(session.id)}`) });
      });
      row.append(exportButton);
    }
    $('sessions').append(row);
  }
}

for (const [id, type] of [['toggle-overlay', 'TOGGLE_OVERLAY'], ['start', 'REC_START'], ['stop', 'REC_STOP']]) {
  $(id).addEventListener('click', async () => {
    try { await tab({ type }); showError(null); await refresh(); }
    catch (error) { showError(error); }
  });
}
refresh().catch(showError);
