(() => {
  'use strict';
  const id = new URLSearchParams(location.search).get('session');
  const $ = name => document.getElementById(name);
  const DB_NAME = 'slitherai-lidar';
  let db;
  let session;

  function requestResult(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  function openDb() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  function sampleForExport(sample) {
    const { rayBytes, ...lidar } = sample.lidar;
    return { ...sample, lidar: { ...lidar, rays: SlitherAIFormat.unpack(rayBytes) } };
  }
  function batchAfter(index) {
    if (index >= session.lastIndex) return Promise.resolve([]);
    const range = IDBKeyRange.bound([id, index + 1], [id, session.lastIndex]);
    return requestResult(db.transaction('samples').objectStore('samples').getAll(range, 64));
  }
  async function* lines() {
    yield JSON.stringify({ type: 'session', exportFormat: 'slitherai-jsonl-v1', ...session }) + '\n';
    let lastIndex = -1;
    for (;;) {
      const batch = await batchAfter(lastIndex);
      if (!batch.length) break;
      for (const sample of batch) {
        yield JSON.stringify({ type: 'sample', ...sampleForExport(sample) }) + '\n';
        lastIndex = sample.index;
      }
      $('progress').textContent = `${lastIndex + 1} / ${session.samples} mesures exportées`;
    }
  }
  async function download() {
    $('download').disabled = true;
    $('error').textContent = '';
    const filename = `slitherai-${new Date(session.startedAt).toISOString().replace(/[:.]/g, '-')}-${session.id.slice(0, 8)}.jsonl`;
    try {
      if ('showSaveFilePicker' in window) {
        const handle = await window.showSaveFilePicker({ suggestedName: filename,
          types: [{ description: 'Données JSONL', accept: { 'application/x-ndjson': ['.jsonl'] } }] });
        const output = await handle.createWritable();
        try { for await (const line of lines()) await output.write(line); await output.close(); }
        catch (error) { await output.abort(); throw error; }
      } else {
        const chunks = [];
        for await (const line of lines()) chunks.push(line);
        const url = URL.createObjectURL(new Blob(chunks, { type: 'application/x-ndjson' }));
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      }
      $('progress').textContent = `${session.samples} mesures exportées dans ${filename}`;
    } catch (error) {
      if (error?.name !== 'AbortError') $('error').textContent = String(error?.message || error);
    } finally { $('download').disabled = false; }
  }
  async function init() {
    if (!id) throw new Error('Identifiant de session manquant.');
    db = await openDb();
    session = await requestResult(db.transaction('sessions').objectStore('sessions').get(id));
    if (!session) throw new Error('Session introuvable.');
    $('summary').textContent = `${new Date(session.startedAt).toLocaleString('fr-FR')} · ${session.samples} mesures · ${session.status} · score final ${session.finalScore ?? '?'} · rang final ${session.finalRank ?? '?'}`;
    const first = (await batchAfter(-1))[0];
    if (first) {
      const sample = sampleForExport(first);
      sample.lidar.rays = sample.lidar.rays.slice(0, 3);
      sample.lidar.visibleFood = sample.lidar.visibleFood.slice(0, 3);
      sample.lidar.visibleSnakes = sample.lidar.visibleSnakes.slice(0, 2);
      $('preview').textContent = JSON.stringify({ type: 'sample', ...sample }, null, 2);
      $('download').disabled = false;
    } else $('preview').textContent = 'Aucune mesure dans cette session.';
  }
  $('download').addEventListener('click', download);
  init().catch(error => { $('error').textContent = String(error?.message || error); $('summary').textContent = 'Export indisponible'; });
})();
