importScripts('format.js');
const DB_NAME = 'slitherai-lidar';
const DB_VERSION = 1;
let dbPromise;

function openDb() {
  if (!dbPromise) dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      db.createObjectStore('sessions', { keyPath: 'id' });
      db.createObjectStore('samples', { keyPath: ['sessionId', 'index'] });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return dbPromise;
}

function resultOf(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getSession(id) {
  const db = await openDb();
  return resultOf(db.transaction('sessions').objectStore('sessions').get(id));
}

async function createSession() {
  const db = await openDb();
  let { contributorId } = await chrome.storage.local.get('contributorId');
  if (!contributorId) {
    contributorId = crypto.randomUUID();
    await chrome.storage.local.set({ contributorId });
  }
  const session = {
    id: crypto.randomUUID(),
    schemaVersion: 'lidar-0.3',
    extensionVersion: chrome.runtime.getManifest().version,
    contributorId,
    startedAt: Date.now(),
    endedAt: null,
    status: 'recording',
    samples: 0,
    lastIndex: -1
  };
  await new Promise((resolve, reject) => {
    const tx = db.transaction('sessions', 'readwrite');
    tx.objectStore('sessions').put(session);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  return session;
}

async function addSample(message) {
  const db = await openDb();
  const session = await getSession(message.sessionId);
  if (!session || session.status !== 'recording') throw new Error('Session inactive');
  if (message.index <= session.lastIndex) return { duplicate: true };
  const lidar = message.sample?.lidar;
  if (!lidar?.rayCount || !lidar.raysBase64) throw new Error('Lidar vide');
  const binary = atob(lidar.raysBase64);
  const rayBytes = Uint8Array.from(binary, char => char.charCodeAt(0));
  if (lidar.rayFormat !== 'slr2_multireturn_v2') throw new Error('Version lidar invalide');
  const parsed = SlitherAIFormat.inspect(rayBytes);
  if (parsed.rayCount !== lidar.rayCount || parsed.returnCount !== lidar.counts?.returns) throw new Error('Format lidar invalide');
  const storedSample = {
    sessionId: message.sessionId,
    index: message.index,
    ...message.sample,
    lidar: { ...lidar, raysBase64: undefined, rayBytes }
  };
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['sessions', 'samples'], 'readwrite');
    tx.objectStore('samples').put(storedSample);
    session.samples++;
    session.lastIndex = message.index;
    tx.objectStore('sessions').put(session);
    tx.oncomplete = () => resolve({ stored: true, samples: session.samples });
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

async function stopSession(message) {
  const session = await getSession(message.sessionId);
  if (!session) throw new Error('Session introuvable');
  session.status = 'stopped';
  session.endedAt = Date.now();
  session.endedReason = ['manual', 'player_left', 'pagehide'].includes(message.reason) ? message.reason : 'unknown';
  session.finalScore = Number.isFinite(message.finalScore) ? message.finalScore : null;
  session.finalRank = Number.isFinite(message.finalRank) ? message.finalRank : null;
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('sessions', 'readwrite');
    tx.objectStore('sessions').put(session);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  return session;
}

async function listSessions() {
  const db = await openDb();
  const sessions = await resultOf(db.transaction('sessions').objectStore('sessions').getAll());
  return sessions.sort((a, b) => b.startedAt - a.startedAt);
}

chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  const handlers = {
    START_SESSION: createSession,
    ADD_SAMPLE: () => addSample(message),
    STOP_SESSION: () => stopSession(message),
    LIST_SESSIONS: listSessions
  };
  if (!Object.hasOwn(handlers, message.type)) return false;
  Promise.resolve().then(handlers[message.type]).then(
    result => respond({ ok: true, result }),
    error => respond({ ok: false, error: String(error?.message || error) })
  );
  return true;
});
