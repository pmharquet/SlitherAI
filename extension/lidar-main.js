/* Passive, page-world lidar. Reads existing client state; never writes game state. */
(function (root) {
  'use strict';
  const EPS = 1e-9;
  const MAX_RANGE = 1000000000;
  const SAFETY_MARGIN = 3;
  const degrees = value => value * Math.PI / 180;
  const BLIND_START = degrees(170);
  // Sorted angles: dense front, medium sides, sparse rear shoulders, no ray directly behind.
  const angles = [
    ...Array.from({ length: 10 }, (_, i) => degrees(-166 + i * 8)),
    ...Array.from({ length: 11 }, (_, i) => degrees(-88 + i * 4)),
    ...Array.from({ length: 45 }, (_, i) => degrees(-44 + i * 2)),
    ...Array.from({ length: 11 }, (_, i) => degrees(48 + i * 4)),
    ...Array.from({ length: 10 }, (_, i) => degrees(94 + i * 8))
  ];

  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
  function point(object) {
    if (!object || !finite(object.xx) || !finite(object.yy)) return null;
    return { x: object.xx + (finite(object.fx) ? object.fx : 0), y: object.yy + (finite(object.fy) ? object.fy : 0) };
  }
  function distance2(a, b) { const x = a.x - b.x, y = a.y - b.y; return x * x + y * y; }
  function dot(a, b) { return a.x * b.x + a.y * b.y; }
  function rayCircle(origin, direction, center, radius) {
    return rayCircleInterval(origin, direction, center, radius)?.[0] ?? null;
  }
  function rayCircleInterval(origin, direction, center, radius) {
    if (!finite(radius) || radius <= 0) return null;
    const offset = { x: origin.x - center.x, y: origin.y - center.y };
    const c = dot(offset, offset) - radius * radius;
    const b = dot(offset, direction);
    const discriminant = b * b - c;
    if (discriminant < 0) return null;
    const root = Math.sqrt(discriminant);
    const enter = -b - root, exit = -b + root;
    return exit >= 0 ? [Math.max(0, enter), exit] : null;
  }
  function rayCapsule(origin, direction, a, b, radius) {
    return rayCapsuleInterval(origin, direction, a, b, radius)?.[0] ?? null;
  }
  function rayCapsuleInterval(origin, direction, a, b, radius) {
    const vx = b.x - a.x, vy = b.y - a.y;
    const length = Math.hypot(vx, vy);
    if (length < EPS) return rayCircleInterval(origin, direction, a, radius);
    const ux = vx / length, uy = vy / length;
    const nx = -uy, ny = ux;
    const ox = origin.x - a.x, oy = origin.y - a.y;
    const alongOrigin = ox * ux + oy * uy;
    const sideOrigin = ox * nx + oy * ny;
    function slab(position, velocity, low, high) {
      if (Math.abs(velocity) < EPS) return position >= low && position <= high ? [-Infinity, Infinity] : null;
      const a = (low - position) / velocity, b = (high - position) / velocity;
      return [Math.min(a, b), Math.max(a, b)];
    }
    const along = slab(alongOrigin, direction.x * ux + direction.y * uy, 0, length);
    const side = slab(sideOrigin, direction.x * nx + direction.y * ny, -radius, radius);
    const intervals = [];
    if (along && side) {
      const enter = Math.max(0, along[0], side[0]);
      const exit = Math.min(along[1], side[1]);
      if (exit >= enter) intervals.push([enter, exit]);
    }
    for (const end of [a, b]) {
      const interval = rayCircleInterval(origin, direction, end, radius);
      if (interval) intervals.push(interval);
    }
    if (!intervals.length) return null;
    return [Math.min(...intervals.map(interval => interval[0])), Math.max(...intervals.map(interval => interval[1]))];
  }
  function rayArenaExit(origin, direction, center, radius) {
    if (!finite(radius) || radius <= 0) return null;
    const ox = origin.x - center.x, oy = origin.y - center.y;
    const c = ox * ox + oy * oy - radius * radius;
    if (c >= 0) return 0;
    const b = ox * direction.x + oy * direction.y;
    return -b + Math.sqrt(b * b - c);
  }
  function hitInRange(hit, range) { return hit !== null && hit <= range; }
  function sizeRadius(snake) {
    // 29 * sc is a community estimate of body width, not the server collision radius.
    return Math.max(6, Math.min(80, (finite(snake?.sc) ? snake.sc : 1) * 14.5));
  }
  function normalizeList(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === 'object') return Object.values(value);
    return [];
  }
  function visibleCircle(center, radius, camera) {
    if (!camera) return true;
    const x = camera.cx + (center.x - camera.x) * camera.scale;
    const y = camera.cy + (center.y - camera.y) * camera.scale;
    const r = radius * camera.scale;
    return x + r >= camera.left && y + r >= camera.top && x - r <= camera.right && y - r <= camera.bottom;
  }
  function visibleCapsule(a, b, radius, camera) {
    if (!camera) return true;
    const ax = camera.cx + (a.x - camera.x) * camera.scale;
    const ay = camera.cy + (a.y - camera.y) * camera.scale;
    const bx = camera.cx + (b.x - camera.x) * camera.scale;
    const by = camera.cy + (b.y - camera.y) * camera.scale;
    const r = radius * camera.scale;
    return Math.max(ax, bx) + r >= camera.left && Math.max(ay, by) + r >= camera.top &&
      Math.min(ax, bx) - r <= camera.right && Math.min(ay, by) - r <= camera.bottom;
  }
  function visiblePath(vertices, radius, camera) {
    const chosen = new Set();
    for (let i = 0; i < vertices.length; i++) {
      if (visibleCircle(vertices[i], radius, camera)) chosen.add(i);
      if (i && distance2(vertices[i], vertices[i - 1]) < 600 * 600 &&
          visibleCapsule(vertices[i], vertices[i - 1], radius, camera)) {
        chosen.add(i - 1);
        chosen.add(i);
      }
    }
    return [...chosen].sort((a, b) => a - b).map(i => vertices[i]);
  }
  function cameraState(game) {
    const scale = finite(game.gsc) && game.gsc > 0 ? game.gsc : null;
    if (!scale || !finite(game.view_xx) || !finite(game.view_yy)) return null;
    const fallbackWidth = finite(game.ww) ? game.ww : (game.innerWidth || 0);
    const fallbackHeight = finite(game.hh) ? game.hh : (game.innerHeight || 0);
    let canvas = game.mc;
    let bounds = canvas?.getBoundingClientRect?.();
    if (!bounds || !(bounds.width > 0) || !(bounds.height > 0)) {
      const candidates = game.document?.querySelectorAll?.('canvas') || [];
      let largestArea = 0;
      for (const candidate of candidates) {
        if (candidate.id === 'slitherai-lidar-overlay' || typeof candidate.getBoundingClientRect !== 'function') continue;
        const candidateBounds = candidate.getBoundingClientRect();
        const area = candidateBounds.width * candidateBounds.height;
        if (area > largestArea && finite(candidate.width) && finite(candidate.height)) {
          canvas = candidate; bounds = candidateBounds; largestArea = area;
        }
      }
    }
    const canvasValid = bounds && finite(canvas.width) && finite(canvas.height) && canvas.width > 0 && canvas.height > 0 &&
      finite(bounds.left) && finite(bounds.top) && finite(bounds.width) && finite(bounds.height) && bounds.width > 0 && bounds.height > 0;
    const width = canvasValid ? canvas.width : fallbackWidth;
    const height = canvasValid ? canvas.height : fallbackHeight;
    if (width <= 0 || height <= 0) return null;
    const cssScaleX = canvasValid ? bounds.width / width : 1;
    const cssScaleY = canvasValid ? bounds.height / height : 1;
    const cssLeft = canvasValid ? bounds.left : 0;
    const cssTop = canvasValid ? bounds.top : 0;
    const viewportWidth = finite(game.innerWidth) ? game.innerWidth : fallbackWidth;
    const viewportHeight = finite(game.innerHeight) ? game.innerHeight : fallbackHeight;
    const left = Math.max(0, (0 - cssLeft) / cssScaleX);
    const top = Math.max(0, (0 - cssTop) / cssScaleY);
    const right = Math.min(width, (viewportWidth - cssLeft) / cssScaleX);
    const bottom = Math.min(height, (viewportHeight - cssTop) / cssScaleY);
    if (right <= left || bottom <= top) return null;
    return {
      x: game.view_xx, y: game.view_yy, scale, width, height,
      left, top, right, bottom, cssLeft, cssTop, cssScaleX, cssScaleY,
      layoutSource: canvasValid ? 'game_canvas' : 'window_fallback',
      cx: finite(game.mww2) ? game.mww2 : width / 2,
      cy: finite(game.mhh2) ? game.mhh2 : height / 2
    };
  }
  function visibleRange(origin, direction, camera) {
    if (!camera) return MAX_RANGE;
    const left = camera.x + (camera.left - camera.cx) / camera.scale;
    const right = camera.x + (camera.right - camera.cx) / camera.scale;
    const top = camera.y + (camera.top - camera.cy) / camera.scale;
    const bottom = camera.y + (camera.bottom - camera.cy) / camera.scale;
    const tx = direction.x > EPS ? (right - origin.x) / direction.x : direction.x < -EPS ? (left - origin.x) / direction.x : Infinity;
    const ty = direction.y > EPS ? (bottom - origin.y) / direction.y : direction.y < -EPS ? (top - origin.y) / direction.y : Infinity;
    return Math.max(0, Math.min(MAX_RANGE, tx, ty));
  }

  function numericDisplay(text) {
    if (typeof text !== 'string') return null;
    const trimmed = text.trim();
    if (!/^\d+$/.test(trimmed) && !/^\d{1,3}(?:[ ,.\u00a0]\d{3})+$/.test(trimmed)) return null;
    const number = Number(trimmed.replace(/[ ,.\u00a0]/g, ''));
    return Number.isSafeInteger(number) && number >= 0 && number <= 1000000000 ? number : null;
  }
  function leaderboardScores(game) {
    const node = game.lbs || game.document?.getElementById?.('lbs');
    if (!node) return [];
    const lines = typeof node.innerText === 'string' ? node.innerText.split(/\r?\n/) : [];
    const values = lines.map(numericDisplay).filter(value => value !== null);
    return values.slice(0, 10).map((score, index) => ({ rank: index + 1, score }));
  }
  function scoreOf(snake, game, own = false) {
    const segmentCount = snake?.sct;
    const fraction = snake?.fam;
    const base = game?.fpsls?.[segmentCount];
    const divisor = game?.fmlts?.[segmentCount];
    if (finite(segmentCount) && finite(fraction) && finite(base) && finite(divisor) && divisor > 0) {
      return { value: Math.floor(15 * (base + fraction / divisor - 1) - 5), source: 'client_formula' };
    }
    if (own) {
      const displayed = numericDisplay(game.span_length?.textContent ?? game.document?.getElementById?.('span_length')?.textContent);
      if (displayed !== null) return { value: displayed, source: 'displayed_length' };
    }
    return { value: null, source: 'unavailable' };
  }
  function snakeInfo(snake, head, vertices, game, own = false, headVisible = true) {
    const size = sizeRadius(snake);
    return {
      id: finite(snake.id) ? snake.id : null, own, x: head.x, y: head.y, headVisible,
      heading: finite(snake.ang) ? snake.ang : null,
      wantedHeading: finite(snake.wang) ? snake.wang : null,
      effectiveHeading: finite(snake.ehang) ? snake.ehang : null,
      direction: finite(snake.dir) ? snake.dir : null,
      speedRaw: finite(snake.sp) ? snake.sp : null,
      scaleRaw: finite(snake.sc) ? snake.sc : null,
      bodyRadiusEstimate: size,
      segmentCount: finite(snake.sct) ? snake.sct : null,
      segmentFraction: finite(snake.fam) ? snake.fam : null,
      visiblePointCount: vertices.length,
      score: scoreOf(snake, game, own),
      rank: own && finite(game.rank) && game.rank > 0 ? game.rank : null,
      playersOnServer: own && finite(game.snake_count) && game.snake_count > 0 ? game.snake_count : null,
      bestRank: own && finite(game.best_rank) && game.best_rank > 0 ? game.best_rank : null,
      colorCode: finite(snake.cv) ? snake.cv : null,
      bodyPoints: vertices
    };
  }
  function beamIndex(angle) {
    const normalized = Math.atan2(Math.sin(angle), Math.cos(angle));
    if (Math.abs(normalized) > BLIND_START + EPS) return -1;
    const next = lowerBound(angles, normalized);
    if (next === 0) return 0;
    if (next === angles.length) return angles.length - 1;
    return normalized - angles[next - 1] <= angles[next] - normalized ? next - 1 : next;
  }
  function mergeBodyReturns(returns) {
    returns.sort((a, b) => a.kind.localeCompare(b.kind) || a.objectIndex - b.objectIndex || a.distance - b.distance);
    const merged = [];
    for (const hit of returns) {
      const last = merged.at(-1);
      if (last && hit.kind === last.kind && hit.objectIndex === last.objectIndex &&
          (hit.kind === 'enemy_body' || hit.kind === 'self_body') && hit.distance <= last.exitDistance + EPS) {
        last.exitDistance = Math.max(last.exitDistance, hit.exitDistance);
      } else merged.push(hit);
    }
    return merged.sort((a, b) => a.distance - b.distance || a.exitDistance - b.exitDistance);
  }
  function lowerBound(values, target) {
    let lo = 0, hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (values[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  function bucketize(obstacles, origin, heading) {
    const buckets = Array.from({ length: angles.length }, () => []);
    for (const obstacle of obstacles) {
      const center = obstacle.shape === 'circle' ? obstacle.center :
        { x: (obstacle.a.x + obstacle.b.x) / 2, y: (obstacle.a.y + obstacle.b.y) / 2 };
      const radius = obstacle.shape === 'circle' ? obstacle.radius :
        obstacle.radius + Math.hypot(obstacle.a.x - obstacle.b.x, obstacle.a.y - obstacle.b.y) / 2;
      const dx = center.x - origin.x, dy = center.y - origin.y;
      const distance = Math.hypot(dx, dy);
      if (distance <= radius) {
        for (const bucket of buckets) bucket.push(obstacle);
        continue;
      }
      const relative = Math.atan2(dy, dx) - heading;
      const centerAngle = Math.atan2(Math.sin(relative), Math.cos(relative));
      const half = Math.asin(Math.min(1, radius / distance)) + 1e-8;
      for (let turn = -1; turn <= 1; turn++) {
        const start = lowerBound(angles, centerAngle - half + turn * Math.PI * 2);
        const end = lowerBound(angles, centerAngle + half + turn * Math.PI * 2 + 1e-8);
        for (let i = start; i < end; i++) buckets[i].push(obstacle);
      }
    }
    return buckets;
  }
  function nextDelayMs(elapsed, status) {
    if (status !== 'ok') return 500;
    return Math.max(100, Math.min(750, elapsed * 8));
  }

  function compute(game) {
    const own = game.snake || game.slither;
    const head = point(own);
    if (game.playing === false || own?.dead || own?.alive_amt === 0 || !head || !finite(own.ang)) {
      return { status: 'waiting_for_player', rays: [], timestampMs: Date.now() };
    }
    const ownRadius = sizeRadius(own);
    const camera = cameraState(game);
    if (!camera) return { status: 'camera_unavailable', rays: [], timestampMs: Date.now() };
    const rawSnakes = normalizeList(game.snakes || game.slithers);
    const obstacles = [];
    const selfBody = [];
    const visibleSnakes = [];
    const visibleFood = [];
    let enemyCount = 0;
    let bodyCount = 0;
    for (const enemy of rawSnakes) {
      if (!enemy || enemy === own || (finite(own.id) && enemy.id === own.id) || enemy.dead || enemy.alive_amt === 0) continue;
      const enemyHead = point(enemy);
      if (!enemyHead) continue;
      const radius = sizeRadius(enemy) + ownRadius + SAFETY_MARGIN;
      const vertices = normalizeList(enemy.pts).filter(vertex => vertex && !vertex.dying).map(point).filter(Boolean);
      const visibleVertices = visiblePath(vertices, radius, camera);
      if (!visibleCircle(enemyHead, radius, camera) && !visibleVertices.length) continue;
      enemyCount++;
      const objectIndex = visibleSnakes.length;
      visibleSnakes.push(snakeInfo(enemy, enemyHead, visibleVertices, game, false, visibleCircle(enemyHead, radius, camera)));
      if (visibleCircle(enemyHead, radius, camera)) obstacles.push({ kind: 'enemy_head', shape: 'circle', center: enemyHead, radius, objectIndex });
      const connected = new Uint8Array(vertices.length);
      for (let j = 1; j < vertices.length; j++) {
        const current = vertices[j];
        if (distance2(current, vertices[j - 1]) < 600 * 600 && visibleCapsule(current, vertices[j - 1], radius, camera)) {
          obstacles.push({ kind: 'enemy_body', shape: 'capsule', a: current, b: vertices[j - 1], radius, objectIndex });
          connected[j] = 1;
          connected[j - 1] = 1;
          bodyCount++;
        }
      }
      for (let j = 0; j < vertices.length; j++) {
        if (!connected[j] && visibleCircle(vertices[j], radius, camera)) {
          obstacles.push({ kind: 'enemy_body', shape: 'circle', center: vertices[j], radius, objectIndex });
        }
      }
      if (vertices.length) {
        const closestEnd = distance2(enemyHead, vertices[0]) < distance2(enemyHead, vertices.at(-1)) ? vertices[0] : vertices.at(-1);
        if (distance2(enemyHead, closestEnd) < (radius * 4) ** 2 && visibleCapsule(enemyHead, closestEnd, radius, camera)) {
          obstacles.push({ kind: 'enemy_body', shape: 'capsule', a: enemyHead, b: closestEnd, radius, objectIndex });
        }
      }
    }
    const ownVertices = normalizeList(own.pts).filter(vertex => vertex && !vertex.dying).map(point).filter(Boolean);
    const ownConnected = new Uint8Array(ownVertices.length);
    const ownBodyRadius = ownRadius * 2 + SAFETY_MARGIN;
    for (let j = 1; j < ownVertices.length; j++) {
      const current = ownVertices[j];
      if (distance2(current, head) < (ownRadius * 2) ** 2) continue;
      if (distance2(ownVertices[j - 1], head) >= (ownRadius * 2) ** 2 &&
          distance2(current, ownVertices[j - 1]) < 600 * 600 && visibleCapsule(current, ownVertices[j - 1], ownBodyRadius, camera)) {
        selfBody.push({ shape: 'capsule', a: current, b: ownVertices[j - 1], radius: ownBodyRadius });
        ownConnected[j] = 1;
        ownConnected[j - 1] = 1;
      }
    }
    for (let j = 0; j < ownVertices.length; j++) {
      const current = ownVertices[j];
      if (!ownConnected[j] && distance2(current, head) >= (ownRadius * 2) ** 2 && visibleCircle(current, ownBodyRadius, camera)) {
        selfBody.push({ shape: 'circle', center: current, radius: ownBodyRadius });
      }
    }
    for (const [source, list] of [['food', game.foods], ['prey', game.preys]]) {
      for (const food of normalizeList(list)) {
        const center = point(food);
        if (!center || food.eaten) continue;
        const size = finite(food.sz) ? food.sz : null;
        const radius = Math.max(3, size ?? 3);
        if (!visibleCircle(center, radius, camera)) continue;
        const dx = center.x - head.x, dy = center.y - head.y;
        visibleFood.push({
          source, id: finite(food.id) ? food.id : null, x: center.x, y: center.y,
          dx, dy, distance: Math.hypot(dx, dy), sizeRaw: size, radiusEstimate: radius,
          colorCode: finite(food.cv) ? food.cv : null,
          heading: finite(food.ang) ? food.ang : null,
          speedRaw: finite(food.sp) ? food.sp : null
        });
      }
    }
    const gameRadius = finite(game.grd) && game.grd > 0 ? game.grd * 0.98 : null;
    const arenaCenter = gameRadius !== null ? { x: game.grd, y: game.grd } : null;
    const directions = angles.map(relativeAngle => ({ x: Math.cos(own.ang + relativeAngle), y: Math.sin(own.ang + relativeAngle) }));
    const obstacleByBeam = bucketize(obstacles, head, own.ang);
    const selfBodyByBeam = bucketize(selfBody, head, own.ang);
    const foodByBeam = Array.from({ length: angles.length }, () => []);
    for (let i = 0; i < visibleFood.length; i++) {
      const food = visibleFood[i];
      const relative = Math.atan2(food.dy, food.dx) - own.ang;
      const index = beamIndex(relative);
      if (index >= 0) foodByBeam[index].push(i);
    }
    const rayData = angles.map((relativeAngle, rayIndex) => {
      const direction = directions[rayIndex];
      const range = visibleRange(head, direction, camera);
      const hits = [];
      for (const obstacle of obstacleByBeam[rayIndex]) {
        const interval = obstacle.shape === 'circle'
          ? rayCircleInterval(head, direction, obstacle.center, obstacle.radius)
          : rayCapsuleInterval(head, direction, obstacle.a, obstacle.b, obstacle.radius);
        if (!interval || !hitInRange(interval[0], range)) continue;
        hits.push({ kind: obstacle.kind, distance: interval[0], exitDistance: Math.min(range, interval[1]), objectIndex: obstacle.objectIndex, sampling: 'intersection', size: obstacle.radius });
      }
      for (const obstacle of selfBodyByBeam[rayIndex]) {
        const interval = obstacle.shape === 'circle'
          ? rayCircleInterval(head, direction, obstacle.center, obstacle.radius)
          : rayCapsuleInterval(head, direction, obstacle.a, obstacle.b, obstacle.radius);
        if (interval && hitInRange(interval[0], range)) hits.push({ kind: 'self_body', distance: interval[0], exitDistance: Math.min(range, interval[1]), objectIndex: -1, sampling: 'intersection', size: obstacle.radius });
      }
      const borderFull = arenaCenter ? rayArenaExit(head, direction, arenaCenter, gameRadius - ownRadius - SAFETY_MARGIN) : null;
      const borderDistance = hitInRange(borderFull, range) ? borderFull : null;
      if (borderDistance !== null) hits.push({ kind: 'border', distance: borderDistance, exitDistance: borderDistance, objectIndex: -1, sampling: 'intersection', size: null });
      for (const foodIndex of foodByBeam[rayIndex]) {
        const food = visibleFood[foodIndex];
        const projected = food.dx * direction.x + food.dy * direction.y;
        if (projected < 0 || projected - food.radiusEstimate > range) continue;
        hits.push({ kind: food.source, distance: Math.max(0, projected - food.radiusEstimate),
          exitDistance: Math.min(range, projected + food.radiusEstimate), objectIndex: foodIndex,
          sampling: 'sector', size: food.sizeRaw });
      }
      const returns = mergeBodyReturns(hits);
      let nearest = null, kind = 'none';
      let enemyHeadDistance = null, enemyBodyDistance = null, selfBodyDistance = null, foodDistance = null, foodSize = null;
      for (const hit of returns) {
        if (hit.kind === 'enemy_head' && enemyHeadDistance === null) enemyHeadDistance = hit.distance;
        if (hit.kind === 'enemy_body' && enemyBodyDistance === null) enemyBodyDistance = hit.distance;
        if (hit.kind === 'self_body' && selfBodyDistance === null) selfBodyDistance = hit.distance;
        if ((hit.kind === 'food' || hit.kind === 'prey') && foodDistance === null) { foodDistance = hit.distance; foodSize = hit.size; }
        if ((hit.kind === 'enemy_head' || hit.kind === 'enemy_body' || hit.kind === 'border') && nearest === null) { nearest = hit.distance; kind = hit.kind; }
      }
      return { relativeAngle, distance: nearest, kind, enemyHeadDistance, enemyBodyDistance, selfBodyDistance, borderDistance, borderFull, foodDistance, foodSize, range, censored: range < MAX_RANGE, returns };
    });
    const headCanvasX = camera.cx + (head.x - camera.x) * camera.scale;
    const headCanvasY = camera.cy + (head.y - camera.y) * camera.scale;
    const screenHead = {
      x: camera.cssLeft + headCanvasX * camera.cssScaleX,
      y: camera.cssTop + headCanvasY * camera.cssScaleY
    };
    const squareHalfExtentWorld = Math.max(0, Math.min(
      headCanvasX - camera.left, camera.right - headCanvasX,
      headCanvasY - camera.top, camera.bottom - headCanvasY
    ) / camera.scale);
    return {
      status: 'ok',
      timestampMs: Date.now(),
      player: snakeInfo(own, head, visiblePath(ownVertices, ownRadius, camera), game, true),
      camera: { scale: camera.scale, screenHead,
        screenScaleX: camera.scale * camera.cssScaleX,
        screenScaleY: camera.scale * camera.cssScaleY,
        viewX: camera.x, viewY: camera.y, width: camera.width, height: camera.height,
        visibleRectCanvas: { left: camera.left, top: camera.top, right: camera.right, bottom: camera.bottom },
        canvasRectCss: { left: camera.cssLeft, top: camera.cssTop,
          width: camera.width * camera.cssScaleX, height: camera.height * camera.cssScaleY },
        layoutSource: camera.layoutSource, squareHalfExtentWorld },
      telemetry: { fpsRaw: finite(game.fps) ? game.fps : null,
        lagMultiplierRaw: finite(game.lag_mult) ? game.lag_mult : null,
        gameMouseX: finite(game.xm) ? game.xm : null, gameMouseY: finite(game.ym) ? game.ym : null,
        serverId: typeof game.bso?.sid === 'string' || finite(game.bso?.sid) ? String(game.bso.sid).slice(0, 32) : null,
        arenaCenter, arenaRadiusEstimate: gameRadius },
      leaderboard: { topScores: leaderboardScores(game) },
      geometry: { radiusModel: 'body_width_29_times_sc_estimate', borderModel: gameRadius === null ? 'unknown' : 'grd_times_0_98_estimate', safetyMargin: SAFETY_MARGIN },
      counts: { enemies: enemyCount, visibleBodySegments: bodyCount, visibleFood: visibleFood.filter(f => f.source === 'food').length,
        visiblePrey: visibleFood.filter(f => f.source === 'prey').length, returns: rayData.reduce((sum, ray) => sum + ray.returns.length, 0) },
      visibleSnakes, visibleFood,
      rays: rayData
    };
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { rayCircle, rayCircleInterval, rayCapsule, rayCapsuleInterval, rayArenaExit, compute, angles, scoreOf, beamIndex, bucketize, nextDelayMs };
    return;
  }
  let previousTickStart = null;
  function tick() {
    const started = root.performance.now();
    const intervalMs = previousTickStart === null ? null : started - previousTickStart;
    previousTickStart = started;
    let status = 'read_error';
    try {
      const data = compute(root);
      status = data.status;
      data.performance = { computeMs: root.performance.now() - started, intervalMs };
      root.postMessage({ channel: 'slitherai:lidar:v1', data }, '*');
    } catch (error) {
      root.postMessage({ channel: 'slitherai:lidar:v1', data: { status: 'read_error', detail: String(error?.message || error), rays: [] } }, '*');
    }
    const elapsed = root.performance.now() - started;
    root.setTimeout(tick, nextDelayMs(elapsed, status));
  }
  root.setTimeout(tick, 100);
})(typeof window === 'undefined' ? globalThis : window);
