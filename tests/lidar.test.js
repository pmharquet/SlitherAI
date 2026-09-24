const test = require('node:test');
const assert = require('node:assert/strict');
const { rayCircle, rayCircleInterval, rayCapsule, rayCapsuleInterval, rayArenaExit, compute, angles, beamIndex, bucketize, nextDelayMs } = require('../extension/lidar-main.js');
const format = require('../extension/format.js');
const overlayGeometry = require('../extension/overlay-geometry.js');

const origin = { x: 0, y: 0 };
const east = { x: 1, y: 0 };

test('intersections cercle, capsule et bordure', () => {
  assert.equal(rayCircle(origin, east, { x: 10, y: 0 }, 2), 8);
  assert.deepEqual(rayCircleInterval(origin, east, { x: 10, y: 0 }, 2), [8, 12]);
  assert.equal(rayCircle(origin, east, { x: -10, y: 0 }, 2), null);
  assert.equal(rayCapsule(origin, east, { x: 10, y: -5 }, { x: 10, y: 5 }, 2), 8);
  assert.deepEqual(rayCapsuleInterval(origin, east, { x: 10, y: -5 }, { x: 10, y: 5 }, 2), [8, 12]);
  assert.equal(rayArenaExit(origin, east, { x: 0, y: 0 }, 100), 100);
});

test('index angulaire ne perd aucune intersection', () => {
  let seed = 17;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000);
  const obstacles = Array.from({ length: 150 }, (_, i) => {
    const a = { x: random() * 800 - 400, y: random() * 800 - 400 };
    const radius = 2 + random() * 30;
    return i % 2 ? { shape: 'circle', center: a, radius } :
      { shape: 'capsule', a, b: { x: a.x + random() * 80 - 40, y: a.y + random() * 80 - 40 }, radius };
  });
  const heading = 0.37;
  const buckets = bucketize(obstacles, origin, heading);
  for (let i = 0; i < angles.length; i++) {
    const direction = { x: Math.cos(heading + angles[i]), y: Math.sin(heading + angles[i]) };
    for (const obstacle of obstacles) {
      const hit = obstacle.shape === 'circle'
        ? rayCircleInterval(origin, direction, obstacle.center, obstacle.radius)
        : rayCapsuleInterval(origin, direction, obstacle.a, obstacle.b, obstacle.radius);
      if (hit) assert.ok(buckets[i].includes(obstacle), `intersection perdue sur le rayon ${i}`);
    }
  }
});

function scene() {
  const own = { id: 1, xx: 500, yy: 500, ang: 0, sc: 1, sp: 5, sct: 3, fam: 0.5, pts: [] };
  const enemy = { id: 2, xx: 620, yy: 500, ang: 1, wang: 1.2, sp: 7, sc: 2,
    sct: 3, fam: 0.5, pts: [{ xx: 700, yy: 500 }, { xx: 730, yy: 500 }] };
  const fpsls = []; fpsls[3] = 10;
  const fmlts = []; fmlts[3] = 2;
  return { snake: own, snakes: [own, enemy], foods: [
    { xx: 540, yy: 500, sz: 3 }, { xx: 560, yy: 500, sz: 8 }, { xx: 850, yy: 500, sz: 20 },
    { xx: 545, yy: 515, sz: 2 }
  ], preys: [{ xx: 575, yy: 500, sz: 6, sp: 2, ang: 0 }],
  grd: 500, gsc: 1, view_xx: 500, view_yy: 500, ww: 1200, hh: 800,
  fpsls, fmlts, rank: 12, snake_count: 289, fps: 60, xm: 250, ym: 0,
  lbs: { innerText: '13617\n7692\n4892\n4721' } };
}

test('87 rayons et retours ordonnés en profondeur', () => {
  const data = compute(scene());
  assert.equal(data.status, 'ok');
  assert.equal(data.rays.length, 87);
  assert.equal(angles.length, 87);
  const forward = data.rays.find(ray => ray.relativeAngle === 0);
  const kinds = forward.returns.map(hit => hit.kind);
  assert.deepEqual(kinds.slice(0, 4), ['food', 'food', 'prey', 'enemy_head']);
  assert.ok(kinds.includes('enemy_body'));
  assert.ok(kinds.lastIndexOf('food') > kinds.indexOf('enemy_body'));
  assert.equal(forward.kind, 'enemy_head');
  assert.equal(forward.returns[0].size, 3);
  assert.equal(forward.returns[1].size, 8);
  assert.ok(forward.returns.every((hit, i) => i === 0 || hit.distance >= forward.returns[i - 1].distance));
  assert.equal(forward.returns.filter(hit => hit.kind === 'food').length, 3);
});

test('toutes les boules visibles restent disponibles entre les rayons', () => {
  const data = compute(scene());
  assert.equal(data.visibleFood.length, 5);
  assert.equal(data.counts.visibleFood, 4);
  assert.equal(data.counts.visiblePrey, 1);
  assert.ok(data.visibleFood.some(food => food.y === 515 && food.sizeRaw === 2));
  assert.equal(data.rays.flatMap(ray => ray.returns).filter(hit => hit.kind === 'food').length, 4);
});

test('angles autour du nord et du passage 0°/360° gardent tous les faisceaux', () => {
  for (let heading = -Math.PI * 3; heading <= Math.PI * 3; heading += 0.19) {
    for (let angle = -Math.PI; angle <= Math.PI; angle += 0.23) {
      const relative = Math.atan2(Math.sin(angle - heading), Math.cos(angle - heading));
      const index = beamIndex(angle - heading);
      if (Math.abs(relative) > 170 * Math.PI / 180) assert.equal(index, -1);
      else assert.ok(Number.isInteger(index) && index >= 0 && index < 87);
    }
  }
  const own = { id: 1, xx: 500, yy: 500, ang: Math.PI * 1.5, sc: 1, pts: [] };
  const data = compute({ snake: own, snakes: [own], foods: [
    { xx: 500, yy: 400, sz: 5 }, { xx: 400, yy: 499.9, sz: 4 }
  ], gsc: 1, view_xx: 500, view_yy: 500, ww: 600, hh: 600 });
  assert.equal(data.status, 'ok');
  assert.equal(data.rays.length, 87);
  assert.equal(data.visibleFood.length, 2);
  assert.ok(data.rays.find(ray => ray.relativeAngle === 0).returns.some(hit => hit.kind === 'food'));
  const nearWrap = scene();
  nearWrap.snake.ang = 5.8;
  nearWrap.foods = [{ xx: 400, yy: 499.9, sz: 4 }];
  const wrapped = compute(nearWrap);
  assert.equal(wrapped.status, 'ok');
  assert.equal(wrapped.rays.flatMap(ray => ray.returns).filter(hit => hit.kind === 'food').length, 1);
});

test('zones symétriques et cône mort derrière', () => {
  const deg = value => value * Math.PI / 180;
  assert.equal(angles.filter(a => Math.abs(a) <= deg(45) + 1e-9).length, 45);
  assert.equal(angles.filter(a => Math.abs(a) > deg(45) + 1e-9 && Math.abs(a) <= deg(90) + 1e-9).length, 22);
  assert.equal(angles.filter(a => Math.abs(a) > deg(90) + 1e-9).length, 20);
  assert.ok(angles.every(a => Math.abs(a) <= deg(170) + 1e-9));
  assert.ok(angles.every(a => Math.abs(a) !== deg(180)));
  assert.equal(beamIndex(Math.PI), -1);
  assert.equal(beamIndex(-Math.PI), -1);
  assert.equal(beamIndex(deg(175)), -1);
  assert.equal(beamIndex(deg(170)), angles.length - 1);
  assert.equal(beamIndex(0), 43);

  const own = { id: 1, xx: 500, yy: 500, ang: 0, sc: 1, pts: [] };
  const enemy = { id: 2, xx: 300, yy: 500, ang: 0, sc: 1, pts: [] };
  const data = compute({ snake: own, snakes: [own, enemy],
    foods: [{ xx: 400, yy: 500, sz: 5 }], gsc: 1,
    view_xx: 500, view_yy: 500, ww: 800, hh: 800 });
  assert.equal(data.visibleFood.length, 1);
  assert.equal(data.visibleSnakes.length, 1);
  assert.ok(data.rays.every(ray => !ray.returns.some(hit => hit.kind === 'food' || hit.kind === 'enemy_head')));
});

test('cadence adaptative limite le temps pris par le calcul', () => {
  assert.equal(nextDelayMs(4, 'ok'), 100);
  assert.equal(nextDelayMs(25, 'ok'), 200);
  assert.equal(nextDelayMs(200, 'ok'), 750);
  assert.equal(nextDelayMs(25, 'waiting_for_player'), 500);
  assert.equal(nextDelayMs(25, 'read_error'), 500);
});

test('deux portions séparées du même ver restent deux couches', () => {
  const own = { id: 1, xx: 1000, yy: 1000, ang: 0, sc: 1, pts: [] };
  const enemy = { id: 2, xx: 1100, yy: 1000, ang: 0, sc: 1,
    pts: [{ xx: 1150, yy: 1000 }, { xx: 1900, yy: 1000 }] };
  const data = compute({ snake: own, snakes: [own, enemy], foods: [],
    gsc: 1, view_xx: 1000, view_yy: 1000, ww: 2200, hh: 800, grd: 2000 });
  const forward = data.rays.find(ray => ray.relativeAngle === 0);
  assert.equal(forward.returns.filter(hit => hit.kind === 'enemy_body').length, 2);
});

test('taille, cinématique, score et rang gardent leur provenance', () => {
  const data = compute(scene());
  assert.equal(data.player.segmentCount, 3);
  assert.equal(data.player.segmentFraction, 0.5);
  assert.deepEqual(data.player.score, { value: 133, source: 'client_formula' });
  assert.equal(data.player.rank, 12);
  assert.equal(data.player.playersOnServer, 289);
  assert.equal(data.telemetry.fpsRaw, 60);
  assert.equal(data.telemetry.gameMouseX, 250);
  assert.deepEqual(data.leaderboard.topScores, [
    { rank: 1, score: 13617 }, { rank: 2, score: 7692 },
    { rank: 3, score: 4892 }, { rank: 4, score: 4721 }
  ]);
  assert.equal(data.visibleSnakes[0].bodyRadiusEstimate, 29);
  assert.equal(data.visibleSnakes[0].speedRaw, 7);
  assert.equal(data.visibleSnakes[0].wantedHeading, 1.2);
  assert.deepEqual(data.visibleSnakes[0].score, { value: 133, source: 'client_formula' });
  assert.equal(data.visibleSnakes[0].rank, null);
  assert.equal(compute({ ...scene(), fpsls: undefined, fmlts: undefined }).player.score.value, null);
  assert.deepEqual(compute({ ...scene(), fpsls: undefined, fmlts: undefined, span_length: { textContent: '194' } }).player.score,
    { value: 194, source: 'displayed_length' });
});

test('hors écran censuré, objets hors vue ignorés', () => {
  const own = { id: 1, xx: 0, yy: 0, ang: 0, sc: 1, pts: [] };
  const data = compute({ snake: own, snakes: [own], foods: [{ xx: 1000, yy: 0, sz: 5 }],
    gsc: 1, view_xx: 0, view_yy: 0, ww: 200, hh: 200 });
  const forward = data.rays.find(ray => ray.relativeAngle === 0);
  assert.equal(forward.censored, true);
  assert.equal(forward.range, 100);
  assert.equal(forward.distance, null);
  assert.equal(data.visibleFood.length, 0);
});

test('canvas déplacé ou redimensionné : tête, rayons et souris restent alignés', () => {
  const own = { id: 1, xx: 100, yy: 50, ang: 0, sc: 1, pts: [] };
  let rect = { left: 100, top: 20, width: 1200, height: 600 };
  const mc = { width: 1500, height: 750, getBoundingClientRect: () => rect };
  const game = { snake: own, snakes: [own], foods: [], mc,
    view_xx: 100, view_yy: 50, gsc: 1, mww2: 750, mhh2: 375,
    ww: 1200, hh: 600, innerWidth: 1200, innerHeight: 600 };
  const first = compute(game);
  assert.equal(first.camera.layoutSource, 'game_canvas');
  assert.deepEqual(first.camera.screenHead, { x: 700, y: 320 });
  assert.equal(first.camera.screenScaleX, 0.8);
  assert.equal(first.rays.find(ray => ray.relativeAngle === 0).range, 625);
  assert.equal(first.camera.squareHalfExtentWorld, 350);
  rect = { left: 0, top: 0, width: 900, height: 900 };
  game.innerWidth = 900; game.innerHeight = 900;
  const second = compute(game);
  assert.deepEqual(second.camera.screenHead, { x: 450, y: 450 });
  assert.equal(second.camera.screenScaleX, 0.6);
  assert.equal(second.camera.screenScaleY, 1.2);
  assert.deepEqual(overlayGeometry.rayPoint(second.camera.screenHead, 0, 100,
    second.camera.screenScaleX, second.camera.screenScaleY), { x: 510, y: 450 });
  assert.ok(Math.abs(overlayGeometry.steeringAngle({ x: 650, y: 550 }, { x: 450, y: 450 }, 0, 2, 1) - Math.PI / 4) < 1e-9);
});

test('encodage variable conserve les couches et détecte une troncature', () => {
  const rays = compute(scene()).rays;
  const bytes = format.pack(rays);
  const descriptor = format.inspect(bytes);
  assert.equal(descriptor.rayCount, 87);
  assert.equal(descriptor.returnCount, rays.reduce((sum, ray) => sum + ray.returns.length, 0));
  const restored = format.unpack(bytes);
  const index = rays.findIndex(ray => ray.relativeAngle === 0);
  assert.deepEqual(restored[index].returns.map(hit => hit.kind), rays[index].returns.map(hit => hit.kind));
  assert.ok(Math.abs(restored[index].returns[0].size - 3) < 0.001);
  assert.throws(() => format.inspect(bytes.subarray(0, -1)));
  const largeIndex = format.unpack(format.pack([{ relativeAngle: 0, range: 100, censored: true,
    returns: [{ distance: 10, exitDistance: 12, size: 3, kind: 'food', objectIndex: 70000, sampling: 'sector' }] }]));
  assert.equal(largeIndex[0].returns[0].objectIndex, 70000);
});

test('aucune mesure enregistrable hors partie', () => {
  const own = { id: 1, xx: 0, yy: 0, ang: 0, sc: 1 };
  assert.equal(compute({ playing: false, snake: own }).status, 'waiting_for_player');
  assert.equal(compute({ playing: false, snake: own }).rays.length, 0);
  assert.equal(compute({ snake: own }).status, 'camera_unavailable');
});
