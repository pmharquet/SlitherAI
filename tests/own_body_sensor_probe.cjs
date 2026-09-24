// Small stdin/stdout bridge so Python tests exercise the real extension sensor.
'use strict';

const fs = require('node:fs');
const { compute } = require('../extension/lidar-main.js');

const scenario = JSON.parse(fs.readFileSync(0, 'utf8'));
const own = {
  id: 1,
  xx: 0,
  yy: 0,
  ang: 0,
  sc: scenario.radius / 14.5,
  pts: scenario.points.map(([xx, yy]) => ({ xx, yy }))
};
const result = compute({
  snake: own,
  snakes: [own],
  foods: [],
  preys: [],
  gsc: 1,
  view_xx: 0,
  view_yy: 0,
  ww: 4000,
  hh: 4000
});
const ray = result.rays.find(item => Math.round(item.relativeAngle * 180 / Math.PI) === -166);
process.stdout.write(JSON.stringify({ status: result.status, ray }));
