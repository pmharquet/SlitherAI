(function (root) {
  'use strict';
  const magic = [83, 76, 82, 50]; // SLR2
  const headerBytes = 12;
  const rayBytes = 12;
  const returnBytes = 20;
  const kinds = ['none', 'enemy_head', 'enemy_body', 'self_body', 'border', 'food', 'prey'];
  const kindCodes = Object.fromEntries(kinds.map((kind, index) => [kind, index]));
  const samplingCodes = { intersection: 0, sector: 1 };

  function pack(rays) {
    const count = rays.reduce((sum, ray) => sum + ray.returns.length, 0);
    if (rays.length > 65535 || count > 0xffffffff) throw new Error('Trop de retours lidar');
    const buffer = new ArrayBuffer(headerBytes + rays.length * rayBytes + count * returnBytes);
    const view = new DataView(buffer);
    magic.forEach((byte, i) => view.setUint8(i, byte));
    view.setUint16(4, rays.length, true);
    view.setUint32(8, count, true);
    let offset = headerBytes;
    for (const ray of rays) {
      if (ray.returns.length > 65535) throw new Error('Trop de retours sur un rayon');
      view.setFloat32(offset, ray.relativeAngle, true);
      view.setFloat32(offset + 4, ray.range, true);
      view.setUint16(offset + 8, ray.returns.length, true);
      view.setUint8(offset + 10, ray.censored ? 1 : 0);
      offset += rayBytes;
      for (const hit of ray.returns) {
        const index = hit.objectIndex ?? -1;
        if (!Number.isInteger(index) || index < -1 || index > 0xfffffffe) throw new Error('Index d’objet invalide');
        view.setFloat32(offset, hit.distance, true);
        view.setFloat32(offset + 4, hit.exitDistance, true);
        view.setFloat32(offset + 8, hit.size ?? NaN, true);
        view.setUint32(offset + 12, index < 0 ? 0xffffffff : index, true);
        view.setUint8(offset + 16, kindCodes[hit.kind] ?? 0);
        view.setUint8(offset + 17, samplingCodes[hit.sampling] ?? 0);
        offset += returnBytes;
      }
    }
    return new Uint8Array(buffer);
  }
  function inspect(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.byteLength < headerBytes || magic.some((byte, i) => view.getUint8(i) !== byte)) throw new Error('En-tête lidar invalide');
    const rayCount = view.getUint16(4, true);
    const returnCount = view.getUint32(8, true);
    if (headerBytes + rayCount * rayBytes + returnCount * returnBytes !== bytes.byteLength) throw new Error('Longueur lidar invalide');
    let offset = headerBytes, actualReturns = 0;
    for (let i = 0; i < rayCount; i++) {
      if (offset + rayBytes > bytes.byteLength) throw new Error('Rayon lidar tronqué');
      const count = view.getUint16(offset + 8, true);
      offset += rayBytes + count * returnBytes;
      actualReturns += count;
    }
    if (actualReturns !== returnCount || offset !== bytes.byteLength) throw new Error('Retours lidar invalides');
    return { rayCount, returnCount };
  }
  function unpack(bytes) {
    const { rayCount } = inspect(bytes);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const rays = [];
    let offset = headerBytes;
    for (let i = 0; i < rayCount; i++) {
      const ray = { relativeAngle: view.getFloat32(offset, true), range: view.getFloat32(offset + 4, true),
        censored: !!view.getUint8(offset + 10), returns: [] };
      const count = view.getUint16(offset + 8, true);
      offset += rayBytes;
      for (let j = 0; j < count; j++) {
        const size = view.getFloat32(offset + 8, true);
        const index = view.getUint32(offset + 12, true);
        ray.returns.push({ distance: view.getFloat32(offset, true), exitDistance: view.getFloat32(offset + 4, true),
          size: Number.isNaN(size) ? null : size, objectIndex: index === 0xffffffff ? -1 : index,
          kind: kinds[view.getUint8(offset + 16)] || 'none',
          sampling: view.getUint8(offset + 17) === 1 ? 'sector' : 'intersection' });
        offset += returnBytes;
      }
      rays.push(ray);
    }
    return rays;
  }
  const api = { headerBytes, rayBytes, returnBytes, pack, inspect, unpack };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SlitherAIFormat = api;
})(typeof window === 'undefined' ? globalThis : window);
