(function (root) {
  'use strict';
  function rayPoint(head, angle, distance, scaleX, scaleY) {
    return { x: head.x + Math.cos(angle) * distance * scaleX,
      y: head.y + Math.sin(angle) * distance * scaleY };
  }
  function pixelLength(angle, distance, scaleX, scaleY) {
    return Math.hypot(Math.cos(angle) * distance * scaleX, Math.sin(angle) * distance * scaleY);
  }
  function steeringAngle(pointer, head, heading, scaleX, scaleY) {
    if (!pointer || !head || !(scaleX > 0) || !(scaleY > 0)) return null;
    const angle = Math.atan2((pointer.y - head.y) / scaleY, (pointer.x - head.x) / scaleX) - heading;
    return Math.atan2(Math.sin(angle), Math.cos(angle));
  }
  const api = { rayPoint, pixelLength, steeringAngle };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SlitherAIOverlayGeometry = api;
})(typeof window === 'undefined' ? globalThis : window);
