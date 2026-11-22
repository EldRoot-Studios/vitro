/**
 * Perspective camera and deterministic helpers shared by the 3D network and the replay scene.
 * Port of cpp/include/vitro/math3d.hpp; coordinates are in graph units, screen output is in CSS pixels.
 */

/**
 * Deterministic pseudo-random number in [0, 1) from two integers; the same inputs give the same node jitter
 * in every session, so a screenshot and a reload agree.
 */
export function hash01(a: number, b: number): number {
  let h = (Math.imul(a, 0x9e3779b1) ^ Math.imul((b + 0x7f4a7c15) | 0, 0x85ebca6b)) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h = (h ^ (h >>> 12)) >>> 0;
  h = Math.imul(h, 0x297a2d39) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return (h & 0xffffff) / 16777216;
}

/**
 * Indices (into [0, n)) of the `k` largest entries of `v[offset .. offset+n)` that exceed `floor`, largest
 * first. Ties go to the lower index so the result is deterministic.
 */
export function topK(v: ArrayLike<number>, offset: number, n: number, k: number, floor: number): number[] {
  const take = Math.min(k, n);
  const out: number[] = [];
  if (take <= 0) return out;
  for (let i = 0; i < n; i++) {
    const w = v[offset + i];
    if (!(w > floor)) continue;
    if (out.length === take && w <= v[offset + out[take - 1]]) continue;
    let at = out.length === take ? take - 1 : out.length;
    while (at > 0 && v[offset + out[at - 1]] < w) {
      out[at] = out[at - 1];
      at--;
    }
    out[at] = i;
  }
  return out;
}

/** Perspective camera that looks at a row of columns centred on `halfSpan`. */
export class Camera3D {
  cy = 1;
  sy = 0;
  cp = 1;
  sp = 0;
  focal = 1;
  dist = 1;
  ox = 0;
  oy = 0;
  vx = 0;
  vy = 0;
  zoom = 1;
  halfSpan = 0;

  /** Projects a graph point to the screen; writes [screen x, screen y, scale] to `out[at..at+2]`. */
  map(wx: number, wy: number, wz: number, out: Float32Array | number[], at = 0): void {
    const x = wx - this.halfSpan;
    const x1 = x * this.cy + wz * this.sy;
    const z1 = -x * this.sy + wz * this.cy;
    const y2 = wy * this.cp - z1 * this.sp;
    const z2 = wy * this.sp + z1 * this.cp;
    const f = this.focal / Math.max(1e-3, this.dist + z2);
    out[at] = this.vx + (this.ox + x1 * f - this.vx) * this.zoom;
    out[at + 1] = this.vy + (this.oy - y2 * f - this.vy) * this.zoom;
    out[at + 2] = f * this.zoom;
  }
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FrameParams {
  yaw: number;
  pitch: number;
  /** User zoom, 1 = framed. */
  zoom: number;
  /** Half of the graph length along x. */
  halfSpan: number;
  /** Drawing rectangle in CSS pixels. */
  view: Rect;
  xs: Float32Array;
  ys: Float32Array;
  zs: Float32Array;
  /** Share of the view height the graph may fill (0.74 for the network, 0.70 for the replay). */
  fitH: number;
  /** Vertical shift of the view centre in pixels, leaves room for labels below. */
  dropY?: number;
}

/**
 * Builds a camera that frames all points inside `view`, then applies the user's zoom. The fit is computed
 * with zoom 1 from the projected bounding box, so zooming never changes the framing centre.
 */
export function frameCamera({ yaw, pitch, zoom, halfSpan, view, xs, ys, zs, fitH, dropY = 8 }: FrameParams): Camera3D {
  const c = new Camera3D();
  c.cy = Math.cos(yaw);
  c.sy = Math.sin(yaw);
  c.cp = Math.cos(pitch);
  c.sp = Math.sin(pitch);
  c.halfSpan = halfSpan;
  c.dist = Math.max(120, halfSpan * 3.4);
  c.vx = view.x + view.w / 2;
  c.vy = view.y + view.h / 2 + dropY;
  c.ox = c.vx;
  c.oy = c.vy;
  c.focal = c.dist;
  const tmp = new Float32Array(3);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    c.map(xs[i], ys[i], zs[i], tmp);
    if (tmp[0] < minX) minX = tmp[0];
    if (tmp[0] > maxX) maxX = tmp[0];
    if (tmp[1] < minY) minY = tmp[1];
    if (tmp[1] > maxY) maxY = tmp[1];
  }
  if (xs.length === 0) return c;
  const fit = Math.min((0.92 * view.w) / Math.max(1, maxX - minX), (fitH * view.h) / Math.max(1, maxY - minY));
  c.focal *= fit;
  c.ox = c.vx - ((minX + maxX) / 2 - c.vx) * fit;
  c.oy = c.vy - ((minY + maxY) / 2 - c.vy) * fit;
  c.zoom = zoom;
  return c;
}

/** Orbit limit shared by both scenes: pitch stays clear of the poles. */
export const PITCH_LIMIT = 1.45;
