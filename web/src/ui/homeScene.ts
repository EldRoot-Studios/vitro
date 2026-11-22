/**
 * The network that turns slowly behind the start card. It is decoration made by a fixed recipe (the same
 * picture on every visit), not a trace: stages of nodes, each node fed by two nodes of the stage before it.
 */

import { WHITE } from '../core/colors.ts';
import { frameCamera, hash01, type Rect } from '../core/math3d.ts';
import { projectNodes } from '../core/net.ts';
import { addLine, glow, spark } from './canvas.ts';

const STAGES = 16;
const STAGE_GAP = 52;

interface HomeNet {
  n: number;
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  stage: Int32Array;
  /** Edges as node index pairs with a weight 0..1. */
  a: Int32Array;
  b: Int32Array;
  w: Float32Array;
  halfSpan: number;
}

function build(): HomeNet {
  const counts = Array.from({ length: STAGES }, (_, s) => 5 + Math.floor(hash01(s, 1) * 12) - (s === 0 || s === STAGES - 1 ? 3 : 0));
  const first: number[] = [];
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const stage: number[] = [];
  counts.forEach((count, s) => {
    first.push(xs.length);
    for (let i = 0; i < count; i++) {
      xs.push(s * STAGE_GAP + (hash01(s, i + 40) - 0.5) * 10);
      ys.push((0.5 - (i + 0.5) / count) * 2 * (60 + 5 * count));
      zs.push((hash01(s, i) - 0.5) * 90);
      stage.push(s);
    }
  });
  const a: number[] = [];
  const b: number[] = [];
  const w: number[] = [];
  for (let s = 1; s < STAGES; s++) {
    for (let i = 0; i < counts[s]; i++) {
      for (let k = 0; k < 2; k++) {
        const near = Math.round(((i + 0.5) / counts[s]) * counts[s - 1] - 0.5);
        const src = Math.min(counts[s - 1] - 1, Math.max(0, near + Math.round((hash01(s * 97 + i, k) - 0.5) * 4)));
        a.push(first[s - 1] + src);
        b.push(first[s] + i);
        w.push(0.25 + 0.75 * hash01(i * 13 + s, k + 5));
      }
    }
  }
  return {
    n: xs.length,
    x: Float32Array.from(xs),
    y: Float32Array.from(ys),
    z: Float32Array.from(zs),
    stage: Int32Array.from(stage),
    a: Int32Array.from(a),
    b: Int32Array.from(b),
    w: Float32Array.from(w),
    halfSpan: ((STAGES - 1) * STAGE_GAP) / 2,
  };
}

let net: HomeNet | null = null;

/** Draws the network at time `t` (seconds) into `ctx`, filling `view`; the picture is dimmed to sit behind the card. */
export function drawHomeScene(ctx: CanvasRenderingContext2D, view: Rect, t: number): void {
  net ??= build();
  const cam = frameCamera({
    yaw: 0.5 + 0.2 * Math.sin(t * 0.18),
    pitch: 0.24,
    zoom: 1,
    halfSpan: net.halfSpan,
    view,
    xs: net.x,
    ys: net.y,
    zs: net.z,
    fitH: 0.9,
    dropY: 0,
  });
  const s = projectNodes(net, cam);
  const wave = ((t / 8) % 1) * (STAGES - 1);
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineWidth = 1;
  for (let e = 0; e < net.a.length; e++) {
    const stageOfEdge = net.stage[net.b[e]];
    const boost = 1 + 1.6 * Math.exp(-((stageOfEdge - 0.5 - wave) ** 2) / 1.4);
    addLine(ctx, s.x[net.a[e]], s.y[net.a[e]], s.x[net.b[e]], s.y[net.b[e]], WHITE, (0.045 + 0.11 * net.w[e]) * boost * 0.8);
  }
  for (let i = 0; i < net.n; i++) {
    const lit = Math.exp(-((net.stage[i] - wave) ** 2) / 1.6);
    const alpha = (0.28 + 0.5 * lit) * 0.75;
    glow(ctx, s.x[i], s.y[i], (2.2 + 1.8 * lit) * Math.min(1.8, Math.max(0.7, s.scale[i] * 1.1)), WHITE, alpha * 0.55);
    spark(ctx, s.x[i], s.y[i], WHITE, Math.min(1, alpha + 0.2));
  }
  ctx.globalCompositeOperation = 'source-over';
}
