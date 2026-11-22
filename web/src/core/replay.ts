/**
 * Replay of a recorded generation: the static scene of a segment (columns of nodes, the attention links drawn
 * while the source context is read) and, for any moment on the timeline, which links, nodes and pulses are lit.
 * Port of scene_for() and the drawing rules of DemoApp::draw_scene() in cpp/src/demo_app.cpp, with the
 * rendering taken out: this module only says what to draw, in terms of node indices, so it can be tested
 * without a canvas and drawn by any renderer.
 *
 * While the source is read, a wave of light travels along the prefill links and across the stages. While a
 * token is written, the last TRAIL steps stay visible, fading by a factor of 0.55 per step; the current step
 * reveals layer by layer: residual, strongest self- and cross-attention sources, then the candidate tokens.
 */

import { STEEL, WHITE, hex, mix, type RGB } from './colors.ts';
import type { Now, Segment, TraceDemo } from './demo.ts';
import { hash01 } from './math3d.ts';
import { linkColor } from './net.ts';

/** Previous steps that stay visible behind the current one. */
export const TRAIL = 5;
/** Playback speeds offered to the user. */
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16] as const;

const NODE_BASE: RGB = hex('#d9d9df');
const NODE_DIM: RGB = hex('#6c6e75');
const CANDIDATE: RGB = hex('#9a9ca5');
/** Colour of the small lights that travel along the strongest links of the current step. */
export const PULSE_COLOR: RGB = WHITE;

export interface Scene {
  nStages: number;
  stageX: Float32Array;
  halfSpan: number;
  first: Int32Array;
  count: Int32Array;
  nodes: { n: number; x: Float32Array; y: Float32Array; z: Float32Array; stage: Int32Array; idx: Int32Array; act: Float32Array };
  edges: { n: number; a: Int32Array; b: Int32Array; w: Float32Array; stage: Int32Array; tint: RGB[] };
  groups: { a: number; b: number; label: string }[];
}

/** Builds the static scene of one segment. */
export function buildScene(sg: Segment): Scene {
  const ns = sg.stages.length;
  const stageX = new Float32Array(ns);
  let curX = 0;
  for (let i = 0; i < ns; i++) {
    if (i > 0) curX += 40 + sg.stages[i].gapBefore;
    stageX[i] = curX;
  }
  const first = new Int32Array(ns);
  const count = new Int32Array(ns);
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const stageOf: number[] = [];
  const idxOf: number[] = [];
  const act: number[] = [];
  sg.stages.forEach((st, s) => {
    first[s] = xs.length;
    count[s] = st.count;
    const a = st.act;
    let amax = 1e-6;
    if (a) for (let i = 0; i < a.length; i++) amax = Math.max(amax, a[i]);
    for (let i = 0; i < st.count; i++) {
      xs.push(stageX[s] + (hash01(s, i + 9000) - 0.5) * st.xJitter);
      ys.push((0.5 - (i + 0.5) / Math.max(1, st.count)) * 2 * st.spanY);
      zs.push((hash01(s, i) - 0.5) * 2 * st.zJitter);
      stageOf.push(s);
      idxOf.push(i);
      act.push(a && i < a.length ? Math.min(Math.max(a[i] / amax, 0.1), 1) : 0.25);
    }
  });

  const groups: Scene['groups'] = [];
  for (let s = 0; s < ns; ) {
    const grp = sg.stages[s].group;
    let e = s;
    while (e + 1 < ns && sg.stages[e + 1].group === grp) e++;
    groups.push({ a: s, b: e, label: grp || sg.stages[s].label });
    s = e + 1;
  }

  const ea: number[] = [];
  const eb: number[] = [];
  const ew: number[] = [];
  const es: number[] = [];
  const et: RGB[] = [];
  for (const pl of sg.prefillLinks) {
    const dstCnt = pl.ids.shape[0];
    const K = pl.ids.shape[1];
    const col = linkColor(pl);
    const ids = pl.ids.int32();
    const ws = pl.weights.float32();
    for (let b = 0; b < dstCnt && b < count[pl.dst]; b++) {
      for (let kk = 0; kk < K; kk++) {
        const a = ids[b * K + kk];
        const w = ws[b * K + kk];
        if (a >= 0 && a < count[pl.src] && w > 0.02) {
          ea.push(first[pl.src] + a);
          eb.push(first[pl.dst] + b);
          ew.push(w);
          es.push(pl.dst);
          et.push(col);
        }
      }
    }
  }

  return {
    nStages: ns,
    stageX,
    halfSpan: curX * 0.5,
    first,
    count,
    nodes: {
      n: xs.length,
      x: Float32Array.from(xs),
      y: Float32Array.from(ys),
      z: Float32Array.from(zs),
      stage: Int32Array.from(stageOf),
      idx: Int32Array.from(idxOf),
      act: Float32Array.from(act),
    },
    edges: { n: ea.length, a: Int32Array.from(ea), b: Int32Array.from(eb), w: Float32Array.from(ew), stage: Int32Array.from(es), tint: et },
    groups,
  };
}

const normMaxCache = new WeakMap<Segment, number>();

/** Largest first-channel activation norm of a segment's tape; the divisor that maps norms to 0..1. */
export function normMax(sg: Segment): number {
  let v = normMaxCache.get(sg);
  if (v === undefined) {
    const [M, Ld, C] = sg.tape.norm.shape;
    const d = sg.tape.norm.float32();
    v = 1e-6;
    for (let i = 0; i < M; i++) for (let l = 0; l < Ld; l++) v = Math.max(v, d[(i * Ld + l) * C]);
    normMaxCache.set(sg, v);
  }
  return v;
}

const clamp01 = (v: number): number => Math.min(Math.max(v, 0), 1);

export interface FrameLine {
  a: number;
  b: number;
  color: RGB;
  alpha: number;
}

export interface FrameNode {
  color: RGB;
  alpha: number;
  /** Radius before perspective. */
  radius: number;
}

export interface FrameHalo {
  node: number;
  radius: number;
  color: RGB;
  alpha: number;
  label?: string;
  labelDim?: boolean;
}

/** What is lit at one moment of a segment. */
export interface Frame {
  /** Additive lines between node indices. */
  lines: FrameLine[];
  /** Look per node; null for the candidate column, which is drawn only as halos. */
  nodes: (FrameNode | null)[];
  /** Small lights travelling along a link, at fraction `t` of its length. */
  pulses: { a: number; b: number; t: number; alpha: number }[];
  halos: FrameHalo[];
  /** Which stage groups are highlighted. */
  groupLit: boolean[];
  caption: string;
  /** Position of the reading wave in stage units, -10 when not reading. */
  wave: number;
}

export function composeFrame(demo: TraceDemo, sg: Segment, scene: Scene, now: Now): Frame {
  const { first, count, nodes } = scene;
  const node = (stage: number, i: number): number => first[stage] + i;
  const tape = sg.tape;
  const Ld = sg.layerStages.length;
  const topK = demo.topK;
  const Kself = tape.selfId.shape[2];
  const Kcross = sg.crossStage >= 0 && tape.crossId ? tape.crossId.shape[2] : 0;
  const Ccross = sg.crossStage >= 0 ? count[sg.crossStage] : 0;
  const reading = now.reading;
  const t = reading ? sg.t0 : now.token;
  const wave = reading ? now.readP * Math.max(1, sg.inStage) : -10;

  const lines: FrameLine[] = [];
  const pulses: Frame['pulses'] = [];
  const halos: FrameHalo[] = [];

  for (let e = 0; e < scene.edges.n; e++) {
    const boost = reading ? 0.45 + 2.4 * Math.exp(-((scene.edges.stage[e] - 0.5 - wave) ** 2) / 1.3) : 0.7;
    lines.push({
      a: scene.edges.a[e],
      b: scene.edges.b[e],
      color: scene.edges.tint[e],
      alpha: Math.min(0.9, (0.03 + 0.14 * Math.sqrt(Math.min(1, scene.edges.w[e]))) * boost),
    });
  }

  const crossUse = new Float32Array(Ccross);
  if (!reading && sg.crossStage >= 0 && Kcross > 0 && tape.crossId && tape.crossW) {
    const ids = tape.crossId.int32();
    const ws = tape.crossW.float32();
    for (let l = 0; l < Ld; l++) {
      for (let kk = 0; kk < Kcross; kk++) {
        const at = (t * Ld + l) * Kcross + kk;
        if (ids[at] >= 0 && ids[at] < Ccross) crossUse[ids[at]] += ws[at] / Ld;
      }
    }
  }
  let crossMax = 1e-6;
  for (const v of crossUse) crossMax = Math.max(crossMax, v);

  const topIds = tape.topId.int32();
  const topPs = tape.topP.float32();
  if (!reading) {
    const Pn = count[sg.inStage];
    const selfId = tape.selfId.int32();
    const selfW = tape.selfW.float32();
    const crossIds = tape.crossId?.int32() ?? null;
    const crossWs = tape.crossW?.float32() ?? null;
    for (let age = TRAIL; age >= 0; age--) {
      const tt = t - age;
      if (tt < sg.t0 || tt - 1 >= Pn) continue;
      const fade = 0.55 ** age;
      const q = tt - 1;
      for (let l = 0; l < Ld; l++) {
        const reveal = age === 0 ? clamp01(now.frac * (Ld + 1) - l + 1) : 1;
        if (reveal <= 0) continue;
        const inSt = l === 0 ? sg.inStage : sg.layerStages[l - 1];
        const outSt = sg.layerStages[l];
        const dst = node(outSt, q);
        lines.push({ a: node(inSt, q), b: dst, color: WHITE, alpha: 0.16 * fade * reveal });
        for (let kk = 0; kk < Kself; kk++) {
          const at = (tt * Ld + l) * Kself + kk;
          const id = selfId[at];
          const w = selfW[at];
          if (id < 0 || id >= Pn || w < 0.03) continue;
          lines.push({
            a: node(inSt, id),
            b: dst,
            color: mix(NODE_DIM, WHITE, Math.min(1, w * 1.8)),
            alpha: Math.min(1, (0.06 + 0.7 * Math.sqrt(w)) * fade * reveal),
          });
        }
        if (sg.crossStage >= 0 && Kcross > 0 && crossIds && crossWs) {
          for (let kk = 0; kk < Kcross; kk++) {
            const at = (tt * Ld + l) * Kcross + kk;
            const id = crossIds[at];
            const w = crossWs[at];
            if (id < 0 || id >= Ccross || w < 0.03) continue;
            lines.push({
              a: node(sg.crossStage, id),
              b: dst,
              color: mix(STEEL, WHITE, Math.min(1, w * 1.6)),
              alpha: Math.min(1, (0.1 + 0.85 * Math.sqrt(w)) * fade * reveal),
            });
          }
        }
      }
      if (age === 0 && Ld > 0) {
        const last = node(sg.layerStages[Ld - 1], q);
        const cands = Math.min(topK, count[sg.outStage]);
        const reveal = clamp01(now.frac * (Ld + 1) - Ld + 1);
        for (let o = 0; o < cands; o++) {
          if (topIds[tt * topK + o] < 0) continue;
          lines.push({
            a: last,
            b: node(sg.outStage, o),
            color: o === 0 ? WHITE : NODE_DIM,
            alpha: (0.08 + 0.9 * topPs[tt * topK + o]) * reveal,
          });
        }
      }
    }
    for (let l = 0; l < Ld; l++) {
      const q = Math.min(Math.max(t - 1, 0), count[sg.layerStages[l]] - 1);
      const dst = node(sg.layerStages[l], q);
      const inSt = l === 0 ? sg.inStage : sg.layerStages[l - 1];
      for (let kk = 0; kk < Math.min(3, Kself); kk++) {
        const at = (t * Ld + l) * Kself + kk;
        const id = selfId[at];
        const w = selfW[at];
        if (id < 0 || id >= count[inSt] || w < 0.14) continue;
        pulses.push({ a: node(inSt, id), b: dst, t: (now.frac * 1.3 + hash01(id, l) * 0.3) % 1, alpha: Math.min(1, w * 2) });
      }
    }
  }

  const nmax = reading ? 1 : normMax(sg);
  const nChan = tape.norm.shape[2];
  const normData = tape.norm.float32();
  const look: (FrameNode | null)[] = new Array<FrameNode | null>(nodes.n).fill(null);
  for (let i = 0; i < nodes.n; i++) {
    const stage = nodes.stage[i];
    if (stage === sg.outStage) continue;
    const crowd = Math.min(Math.max(70 / Math.max(1, count[stage]), 0.22), 1);
    let g: number;
    let color: RGB;
    let alpha: number;
    let radius: number;
    if (stage >= sg.inStage) {
      const q = nodes.idx[i];
      const written = !reading && q + 1 <= t;
      const layer = sg.layerStages.indexOf(stage);
      let actV = 0.2;
      if (written && layer >= 0) actV = normData[((q + 1) * Ld + layer) * nChan] / nmax;
      g = written ? Math.min(1, actV) : 0;
      color = mix(NODE_DIM, WHITE, g);
      alpha = written ? Math.min(1, (0.12 + 0.8 * g) * (Math.sqrt(crowd) + 0.15)) : 0.1;
      radius = (0.9 + 1.4 * g) * Math.sqrt(crowd) + 0.3;
    } else if (stage === sg.crossStage) {
      g = reading ? 0.25 : Math.sqrt(crossUse[nodes.idx[i]] / crossMax);
      color = mix(NODE_DIM, STEEL, 0.4 + 0.6 * g);
      alpha = Math.min(1, 0.35 + 0.65 * g);
      radius = 2.8 + 2.2 * g;
    } else {
      const lit = reading ? 0.35 + 1.3 * Math.exp(-((stage - wave) ** 2) / 1.3) : 1;
      g = nodes.act[i] * 0.6;
      color = mix(NODE_BASE, WHITE, Math.min(1, g * 0.9));
      alpha = Math.min(1, (0.1 + 0.4 * nodes.act[i]) * (Math.sqrt(crowd) + 0.15) * lit);
      radius = (0.9 + nodes.act[i]) * Math.sqrt(crowd) + 0.3;
    }
    look[i] = { color, alpha, radius };
  }

  if (!reading && Ld > 0 && t - 1 >= 0 && t - 1 < count[sg.layerStages[Ld - 1]]) {
    halos.push({
      node: node(sg.layerStages[Ld - 1], t - 1),
      radius: 12 * (0.7 + 0.3 * Math.sin(now.frac * 6.2832)),
      color: WHITE,
      alpha: 0.9,
    });
    const writtenId = sg.ftok[t];
    const cands = Math.min(topK, count[sg.outStage]);
    for (let o = 0; o < cands; o++) {
      const id = topIds[t * topK + o];
      if (id < 0) continue;
      const pr = topPs[t * topK + o];
      const chosen = id === writtenId;
      halos.push({
        node: node(sg.outStage, o),
        radius: 4 + 9 * Math.sqrt(pr),
        color: chosen ? WHITE : CANDIDATE,
        alpha: chosen ? 0.95 : 0.55,
        label: demo.label(id),
        labelDim: !chosen,
      });
    }
  }

  const groupLit = scene.groups.map((g) => (reading ? wave + 0.5 >= g.a && wave - 0.5 <= g.b : g.a >= sg.inStage));
  const caption = reading
    ? `${sg.label}: encoding source context (${Math.floor(now.readP * 100)}%)`
    : `${sg.label}: generating token ${t} of ${sg.ftok.length - 1}`;
  return { lines, nodes: look, pulses, halos, groupLit, caption, wave };
}

/** Colours of segments on the timeline: no reference, matches the reference, or differs from it. */
export function segmentTone(sg: Segment): 'plain' | 'exact' | 'differs' {
  if (sg.refTok === null) return 'plain';
  return sg.freeExact ? 'exact' : 'differs';
}

/** Timeline position after stepping `by` tokens from the current one, or from the end of reading. */
export function stepPosition(demo: TraceDemo, pos: number, by: number): number {
  const n = demo.locate(pos);
  const sg = demo.segments[n.seg];
  const at = n.reading ? sg.start + sg.readTicks : sg.start + sg.readTicks + (n.token - sg.t0) + 0.5;
  return Math.min(Math.max(at + by, 0), Math.max(0, demo.total - 1e-6));
}
