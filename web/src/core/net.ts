/**
 * The 3D network graph of a trace dump: stages become columns of nodes, residual and attention links become
 * edges, and a backward "attention rollout" from one chosen token says how much every node and edge
 * contributed to that token's prediction. Port of build_net() in cpp/src/view_net.cpp, for arbitrary
 * stage/link graphs (decoder-only, encoder-decoder, latent branches).
 *
 * Rollout: the selected position in the last stage starts with influence 1. Going back over the stages, each
 * stage splits the influence of a node between its residual link (share RHO when attention also exists, else
 * all) and its attention links, which pass it on to their source nodes in proportion to the attention weights.
 */

import { headAverage, sourceText, tokenText } from './analysis.ts';
import { STEEL, WHITE, mix, parseTint, type RGB } from './colors.ts';
import { hash01, topK, type Camera3D } from './math3d.ts';
import type { Link, TraceDump, UnitTrace } from './trace.ts';

/** Seconds for the activation wave to travel the whole network. */
export const WAVE_PERIOD = 8;
/** Share of a node's influence that follows the residual stream when the stage also has attention. */
const RHO = 0.5;
/** Attention weights at or below this are never drawn as edges. */
const EDGE_FLOOR = Math.fround(0.005);
/** At most this many of the strongest edges can become "hot" (carry pulses). */
const HOT_LIMIT = 900;
/** Edges at least this important carry pulses. */
const HOT_MIN_IMP = 0.2;

/** Edge kinds. */
export const EDGE = { RESID: 0, SELF: 1, CROSS: 2, OUT: 3 } as const;

/** Choices for "incoming edges per node". */
export const NET_KS = [1, 2, 3, 4, 6, 8] as const;

/** Colour of the rollout path and of the pulses on it. */
export const ROLLOUT_COLOR: RGB = WHITE;

/**
 * Colour of an edge set. Cross-attention is always the steel of the interface, the other side of the picture.
 * Any other set is white, unless its author named a tint to tell families apart; that tint is kept as a hint,
 * softened toward white so the picture stays quiet. The writer's default tint ("violet") means "none".
 */
export function linkColor(lk: Pick<Link, 'tint' | 'kind'>): RGB {
  if (lk.kind === 'cross') return STEEL;
  if (lk.tint === '' || lk.tint === 'violet') return WHITE;
  return mix(parseTint(lk.tint, WHITE), WHITE, 0.55);
}

export interface Net {
  /** Target token the rollout starts from. */
  token: number;
  /** Attention head used (-1: mean). */
  head: number;
  /** Incoming edges kept per node. */
  k: number;
  /** Stages of the trace plus the output-candidate column. */
  nStages: number;
  /** Index of the output-candidate column. */
  sOut: number;
  /** Index of the last real stage. */
  sPred: number;
  /** Node of sPred whose prediction is explained. */
  selP: number;
  /** Node index of the target token among the candidates, -1 when it is not listed. */
  trueOut: number;
  /** Index of the first candidate node. */
  outFirst: number;
  /** Half the length of the graph along x. */
  halfSpan: number;
  stageX: Float32Array;
  /** Index of the first node of every stage. */
  first: Int32Array;
  /** Nodes per stage. */
  count: Int32Array;
  nodes: { n: number; x: Float32Array; y: Float32Array; z: Float32Array; stage: Int32Array; idx: Int32Array; act: Float32Array; imp: Float32Array };
  edges: { n: number; a: Int32Array; b: Int32Array; w: Float32Array; imp: Float32Array; kind: Uint8Array; stage: Int32Array; tint: Uint16Array };
  /** Colours referenced by `edges.tint`. */
  tints: RGB[];
  /** Runs of stages with one group label. */
  groups: { a: number; b: number; label: string }[];
  /** Edges that carry pulses, strongest first. */
  hot: Int32Array;
}

/** The target token the net explains: the selection when it is a predicted token, else the first predicted one. */
export function selectedToken(unit: UnitTrace, token: number): number {
  if (token >= unit.t0 && token < unit.nTokens) return token;
  return Math.min(unit.t0, Math.max(0, unit.nTokens - 1));
}

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

/**
 * Builds the graph and the rollout for target token `jSel`.
 * @param headSel attention head, negative for the mean over heads
 * @param K strongest incoming attention edges kept per destination node
 */
export function buildNet(dump: TraceDump, unit: UnitTrace, jSel: number, headSel: number, K: number): Net {
  const ns = unit.stages.length;
  const nStages = ns + 1;
  const sOut = ns;
  const sPred = Math.max(0, ns - 1);

  const stageX = new Float32Array(nStages);
  let curX = 0;
  for (let s = 0; s < ns; s++) {
    if (s > 0) curX += 40 + unit.stages[s].gapBefore;
    stageX[s] = curX;
  }
  stageX[sOut] = curX + 56;
  const halfSpan = stageX[nStages - 1] * 0.5;

  const first = new Int32Array(nStages);
  const count = new Int32Array(nStages);
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const stageOf: number[] = [];
  const idxOf: number[] = [];
  const act: number[] = [];

  for (let s = 0; s < ns; s++) {
    const st = unit.stages[s];
    const a = st.act;
    let amax = 1e-6;
    if (a) for (let i = 0; i < a.length; i++) amax = Math.max(amax, a[i]);
    first[s] = xs.length;
    count[s] = st.count;
    for (let i = 0; i < st.count; i++) {
      xs.push(stageX[s] + (hash01(s, i + 9000) - 0.5) * st.xJitter);
      ys.push((0.5 - (i + 0.5) / Math.max(1, st.count)) * 2 * st.spanY);
      zs.push((hash01(s, i) - 0.5) * 2 * st.zJitter);
      stageOf.push(s);
      idxOf.push(i);
      act.push(a && i < a.length ? clamp(a[i] / amax, 0.05, 1) : 0.35);
    }
  }

  // Output-candidate column: the model's top-k for the selected token, plus the target if it is not among them.
  const jj = clamp(jSel, 0, Math.max(0, unit.nTokens - 1));
  const outs: [number, number][] = [];
  if (unit.nTokens > 0 && jSel >= unit.t0) {
    const right = unit.tok[jj];
    let listed = false;
    for (let k = 0; k < dump.topK; k++) {
      const id = unit.topId[jj * dump.topK + k];
      if (id < 0) break;
      outs.push([id, unit.topP[jj * dump.topK + k]]);
      listed ||= id === right;
    }
    if (!listed) outs.push([right, unit.p[jj]]);
  }
  const outFirst = xs.length;
  first[sOut] = outFirst;
  count[sOut] = outs.length;
  const rightId = unit.nTokens > 0 && jSel >= 0 && jSel < unit.nTokens ? unit.tok[jj] : -1;
  let trueOut = -1;
  outs.forEach(([id, pr], i) => {
    xs.push(stageX[sOut]);
    ys.push((0.5 - (i + 0.5) / Math.max(1, outs.length)) * 90);
    zs.push(0);
    stageOf.push(sOut);
    idxOf.push(id);
    act.push(pr);
    if (id === rightId) trueOut = outFirst + i;
  });

  // Group brackets under the graph: consecutive stages with the same group label.
  const groups: Net['groups'] = [];
  for (let s = 0; s < ns; ) {
    const grp = unit.stages[s].group;
    let e = s;
    while (e + 1 < ns && unit.stages[e + 1].group === grp) e++;
    groups.push({ a: s, b: e, label: grp || unit.stages[s].label });
    s = e + 1;
  }
  groups.push({ a: sOut, b: sOut, label: 'Output' });

  // Backward rollout over the stage graph.
  const imp = Array.from({ length: nStages }, (_, s) => new Float32Array(count[s]));
  const selP = ns > 0 ? clamp(jSel - 1, 0, Math.max(0, count[sPred] - 1)) : 0;
  if (ns > 0 && count[sPred] > 0) imp[sPred][selP] = 1;

  const ea: number[] = [];
  const eb: number[] = [];
  const ew: number[] = [];
  const ekind: number[] = [];
  const estage: number[] = [];
  const eflow: number[] = [];
  const etint: number[] = [];
  const tints: RGB[] = [];
  const tintIndex = new Map<string, number>();
  const tintId = (c: RGB): number => {
    const key = c.join(',');
    let i = tintIndex.get(key);
    if (i === undefined) {
      i = tints.length;
      tints.push(c);
      tintIndex.set(key, i);
    }
    return i;
  };
  const push = (a: number, b: number, w: number, kind: number, stage: number, flow: number, tint: RGB): void => {
    ea.push(a);
    eb.push(b);
    ew.push(w);
    ekind.push(kind);
    estage.push(stage);
    eflow.push(flow);
    etint.push(tintId(tint));
  };

  if (ns > 0 && count[sPred] > 0) {
    for (let o = 0; o < count[sOut]; o++) {
      const pr = act[outFirst + o];
      push(first[sPred] + selP, outFirst + o, pr, EDGE.OUT, sOut, pr, ROLLOUT_COLOR);
    }
  }

  for (let dst = ns - 1; dst >= 0; dst--) {
    const resLinks: number[] = [];
    const attnLinks: number[] = [];
    unit.links.forEach((lk, li) => {
      if (lk.dst !== dst) return;
      if (lk.kind === 'resid' || lk.weights === null) resLinks.push(li);
      else attnLinks.push(li);
    });
    const dstCnt = count[dst];
    const resShare = resLinks.length === 0 ? 0 : attnLinks.length === 0 ? 1 : RHO;
    const attnShare = 1 - resShare;

    if (resLinks.length) {
      const perRes = resShare / resLinks.length;
      for (const li of resLinks) {
        const src = unit.links[li].src;
        const m = Math.min(dstCnt, count[src]);
        for (let p = 0; p < m; p++) {
          const f = perRes * imp[dst][p];
          imp[src][p] += f;
          push(first[src] + p, first[dst] + p, 1, EDGE.RESID, dst, f, WHITE);
        }
      }
    }
    if (attnLinks.length) {
      const perAttn = attnShare / attnLinks.length;
      for (const li of attnLinks) {
        const lk = unit.links[li];
        const src = lk.src;
        const srcCnt = count[src];
        const shape = lk.weights!.shape;
        const R = shape[shape.length - 2];
        const C = shape[shape.length - 1];
        const mat = headAverage(lk, headSel);
        const col = linkColor(lk);
        const kind = lk.kind === 'cross' ? EDGE.CROSS : EDGE.SELF;
        for (let p = 0; p < dstCnt; p++) {
          const r = p - lk.rowOffset;
          if (r < 0 || r >= R) continue;
          const cImp = imp[dst][p];
          const row = r * C;
          const lim = Math.min(C, srcCnt, lk.causal ? p + 1 : Infinity);
          if (cImp > 0) for (let q = 0; q < lim; q++) imp[src][q] += perAttn * cImp * mat[row + q];
          for (const q of topK(mat, row, lim, K, EDGE_FLOOR)) {
            push(first[src] + q, first[dst] + p, mat[row + q], kind, dst, perAttn * cImp * mat[row + q], col);
          }
        }
      }
    }
  }

  const nNodes = xs.length;
  const nodeImp = new Float32Array(nNodes);
  for (let s = 0; s < ns; s++) {
    let mx = 1e-12;
    for (const x of imp[s]) mx = Math.max(mx, x);
    for (let i = 0; i < count[s]; i++) nodeImp[first[s] + i] = imp[s][i] / mx;
  }
  for (let o = 0; o < count[sOut]; o++) nodeImp[outFirst + o] = act[outFirst + o];

  const nEdges = ea.length;
  const stageMax = new Float32Array(nStages).fill(1e-12);
  for (let e = 0; e < nEdges; e++) {
    if (ekind[e] !== EDGE.RESID) stageMax[estage[e]] = Math.max(stageMax[estage[e]], eflow[e]);
  }
  const edgeImp = new Float32Array(nEdges);
  for (let e = 0; e < nEdges; e++) edgeImp[e] = Math.min(1, eflow[e] / stageMax[estage[e]]);

  const order = Array.from({ length: nEdges }, (_, i) => i);
  order.sort((p, q) => edgeImp[q] - edgeImp[p] || p - q);
  const hot = order.slice(0, Math.min(HOT_LIMIT, nEdges)).filter((e) => edgeImp[e] >= HOT_MIN_IMP);

  return {
    token: jSel,
    head: headSel,
    k: K,
    nStages,
    sOut,
    sPred,
    selP,
    trueOut,
    outFirst,
    halfSpan,
    stageX,
    first,
    count,
    nodes: {
      n: nNodes,
      x: Float32Array.from(xs),
      y: Float32Array.from(ys),
      z: Float32Array.from(zs),
      stage: Int32Array.from(stageOf),
      idx: Int32Array.from(idxOf),
      act: Float32Array.from(act),
      imp: nodeImp,
    },
    edges: {
      n: nEdges,
      a: Int32Array.from(ea),
      b: Int32Array.from(eb),
      w: Float32Array.from(ew),
      imp: edgeImp,
      kind: Uint8Array.from(ekind),
      stage: Int32Array.from(estage),
      tint: Uint16Array.from(etint),
    },
    tints,
    groups,
    hot: Int32Array.from(hot),
  };
}

export interface Projection {
  x: Float32Array;
  y: Float32Array;
  scale: Float32Array;
}

/** Projects all nodes of a graph with `camera`: screen position and size factor per node. */
export function projectNodes(nodes: { n: number; x: Float32Array; y: Float32Array; z: Float32Array }, camera: Camera3D): Projection {
  const x = new Float32Array(nodes.n);
  const y = new Float32Array(nodes.n);
  const scale = new Float32Array(nodes.n);
  const tmp = new Float32Array(3);
  for (let i = 0; i < nodes.n; i++) {
    camera.map(nodes.x[i], nodes.y[i], nodes.z[i], tmp);
    x[i] = tmp[0];
    y[i] = tmp[1];
    scale[i] = tmp[2];
  }
  return { x, y, scale };
}

/** Index of the node under screen point (x, y), or -1; nearer nodes and bigger ones win. */
export function pickNode(screen: Projection, x: number, y: number): number {
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < screen.x.length; i++) {
    const dx = screen.x[i] - x;
    const dy = screen.y[i] - y;
    const d2 = dx * dx + dy * dy;
    const reach = Math.max(5, 3.5 * screen.scale[i]);
    if (d2 < reach * reach && d2 < bestD) {
      bestD = d2;
      best = i;
    }
  }
  return best;
}

/** One-line description of node `i` for the tooltip and the inspector. */
export function describeNode(dump: TraceDump, unit: UnitTrace, net: Net, i: number): string {
  const nd = net.nodes;
  if (nd.stage[i] === net.sOut) {
    return `output "${dump.label(nd.idx[i])}" (p=${nd.act[i].toFixed(3)})${i === net.trueOut ? ' [target]' : ''}`;
  }
  const st = unit.stages[nd.stage[i]];
  let s = `${st.label} [${nd.idx[i]}]`;
  if (st.tokenSource === 'target') s += ` "${tokenText(dump, unit, nd.idx[i] + st.tokenOffset)}"`;
  else if (st.tokenSource === 'source') s += ` "${sourceText(dump, unit, nd.idx[i] + st.tokenOffset)}"`;
  return `${s} | act ${nd.act[i].toFixed(2)}, rollout ${nd.imp[i].toFixed(2)}`;
}
