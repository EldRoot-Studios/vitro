/**
 * Quantities derived from a trace for display: token colours, averaged attention maps, latent similarity,
 * token descriptions and jumps between errors. Port of the helpers in cpp/src/view_app.cpp.
 * Results that cost a pass over a matrix are cached per unit.
 */

import { deltaError, hex, heat, surprisal, type RGB } from './colors.ts';
import type { AttnFamily, Link, TraceDump, UnitTrace } from './trace.ts';

/** Surprisal of probability `p` in bits; the floor keeps log2(0) finite. */
export function bitsOf(p: number): number {
  return -Math.log2(Math.max(p, 1e-30));
}

/** Text of target token `j`, or "?" outside the sequence. */
export function tokenText(dump: TraceDump, unit: UnitTrace, j: number): string {
  return j >= 0 && j < unit.nTokens ? dump.label(unit.tok[j]) : '?';
}

/** Text of source token `j`; equals the target text for decoder-only traces. */
export function sourceText(dump: TraceDump, unit: UnitTrace, j: number): string {
  if (!unit.srcTok) return tokenText(dump, unit, j);
  return j >= 0 && j < unit.nSource ? dump.label(unit.srcTok[j]) : '?';
}

/** Text of free-rollout token `j`, or '' past its end. */
export function freeText(dump: TraceDump, unit: UnitTrace, j: number): string {
  return unit.ftok !== null && j >= 0 && j < unit.freeN ? dump.label(unit.ftok[j]) : '';
}

/**
 * Labels for the rows and columns of an attention family's matrix. A row is a position of the destination
 * stage and a column a position of the source stage; each is named by the token the stage shows there (target
 * token, source token), or by its number when the stage carries no tokens.
 */
export function attnAxes(dump: TraceDump, unit: UnitTrace, af: AttnFamily): { row: (r: number) => string; col: (c: number) => string } {
  const link = unit.links[af.linkIndices[0]];
  const dst = unit.stages[link.dst];
  const src = unit.stages[link.src];
  const name = (stage: { tokenSource: string; tokenOffset: number }, idx: number): string => {
    if (stage.tokenSource === 'source') return sourceText(dump, unit, idx + stage.tokenOffset);
    if (stage.tokenSource === 'target') return tokenText(dump, unit, idx + stage.tokenOffset);
    return `#${idx}`;
  };
  return { row: (r) => name(dst, af.rowOffset + r), col: (c) => name(src, c) };
}

/** How tokens are coloured in the Tokens view. */
export const COLOR_MODES = ['delta', 'surprisal', 'rank', 'entropy'] as const;
export type ColorMode = (typeof COLOR_MODES)[number];

const RANK_CLASS_T = [0.0, 0.34, 0.62, 0.9];

/** True when target token `j` carries a numeric value in the vocabulary and the trace predicted a mean for it. */
export function isScalarToken(dump: TraceDump, unit: UnitTrace, j: number): boolean {
  if (j < unit.t0 || j >= unit.nTokens || unit.valMean === null) return false;
  return !Number.isNaN(unit.valMean[j]) && dump.vocab.hasValue(unit.tok[j]);
}

/** Id of the model's top-1 candidate at token `j`, or -1. */
export function topId(dump: TraceDump, unit: UnitTrace, j: number): number {
  return unit.topId[j * dump.topK];
}

/** Cell colour of target token `j` and whether the model's top-1 prediction missed it. */
export function cellColor(dump: TraceDump, unit: UnitTrace, j: number, mode: ColorMode): { color: RGB; wrong: boolean } {
  if (j < unit.t0) return { color: hex('#1d1d1b'), wrong: false };
  const rank = unit.rank[j];
  const wrong = rank > 1;
  if (mode === 'surprisal') return { color: surprisal(bitsOf(unit.p[j])), wrong };
  if (mode === 'entropy') return { color: surprisal(unit.entropy[j]), wrong };
  if (mode === 'rank') {
    const cls = rank <= 1 ? 0 : rank === 2 ? 1 : rank <= 5 ? 2 : 3;
    return { color: heat(RANK_CLASS_T[cls]), wrong };
  }
  if (isScalarToken(dump, unit, j) && unit.valMean !== null) {
    const top = topId(dump, unit, j);
    const vTrue = dump.vocab.valueOf(unit.tok[j]);
    const vTop = dump.vocab.hasValue(top) ? dump.vocab.valueOf(top) : unit.valMean[j];
    return { color: deltaError(Math.abs(vTop - vTrue), dump.info.deltaRedAt), wrong };
  }
  return { color: surprisal(bitsOf(unit.p[j])), wrong };
}

/** Indices of target tokens the model's top-1 prediction missed (rank above 1). */
export function errorTokens(unit: UnitTrace): number[] {
  const out: number[] = [];
  for (let j = unit.t0; j < unit.nTokens; j++) if (unit.rank[j] > 1) out.push(j);
  return out;
}

/** First error strictly after (`dir` 1) or before (`dir` -1) token `from`, or -1. */
export function nextError(unit: UnitTrace, from: number, dir: 1 | -1): number {
  const start = from < 0 ? (dir > 0 ? unit.t0 - 1 : unit.nTokens) : from;
  for (let j = start + dir; j >= unit.t0 && j < unit.nTokens; j += dir) if (unit.rank[j] > 1) return j;
  return -1;
}

/** One-line description of target token `j` for tooltips. */
export function describeToken(dump: TraceDump, unit: UnitTrace, j: number): string {
  if (j < 0 || j >= unit.nTokens) return '';
  let s = `token ${j} "${dump.label(unit.tok[j])}"`;
  if (unit.tokenTags) {
    const tag = dump.vocab.tagName(unit.tokenTags[j]);
    if (tag) s += ` [${tag}]`;
  }
  const cov = dump.spansCovering(j);
  if (cov.length) s += ` (${cov[cov.length - 1].label})`;
  if (j < unit.t0) return `${s} - context prefix`;
  s += ` - p=${unit.p[j].toFixed(3)}, rank=${unit.rank[j]}`;
  const top = topId(dump, unit, j);
  if (top >= 0 && unit.rank[j] > 1) s += ` (top-1: "${dump.label(top)}" ${unit.topP[j * dump.topK].toFixed(2)})`;
  if (isScalarToken(dump, unit, j) && unit.valMean !== null) {
    const vt = dump.vocab.valueOf(unit.tok[j]);
    const vp = dump.vocab.hasValue(top) ? dump.vocab.valueOf(top) : unit.valMean[j];
    s += `, delta ${Math.abs(vp - vt).toPrecision(2)} (target ${vt}, top ${vp})`;
  }
  return s;
}

/** Errors and total bits of the tokens in [start, end) of a unit, for the Spans view. */
export function spanStats(unit: UnitTrace, start: number, end: number): { lo: number; hi: number; errors: number; bits: number } {
  const lo = Math.min(Math.max(start, 0), unit.nTokens);
  const hi = Math.min(Math.max(end, lo), unit.nTokens);
  let errors = 0;
  let bits = 0;
  for (let j = Math.max(lo, unit.t0); j < hi; j++) {
    bits += bitsOf(unit.p[j]);
    if (unit.rank[j] > 1) errors++;
  }
  return { lo, hi, errors, bits };
}

const caches = new WeakMap<object, Map<string, Float32Array>>();

function cached(owner: object, key: string, make: () => Float32Array): Float32Array {
  let m = caches.get(owner);
  if (!m) caches.set(owner, (m = new Map()));
  let v = m.get(key);
  if (!v) m.set(key, (v = make()));
  return v;
}

/** Attention weights of one link averaged over heads: all heads when `head` is negative. Rows * cols values. */
export function headAverage(link: Link, head: number): Float32Array {
  const w = link.weights;
  if (w === null) throw new RangeError(`link "${link.name}" has no attention weights`);
  const hasHeads = w.shape.length === 3;
  const H = hasHeads ? w.shape[0] : 1;
  const rc = w.shape[w.shape.length - 2] * w.shape[w.shape.length - 1];
  const one = head >= 0 && head < H;
  return cached(link, `h${one ? head : -1}`, () => {
    const out = new Float32Array(rc);
    const h0 = one ? head : 0;
    const h1 = one ? head + 1 : H;
    const inv = 1 / Math.max(1, h1 - h0);
    const data = w.float32();
    for (let h = h0; h < h1; h++) {
      const base = h * rc;
      for (let i = 0; i < rc; i++) out[i] += data[base + i] * inv;
    }
    return out;
  });
}

/** Attention matrix of a family averaged over layers and heads (a negative layer or head means "all"). */
export function attnMap(unit: UnitTrace, famIdx: number, layer: number, head: number): Float32Array {
  const af = unit.attnFamilies[famIdx];
  return cached(unit, `a${famIdx}:${layer}:${head}`, () => {
    const n = af.rows * af.cols;
    const m = new Float32Array(n);
    const l0 = layer < 0 ? 0 : layer;
    const l1 = layer < 0 ? af.layers : layer + 1;
    const h0 = head < 0 ? 0 : head;
    const h1 = head < 0 ? af.heads : head + 1;
    const inv = 1 / ((l1 - l0) * (h1 - h0));
    for (let l = l0; l < l1; l++) {
      const link = unit.links[af.linkIndices[l]];
      const w = link.weights!.float32();
      for (let h = h0; h < h1; h++) {
        const base = h * n;
        for (let i = 0; i < n; i++) m[i] += w[base + i] * inv;
      }
    }
    return m;
  });
}

/** Largest value in `m`, never below 1e-9 so it can be used as a divisor. */
export function maxOf(m: ArrayLike<number>): number {
  let v = 1e-9;
  for (let i = 0; i < m.length; i++) if (m[i] > v) v = m[i];
  return v;
}

/** Largest absolute value in `m`, never below `floor`. */
export function maxAbs(m: ArrayLike<number>, floor = 1e-6): number {
  let v = floor;
  for (let i = 0; i < m.length; i++) {
    const a = Math.abs(m[i]);
    if (a > v) v = a;
  }
  return v;
}

/** Cosine similarity between the rows of a latent map. Slots * slots values. */
export function latentCos(unit: UnitTrace, latIdx: number): Float32Array {
  const lm = unit.latents[latIdx];
  const S = lm.data.shape[0];
  const D = lm.data.shape[1];
  return cached(lm, 'cos', () => {
    const data = lm.data.float32();
    const norm = new Float32Array(S * D);
    for (let s = 0; s < S; s++) {
      let ss = 0;
      for (let c = 0; c < D; c++) ss += data[s * D + c] ** 2;
      const inv = ss > 1e-18 ? 1 / Math.sqrt(ss) : 0;
      for (let c = 0; c < D; c++) norm[s * D + c] = data[s * D + c] * inv;
    }
    const cos = new Float32Array(S * S);
    for (let i = 0; i < S; i++) {
      for (let j = i; j < S; j++) {
        let s = 0;
        for (let c = 0; c < D; c++) s += norm[i * D + c] * norm[j * D + c];
        cos[i * S + j] = cos[j * S + i] = s;
      }
    }
    return cos;
  });
}
