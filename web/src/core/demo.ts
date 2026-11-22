/**
 * Model of a recorded generation (.vtdemo, header kind "demo"): per segment, the tokens that were written one
 * after another plus a "tape" of what the network did at each step (top candidates, strongest attention
 * links, activation norms). Browser twin of cpp/src/demo_file.hpp, together with the playback clock of the old
 * player.
 *
 * Time is counted in ticks: a segment first spends `readTicks` encoding its source context (when it has one),
 * then one tick per generated token. The player advances TICKS_PER_SECOND ticks per second at 1x speed.
 */

import type { Container, JsonObject, NdArray } from './container.ts';
import {
  Reader,
  checkTokenIds,
  isObject,
  parseSpans,
  parseStages,
  parseTraceInfo,
  parseVocab,
  type SpanInfo,
  type Stage,
  type TraceInfo,
  type Vocab,
} from './header.ts';

export const DEMO_READ_TICKS = 8;
export const TICKS_PER_SECOND = 10;

/** Attention links drawn while the source context is read (everything before the first generated token). */
export interface PrefillLink {
  src: number;
  dst: number;
  kind: string;
  tint: string;
  /** (dst nodes, k) source node of each strongest link, negative for none. */
  ids: NdArray;
  /** (dst nodes, k) */
  weights: NdArray;
}

/** What the network did at every generated step; `M` is the number of tokens in the segment. */
export interface Tape {
  /** (M, topK) candidate token ids, best first. */
  topId: NdArray;
  topP: NdArray;
  /** (M) */
  entropy: NdArray;
  /** (M) probability of the reference token, when a reference was given. */
  pRef: NdArray | null;
  /** (M, layers, k) strongest self-attention sources per decoder layer. */
  selfId: NdArray;
  selfW: NdArray;
  crossId: NdArray | null;
  crossW: NdArray | null;
  /** (M, layers, channels) activation norm of the step in each layer. */
  norm: NdArray;
}

export interface Segment {
  k: number;
  label: string;
  kind: string;
  /** First generated token index (the context before it is given). */
  t0: number;
  /** First token where the rollout left the reference, -1 for none. */
  diverge: number;
  freeExact: boolean;
  freeEnded: boolean;
  hasRead: boolean;
  readTicks: number;
  writeTicks: number;
  /** Tick at which the segment begins on the global timeline. */
  start: number;
  srcTok: Int32Array | null;
  refTok: Int32Array | null;
  /** Tokens as written. */
  ftok: Int32Array;
  /** Probability of each written token. */
  fp: Float32Array;
  stages: Stage[];
  prefillLinks: PrefillLink[];
  inStage: number;
  outStage: number;
  /** -1 when the model has no cross-attention. */
  crossStage: number;
  layerStages: number[];
  tape: Tape;
}

/** Where playback is at a given position on the timeline. */
export interface Now {
  /** Segment index. */
  seg: number;
  /** True while the source context is being read. */
  reading: boolean;
  /** Progress of the reading phase, 0..1. */
  readP: number;
  /** Index of the token being written. */
  token: number;
  /** Progress inside the current token, 0..1. */
  frac: number;
}

export class TraceDemo {
  readonly kind = 'demo';
  readonly name: string;
  readonly info: TraceInfo;
  readonly vocab: Vocab;
  readonly spans: SpanInfo[];
  readonly segments: Segment[];
  readonly dataBytes: number;
  /** Length of the whole timeline in ticks. */
  readonly total: number;
  readonly totalTokens: number;

  constructor(f: { name: string; info: TraceInfo; vocab: Vocab; spans: SpanInfo[]; segments: Segment[]; dataBytes: number }) {
    this.name = f.name;
    this.info = f.info;
    this.vocab = f.vocab;
    this.spans = f.spans;
    this.segments = f.segments;
    this.dataBytes = f.dataBytes;
    const last = f.segments[f.segments.length - 1];
    this.total = last.start + last.readTicks + last.writeTicks;
    this.totalTokens = f.segments.reduce((n, s) => n + s.ftok.length, 0);
  }

  get uid(): string {
    return this.info.id;
  }

  get title(): string {
    return this.info.title;
  }

  get topK(): number {
    return this.info.topK;
  }

  label(id: number): string {
    return this.vocab.label(id);
  }

  /** Index of the segment that contains timeline position `pos` (the last one starting at or before it). */
  segmentAt(pos: number): number {
    let lo = 0;
    let hi = this.segments.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.segments[mid].start <= pos) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** What is happening at timeline position `pos`; the position is clamped into the timeline. */
  locate(pos: number): Now {
    const p = Math.min(Math.max(pos, 0), Math.max(0, this.total - 1e-9));
    const seg = this.segmentAt(p);
    const sg = this.segments[seg];
    const local = p - sg.start;
    if (sg.readTicks > 0 && local < sg.readTicks) {
      return { seg, reading: true, readP: local / sg.readTicks, token: sg.t0, frac: 0.5 };
    }
    const w = local - sg.readTicks;
    const maxSteps = Math.max(1, sg.ftok.length - sg.t0);
    const i = Math.min(Math.max(Math.floor(w), 0), maxSteps - 1);
    return { seg, reading: false, readP: 1, token: sg.t0 + i, frac: Math.min(Math.max(w - i, 0), 1) };
  }

  /** Timeline position in the middle of token `token` of segment `seg`; the token is clamped into the segment. */
  positionOfToken(seg: number, token: number): number {
    const sg = this.segments[seg];
    const tok = Math.min(Math.max(token, sg.t0), sg.ftok.length - 1);
    return sg.start + sg.readTicks + (tok - sg.t0) + 0.5;
  }
}

/** Reads and cross-checks one segment record. */
function parseSegment(r: Reader, js: unknown, index: number, vocab: Vocab, topK: number, start: number): Segment {
  const p = `segments[${index}].`;
  if (!isObject(js)) r.fail(`"segments[${index}]" must be an object`);
  const k = r.int(js, 'k', p, index);
  const where = `segment ${k}`;
  const hasRead = r.bool(js, 'has_read', p, false);
  const t0 = Math.max(1, r.int(js, 't0', p, 1));
  const ftok = r.ints(js, 'free_tok', p);
  const fp = r.floats(js, 'free_p', p);
  if (ftok.length <= t0) r.fail(`${where}: no generated tokens past t0=${t0}`);
  if (fp.length !== ftok.length) r.fail(`${where}: free_p has ${fp.length} entries for ${ftok.length} tokens`);
  const srcTok = r.optionalInts(js, 'source_tokens', p);
  const refTok = r.optionalInts(js, 'ref_tokens', p);
  checkTokenIds(r, `${where} free tokens`, ftok, vocab.size);
  if (srcTok) checkTokenIds(r, `${where} source tokens`, srcTok, vocab.size);
  if (refTok) checkTokenIds(r, `${where} reference tokens`, refTok, vocab.size);

  const stages = parseStages(r, r.list(js, 'stages', p), `${p}stages`);
  const stageOk = (s: unknown): boolean => typeof s === 'number' && Number.isInteger(s) && s >= 0 && s < stages.length;

  const prefillLinks = r.list(js, 'prefill_links', p, false).map((je, e): PrefillLink => {
    const lp = `${p}prefill_links[${e}].`;
    if (!isObject(je)) r.fail(`"prefill_links[${e}]" must be an object`);
    const link: PrefillLink = {
      src: r.int(je, 'src', lp),
      dst: r.int(je, 'dst', lp),
      kind: r.str(je, 'kind', lp, 'self'),
      tint: r.str(je, 'tint', lp, 'violet'),
      ids: r.intArray(je, 'ids', lp),
      weights: r.array(je, 'weights', lp),
    };
    if (!stageOk(link.src) || !stageOk(link.dst)) r.fail(`${where}: prefill link stage index out of range`);
    if (link.ids.shape.length !== 2 || link.weights.length !== link.ids.length) {
      r.fail(`${where}: prefill link ids and weights must be (nodes, k) of the same size`);
    }
    return link;
  });

  const dc = r.object(js, 'decode', p);
  const dp = `${p}decode.`;
  const layerStages = r.list(dc, 'layer_stages', dp);
  const inStage = r.int(dc, 'in_stage', dp);
  const outStage = r.int(dc, 'out_stage', dp);
  const crossStage = r.int(dc, 'cross_stage', dp, -1);
  if (!stageOk(inStage) || !stageOk(outStage) || !layerStages.every(stageOk) || (crossStage >= 0 && !stageOk(crossStage))) {
    r.fail(`${where}: decode stage index out of range`);
  }

  const tp = r.object(js, 'tape', p);
  const tpp = `${p}tape.`;
  const tape: Tape = {
    topId: r.intArray(tp, 'top_id', tpp),
    topP: r.array(tp, 'top_p', tpp),
    entropy: r.array(tp, 'entropy', tpp),
    pRef: r.optionalArray(tp, 'p_ref'),
    selfId: r.intArray(tp, 'self_id', tpp),
    selfW: r.array(tp, 'self_w', tpp),
    crossId: r.optionalIntArray(tp, 'cross_id', tpp),
    crossW: r.optionalArray(tp, 'cross_w'),
    norm: r.array(tp, 'norm', tpp),
  };
  const M = ftok.length;
  const layers = layerStages.length;
  if (tape.topId.shape.length !== 2 || tape.topId.shape[0] !== M || tape.topId.shape[1] !== topK || tape.topP.length !== tape.topId.length) {
    r.fail(`${where}: tape top-k must be (${M}, ${topK}); got [${tape.topId.shape}]`);
  }
  if (tape.entropy.length !== M) r.fail(`${where}: tape dimensions do not match generated length`);
  if (tape.selfId.shape.length !== 3 || tape.selfId.shape[0] !== M || tape.selfId.shape[1] !== layers || tape.selfW.length !== tape.selfId.length) {
    r.fail(`${where}: tape self-attention must be (${M}, ${layers}, k); got [${tape.selfId.shape}]`);
  }
  if (tape.norm.shape.length !== 3 || tape.norm.shape[0] !== M || tape.norm.shape[1] !== layers) {
    r.fail(`${where}: tape norm must be (${M}, ${layers}, channels); got [${tape.norm.shape}]`);
  }
  if (crossStage >= 0) {
    if (!tape.crossId || !tape.crossW) r.fail(`${where}: cross-attention stage given but the tape has no cross links`);
    if (tape.crossId.shape.length !== 3 || tape.crossId.shape[0] !== M || tape.crossId.shape[1] !== layers || tape.crossW.length !== tape.crossId.length) {
      r.fail(`${where}: tape cross-attention must be (${M}, ${layers}, k); got [${tape.crossId.shape}]`);
    }
  }
  checkTokenIds(r, `${where} tape top ids`, tape.topId.int32(), vocab.size, true);

  const readTicks = hasRead ? r.num(js, 'read_ticks', p, DEMO_READ_TICKS) : 0;
  return {
    k,
    label: r.str(js, 'label', p, `Segment ${k}`),
    kind: r.str(js, 'kind', p, 'decode'),
    t0,
    diverge: r.int(js, 'diverge', p, -1),
    freeExact: r.bool(js, 'free_exact', p, false),
    freeEnded: r.bool(js, 'free_ended', p, false),
    hasRead,
    readTicks,
    writeTicks: ftok.length - t0,
    start,
    srcTok,
    refTok,
    ftok,
    fp,
    stages,
    prefillLinks,
    inStage,
    outStage,
    crossStage,
    layerStages: layerStages as number[],
    tape,
  };
}

/** Parses a container whose header kind is "demo". */
export function parseDemo(container: Container): TraceDemo {
  const r = new Reader(container);
  const head: JsonObject = container.header;
  const kind = r.str(head, 'kind', '', '');
  if (kind !== 'demo') r.fail(`this is a '${kind}' trace, not a generation recording`);
  const info = parseTraceInfo(r, head);
  const vocab = parseVocab(r, head);
  const spans = parseSpans(r, head);
  const raw = r.list(head, 'segments', '');
  if (raw.length === 0) r.fail('recording contains zero segments');
  const segments: Segment[] = [];
  let t = 0;
  raw.forEach((js, i) => {
    const seg = parseSegment(r, js, i, vocab, info.topK, t);
    t += seg.readTicks + seg.writeTicks;
    segments.push(seg);
  });
  return new TraceDemo({ name: container.name, info, vocab, spans, segments, dataBytes: container.dataSize });
}
