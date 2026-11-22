/**
 * Model of a trace dump (.vtdump, header kind "dump"): one or more forward passes ("units") with per-token
 * statistics, attention matrices, activation norms and optional latent maps. Browser twin of
 * cpp/src/vitro_file.hpp: same fields, same defaults, same refusals.
 *
 * Conventions: per-token arrays have one entry per target token; top-k arrays are flat (T * topK) in C order.
 * Token ids are validated against the vocabulary here, once, so renderers can call `dump.label(id)` freely.
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

/** A directed edge set between two stages: a residual connection or an attention matrix. */
export interface Link {
  src: number;
  dst: number;
  layerIdx: number;
  /** First target token that the matrix rows describe. */
  rowOffset: number;
  name: string;
  /** Attention family it belongs to ('' for residuals). */
  family: string;
  /** resid | self | cross | out */
  kind: string;
  tint: string;
  causal: boolean;
  /** (heads, rows, cols) or (rows, cols); null for residual links. */
  weights: NdArray | null;
}

/** All layers of one kind of attention, e.g. "Decoder Self-Attn". */
export interface AttnFamily {
  name: string;
  rowLabel: string;
  colLabel: string;
  /** One link per layer, in layer order. */
  linkIndices: number[];
  layers: number;
  heads: number;
  rows: number;
  cols: number;
  rowOffset: number;
  causal: boolean;
}

export interface NormBlock {
  name: string;
  layers: string[];
  components: string[];
  /** (layers, components, rows) L2 norms. */
  data: NdArray | null;
  rowOffset: number;
}

export interface LatentMap {
  name: string;
  rowLabel: string;
  colLabel: string;
  /** (slots, dims) */
  data: NdArray;
}

/** One forward pass through the model, with everything recorded about it. */
export interface UnitTrace {
  u: number;
  label: string;
  kind: string;
  /** First predicted token; the tokens before it are context. */
  t0: number;
  bits: number;
  forcedExact: boolean;
  freeExact: boolean;
  freeEnded: boolean;
  /** First token where the free rollout left the target, -1 for none. */
  diverge: number;
  anomalies: number;
  /** Number of target tokens. */
  nTokens: number;
  /** Number of source (encoder) tokens; equals nTokens for decoder-only traces. */
  nSource: number;
  /** Length of the free-generation rollout; 0 when none was recorded. */
  freeN: number;

  tok: Int32Array;
  srcTok: Int32Array | null;
  tokenTags: Int32Array | null;
  p: Float32Array;
  rank: Int32Array;
  /** Flat (nTokens * topK): candidate ids, best first, -1 for padding. */
  topId: Int32Array;
  topP: Float32Array;
  entropy: Float32Array;
  nAllowed: Int32Array | null;
  right: Uint8Array;
  valMean: Float32Array | null;
  valStd: Float32Array | null;
  ftok: Int32Array | null;
  fp: Float32Array | null;

  stages: Stage[];
  links: Link[];
  attnFamilies: AttnFamily[];
  norms: NormBlock;
  latents: LatentMap[];
}

export class TraceDump {
  readonly kind = 'dump';
  readonly name: string;
  readonly info: TraceInfo;
  readonly vocab: Vocab;
  readonly spans: SpanInfo[];
  readonly units: UnitTrace[];
  /** Size of the tensor section in bytes. */
  readonly dataBytes: number;

  constructor(f: { name: string; info: TraceInfo; vocab: Vocab; spans: SpanInfo[]; units: UnitTrace[]; dataBytes: number }) {
    this.name = f.name;
    this.info = f.info;
    this.vocab = f.vocab;
    this.spans = f.spans;
    this.units = f.units;
    this.dataBytes = f.dataBytes;
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

  /** Spans that contain target token `j`, in the order of the header. */
  spansCovering(j: number): SpanInfo[] {
    return this.spans.filter((s) => j >= s.start && j < s.end);
  }

  /** Nesting depth of span `idx` (0 for a top-level span); cycles are cut after 32 levels. */
  spanDepth(idx: number): number {
    let depth = 0;
    let cur = idx;
    for (let guard = 0; guard < 32 && cur >= 0 && cur < this.spans.length; guard++) {
      const p = this.spans[cur].parent;
      if (p < 0 || p === cur) break;
      depth++;
      cur = p;
    }
    return depth;
  }
}

/** Groups the links of one unit into attention families, checking that layers agree on shape. */
function buildAttnFamilies(r: Reader, u: number, stages: Stage[], links: Link[]): AttnFamily[] {
  const families: AttnFamily[] = [];
  links.forEach((lk, i) => {
    if (lk.weights === null || lk.family === '') return;
    const shape = lk.weights.shape;
    if (shape.length !== 2 && shape.length !== 3) r.fail(`unit ${u}: attention weights must be 2D or 3D`);
    const heads = shape.length === 3 ? shape[0] : 1;
    const rows = shape[shape.length - 2];
    const cols = shape[shape.length - 1];
    let fam = families.find((f) => f.name === lk.family);
    if (!fam) {
      fam = {
        name: lk.family,
        rowLabel: stages[lk.dst].label || 'Query position',
        colLabel: stages[lk.src].label || 'Key position',
        linkIndices: [],
        layers: 0,
        heads,
        rows,
        cols,
        rowOffset: lk.rowOffset,
        causal: lk.causal,
      };
      families.push(fam);
    } else if (heads !== fam.heads || rows !== fam.rows || cols !== fam.cols) {
      r.fail(`unit ${u}: attention family '${fam.name}' has mismatched layer shapes`);
    }
    fam.linkIndices.push(i);
    fam.layers = fam.linkIndices.length;
  });
  return families;
}

function parseUnit(r: Reader, js: unknown, index: number, vocab: Vocab, topK: number): UnitTrace {
  const p = `units[${index}].`;
  if (!isObject(js)) r.fail(`"units[${index}]" must be an object`);
  const u = r.int(js, 'u', p, index);
  const where = `unit ${u}`;
  const seq = r.object(js, 'sequence', p);
  const sp = `${p}sequence.`;

  const tok = r.ints(seq, 'tokens', sp);
  const T = tok.length;
  const srcTok = r.optionalInts(seq, 'source_tokens', sp);
  const unit: UnitTrace = {
    u,
    label: r.str(js, 'label', p, `Unit ${u}`),
    kind: r.str(js, 'kind', p, 'sequence'),
    t0: r.int(js, 't0', p, 1),
    bits: r.num(js, 'bits', p, 0),
    forcedExact: r.bool(js, 'forced_exact', p, false),
    freeExact: r.bool(js, 'free_exact', p, false),
    freeEnded: r.bool(js, 'free_ended', p, false),
    diverge: r.int(js, 'diverge', p, -1),
    anomalies: r.int(js, 'anomalies', p, 0),
    nTokens: T,
    nSource: srcTok ? srcTok.length : T,
    freeN: 0,
    tok,
    srcTok,
    tokenTags: r.optionalInts(seq, 'token_tags', sp),
    p: r.floats(seq, 'p', sp),
    rank: r.ints(seq, 'rank', sp),
    topId: r.ints(seq, 'top_id', sp),
    topP: r.floats(seq, 'top_p', sp),
    entropy: r.floats(seq, 'entropy', sp),
    nAllowed: r.optionalInts(seq, 'n_allowed', sp),
    right: r.flags(seq, 'right', sp),
    valMean: r.optionalFloats(seq, 'val_mean'),
    valStd: r.optionalFloats(seq, 'val_std'),
    ftok: r.optionalInts(seq, 'free_tok', sp),
    fp: r.optionalFloats(seq, 'free_p'),
    stages: [],
    links: [],
    attnFamilies: [],
    norms: { name: 'Layer Activations', layers: [], components: [], data: null, rowOffset: 0 },
    latents: [],
  };
  unit.freeN = unit.ftok ? unit.ftok.length : 0;

  if (unit.p.length !== T || unit.rank.length !== T || unit.entropy.length !== T) {
    r.fail(`${where}: per-token metric shapes do not match tokens length`);
  }
  if (unit.t0 < 0 || unit.t0 > T) r.fail(`${where}: t0 ${unit.t0} lies outside the ${T} tokens`);
  if (unit.topId.length !== T * topK || unit.topP.length !== T * topK) {
    r.fail(`${where}: top-k arrays hold ${unit.topId.length} ids for ${T} tokens, expected top_k (${topK}) per token`);
  }
  if ((unit.ftok === null) !== (unit.fp === null) || (unit.ftok !== null && unit.fp !== null && unit.ftok.length !== unit.fp.length)) {
    r.fail(`${where}: free rollout tokens and probabilities do not match`);
  }
  checkTokenIds(r, `${where} tokens`, tok, vocab.size);
  checkTokenIds(r, `${where} top ids`, unit.topId, vocab.size, true);
  if (unit.srcTok) checkTokenIds(r, `${where} source tokens`, unit.srcTok, vocab.size);
  if (unit.ftok) checkTokenIds(r, `${where} free rollout`, unit.ftok, vocab.size);

  unit.stages = parseStages(r, r.list(js, 'stages', p), `${p}stages`);
  r.list(js, 'links', p).forEach((je, e) => {
    const lp = `${p}links[${e}].`;
    if (!isObject(je)) r.fail(`"links[${e}]" must be an object`);
    const src = r.int(je, 'src', lp);
    const dst = r.int(je, 'dst', lp);
    if (src < 0 || src >= unit.stages.length || dst < 0 || dst >= unit.stages.length) {
      r.fail(`${where}: link stage index out of range`);
    }
    unit.links.push({
      src,
      dst,
      name: r.str(je, 'name', lp, ''),
      family: r.str(je, 'family', lp, ''),
      kind: r.str(je, 'kind', lp, 'self'),
      tint: r.str(je, 'tint', lp, 'violet'),
      layerIdx: r.int(je, 'layer_idx', lp, -1),
      rowOffset: r.int(je, 'row_offset', lp, 0),
      causal: r.bool(je, 'causal', lp, false),
      weights: r.optionalArray(je, 'weights'),
    });
  });
  unit.attnFamilies = buildAttnFamilies(r, u, unit.stages, unit.links);

  const nm = js.norms;
  if (nm !== undefined && nm !== null) {
    if (!isObject(nm)) r.fail(`"${p}norms" must be an object`);
    const np = `${p}norms.`;
    const data = r.array(nm, 'data', np);
    if (data.shape.length !== 3) r.fail(`${where}: norms data must be (layers, components, tokens)`);
    unit.norms = {
      name: r.str(nm, 'name', np, 'Layer Activations'),
      layers: r.list(nm, 'layers', np).map(String),
      components: r.list(nm, 'components', np).map(String),
      data,
      rowOffset: r.int(nm, 'row_offset', np, 0),
    };
  }
  r.list(js, 'latents', p, false).forEach((jm, m) => {
    const mp = `${p}latents[${m}].`;
    if (!isObject(jm)) r.fail(`"latents[${m}]" must be an object`);
    const data = r.array(jm, 'data', mp);
    const name = r.str(jm, 'name', mp);
    if (data.shape.length !== 2) r.fail(`${where}: latent map "${name}" must be 2D (slots, dims)`);
    unit.latents.push({ name, rowLabel: r.str(jm, 'row_label', mp, 'Slot'), colLabel: r.str(jm, 'col_label', mp, 'Dim'), data });
  });
  return unit;
}

/** Parses a container whose header kind is "dump". */
export function parseDump(container: Container): TraceDump {
  const r = new Reader(container);
  const head: JsonObject = container.header;
  const kind = r.str(head, 'kind', '', 'dump');
  if (kind !== 'dump') r.fail(`this is a '${kind}' recording, not a trace dump`);
  const info = parseTraceInfo(r, head);
  const vocab = parseVocab(r, head);
  const spans = parseSpans(r, head);
  const rawUnits = r.list(head, 'units', '');
  if (rawUnits.length === 0) r.fail('trace contains zero units');
  const units = rawUnits.map((js, i) => parseUnit(r, js, i, vocab, info.topK));
  return new TraceDump({ name: container.name, info, vocab, spans, units, dataBytes: container.dataSize });
}
