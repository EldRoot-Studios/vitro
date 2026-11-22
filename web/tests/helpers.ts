/**
 * Test support: an independent JS writer for the VITRO01 container, small valid dumps and recordings built from
 * it (so a refusal test can break exactly one field), and loaders for the sample traces.
 *
 * The writer follows the Python writer's layout on purpose. If the browser reader and this writer agree and the
 * sample files (written by Python) also parse, both ends of the format are covered.
 */

/* Fixtures here are JSON trees that tests damage one field at a time; typing every field would only hide that,
   so they are deliberately loose. */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { openTrace } from '../src/core/open.ts';
import type { TraceDemo } from '../src/core/demo.ts';
import type { TraceDump } from '../src/core/trace.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
export const SAMPLES = join(HERE, '..', '..', 'samples');
const ALIGN = 16;
const padTo = (n: number): number => (ALIGN - (n % ALIGN)) % ALIGN;

export type Doc = any;

export interface ArrayDescriptor {
  dtype: string;
  off: number;
  shape: number[];
}

type Typed = Float32Array | Int32Array | Int16Array | Int8Array | Uint8Array | Uint16Array;

/** Packs typed arrays at 16-byte offsets and hands back descriptors, like python/vitro/container.py Writer. */
export class TestWriter {
  parts: Uint8Array[] = [];
  size = 0;

  /** `typed` is already in the byte layout of `dtype`. */
  add(dtype: string, typed: Typed, shape: number[]): ArrayDescriptor {
    const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    const off = this.size;
    const pad = padTo(bytes.length);
    this.parts.push(bytes, new Uint8Array(pad));
    this.size += bytes.length + pad;
    return { dtype, off, shape };
  }

  f4(values: ArrayLike<number>, shape: number[] = [values.length]): ArrayDescriptor {
    return this.add('f4', Float32Array.from(values), shape);
  }

  i4(values: ArrayLike<number>, shape: number[] = [values.length]): ArrayDescriptor {
    return this.add('i4', Int32Array.from(values), shape);
  }

  /** float16 from raw 16-bit patterns */
  f2bits(patterns: ArrayLike<number>, shape: number[] = [patterns.length]): ArrayDescriptor {
    return this.add('f2', Uint16Array.from(patterns), shape);
  }

  u1(values: ArrayLike<number>, shape: number[] = [values.length]): ArrayDescriptor {
    return this.add('u1', Uint8Array.from(values), shape);
  }
}

/** Assembles a file. `header` may be an object (version 1 is added) or raw text to test bad headers. */
export function buildFile(header: object | string, writer: TestWriter = new TestWriter(), opts: { magic?: string; headerLen?: number | null } = {}): ArrayBuffer {
  const { magic = 'VITRO01\n', headerLen = null } = opts;
  const text = typeof header === 'string' ? header : JSON.stringify({ version: 1, ...header });
  const head = new TextEncoder().encode(text);
  const blob = new Uint8Array(writer.size);
  let at = 0;
  for (const p of writer.parts) {
    blob.set(p, at);
    at += p.length;
  }
  const out = new Uint8Array(16 + head.length + padTo(head.length) + blob.length);
  out.set(new TextEncoder().encode(magic), 0);
  new DataView(out.buffer).setBigUint64(8, BigInt(headerLen ?? head.length), true);
  out.set(head, 16);
  out.set(blob, 16 + head.length + padTo(head.length));
  return out.buffer;
}

/** The bytes of a file as an ArrayBuffer that starts at byte 0 of the file. */
export function readArrayBuffer(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

/** Reads a sample dump through the app's own entry point. */
export function loadDump(name: string): TraceDump {
  const t = openTrace(readArrayBuffer(join(SAMPLES, name)), name);
  if (t.kind !== 'dump') throw new Error(`${name} is a recording, not a dump`);
  return t;
}

/** Reads a sample recording through the app's own entry point. */
export function loadDemo(name: string): TraceDemo {
  const t = openTrace(readArrayBuffer(join(SAMPLES, name)), name);
  if (t.kind !== 'demo') throw new Error(`${name} is a dump, not a recording`);
  return t;
}

const VOCAB = ['<s>', 'the', 'cat', 'sat', 'on', 'mat', '.', '</s>'];
const T = 4;
const TOP_K = 2;

export interface DumpParts {
  header: Doc;
  w: TestWriter;
  unit: Doc;
}

export interface DemoParts {
  header: Doc;
  w: TestWriter;
  seg: Doc;
}

/**
 * A valid two-stage dump of 4 tokens. `mutate({header, w, unit})` may change the header record before it is
 * written, to produce one specific defect.
 */
export function miniDump(mutate: (parts: DumpParts) => void = () => {}): ArrayBuffer {
  const w = new TestWriter();
  const f16One = 0x3c00;
  const f16Half = 0x3800;
  // head 0 is all 0.5, head 1 is all 1.0: their mean is 0.75, and each head can be told apart
  const weights = Array.from({ length: 2 * T * T }, (_, i) => (i < T * T ? f16Half : f16One));
  const unit: Doc = {
    u: 0,
    label: 'Mini',
    kind: 'decoder_only',
    t0: 1,
    bits: 3.5,
    forced_exact: true,
    free_exact: false,
    free_ended: true,
    diverge: 2,
    anomalies: 0,
    sequence: {
      tokens: w.i4([0, 1, 2, 3]),
      p: w.f4([Number.NaN, 0.5, 0.25, 0.75]),
      rank: w.i4([0, 1, 2, 1]),
      top_id: w.i4([-1, -1, 1, 4, 2, 6, 3, 5], [T, TOP_K]),
      top_p: w.f4([0, 0, 0.5, 0.2, 0.4, 0.3, 0.75, 0.1], [T, TOP_K]),
      entropy: w.f4([0, 1, 2, 3]),
      n_allowed: w.i4([8, 8, 8, 8]),
      right: w.u1([1, 1, 0, 1]),
      free_tok: w.i4([0, 1, 4, 3]),
      free_p: w.f4([1, 0.5, 0.4, 0.75]),
    },
    stages: [
      { name: 'in', label: 'Input', group: 'Tokens', kind: 'input', count: T, token_source: 'target', act: w.f4([1, 2, 3, 4]) },
      { name: 'b1', label: 'Block 1', group: 'Layers', kind: 'layer', count: T, token_source: 'target', act: null },
    ],
    links: [
      { src: 0, dst: 1, name: 'Resid', family: '', kind: 'resid', tint: 'violet', layer_idx: 0, row_offset: 0, causal: false, weights: null },
      {
        src: 0,
        dst: 1,
        name: 'Self L1',
        family: 'Self-Attn',
        kind: 'self',
        tint: 'blue',
        layer_idx: 0,
        row_offset: 0,
        causal: true,
        weights: w.f2bits(weights, [2, T, T]),
      },
    ],
    norms: { name: 'Norms', layers: ['L1'], components: ['attn'], data: w.f4([1, 2, 3, 4], [1, 1, T]), row_offset: 0 },
    latents: [{ name: 'State', row_label: 'Slot', col_label: 'Dim', data: w.f4([1, 0, 0, 1, 1, 1], [3, 2]) }],
  };
  const header: Doc = {
    kind: 'dump',
    tool: 'vitro',
    created: '2026-01-01 00:00:00',
    trace: { id: 'mini', title: 'Mini trace', model_name: 'mini-model', arch: 'decoder_only', precision: 'fp32', device: 'cpu', step: null, top_k: TOP_K, delta_red_at: 12 },
    vocab: { labels: VOCAB },
    spans: [{ id: 0, start: 0, end: 2, label: 'Prompt', kind: 'region', parent: -1, tint: 'cyan', meta: '' }],
    units: [unit],
  };
  mutate({ header, w, unit });
  return buildFile(header, w);
}

/** A valid one-segment recording with a 3-stage network (input, one layer, candidates). */
export function miniDemo(mutate: (parts: DemoParts) => void = () => {}): ArrayBuffer {
  const w = new TestWriter();
  const M = 4;
  const seg: Doc = {
    k: 0,
    label: 'Mini rollout',
    kind: 'causal',
    t0: 1,
    diverge: -1,
    free_exact: false,
    free_ended: true,
    has_read: true,
    read_ticks: 4,
    source_tokens: null,
    ref_tokens: null,
    free_tok: w.i4([0, 1, 2, 3]),
    free_p: w.f4([1, 0.5, 0.25, 0.75]),
    stages: [
      { name: 'in', label: 'Tokens', group: 'Input', kind: 'input', count: M, token_source: 'target', act: null },
      { name: 'b1', label: 'Block 1', group: 'Layers', kind: 'layer', count: M, token_source: 'target', act: null },
      { name: 'out', label: 'Candidates', group: 'Output', kind: 'output', count: TOP_K, token_source: 'none', act: null },
    ],
    prefill_links: [],
    decode: { in_stage: 0, out_stage: 2, cross_stage: -1, layer_stages: [1] },
    tape: {
      top_id: w.i4([-1, -1, 1, 4, 2, 6, 3, 5], [M, TOP_K]),
      top_p: w.f4([0, 0, 0.5, 0.2, 0.4, 0.3, 0.75, 0.1], [M, TOP_K]),
      entropy: w.f4([0, 1, 2, 3]),
      p_ref: null,
      self_id: w.i4([0, 0, 0, 1, 0, 1, 1, 2], [M, 1, 2]),
      self_w: w.f2bits([0x3c00, 0, 0x3800, 0x3800, 0x3800, 0x3800, 0x3800, 0x3800], [M, 1, 2]),
      cross_id: null,
      cross_w: null,
      norm: w.f2bits([0x3c00, 0x4000, 0x4200, 0x4400], [M, 1, 1]),
    },
  };
  const header: Doc = {
    kind: 'demo',
    tool: 'vitro',
    created: '2026-01-01 00:00:00',
    trace: { id: 'mini', title: 'Mini rollout', model_name: 'mini-model', arch: 'decoder_only', precision: 'fp32', step: null, top_k: TOP_K },
    vocab: { labels: VOCAB },
    spans: [],
    segments: [seg],
  };
  mutate({ header, w, seg });
  return buildFile(header, w);
}
