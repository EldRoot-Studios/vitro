/**
 * Pieces of the trace header shared by dumps (.vtdump) and recordings (.vtdemo): typed field access with
 * error messages that name the file and the path of the bad field, the vocabulary, user spans and 3D stage
 * definitions.
 *
 * Field access is strict where the Python writer always writes a field and lenient only where the C++
 * viewer was: a missing optional field takes its documented default, a field of the wrong type is a refusal,
 * never a guess.
 */

import { ContainerError, type Container, type JsonObject, type NdArray } from './container.ts';

export function isObject(v: unknown): v is JsonObject {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** Typed access to header JSON for one container; every failure becomes a ContainerError naming the file. */
export class Reader {
  readonly container: Container;
  readonly name: string;

  constructor(container: Container) {
    this.container = container;
    this.name = container.name;
  }

  fail(message: string): never {
    throw new ContainerError(`${this.name}: ${message}`);
  }

  /** The object at `o[key]`, which must exist. */
  object(o: JsonObject | undefined, key: string, path: string): JsonObject {
    const v = o?.[key];
    if (v === undefined || v === null) this.fail(`missing "${path}${key}"`);
    if (!isObject(v)) this.fail(`"${path}${key}" must be an object`);
    return v;
  }

  /** The list at `o[key]`; with `required` false a missing or null field gives []. */
  list(o: JsonObject | undefined, key: string, path: string, required = true): unknown[] {
    const v = o?.[key];
    if (v === undefined || v === null) {
      if (required) this.fail(`missing "${path}${key}"`);
      return [];
    }
    if (!Array.isArray(v)) this.fail(`"${path}${key}" must be a list`);
    return v;
  }

  str(o: JsonObject | undefined, key: string, path: string, fallback?: string): string {
    const v = o?.[key];
    if (v === undefined || v === null) {
      if (fallback === undefined) this.fail(`missing "${path}${key}"`);
      return fallback;
    }
    if (typeof v !== 'string') this.fail(`"${path}${key}" must be a string, got ${JSON.stringify(v)}`);
    return v;
  }

  int(o: JsonObject | undefined, key: string, path: string, fallback?: number): number {
    const v = o?.[key];
    if (v === undefined || v === null) {
      if (fallback === undefined) this.fail(`missing "${path}${key}"`);
      return fallback;
    }
    if (typeof v !== 'number' || !Number.isInteger(v)) this.fail(`"${path}${key}" must be an integer, got ${JSON.stringify(v)}`);
    return v;
  }

  num(o: JsonObject | undefined, key: string, path: string, fallback?: number): number {
    const v = o?.[key];
    if (v === undefined || v === null) {
      if (fallback === undefined) this.fail(`missing "${path}${key}"`);
      return fallback;
    }
    if (typeof v !== 'number' || !Number.isFinite(v)) this.fail(`"${path}${key}" must be a number, got ${JSON.stringify(v)}`);
    return v;
  }

  bool(o: JsonObject | undefined, key: string, path: string, fallback?: boolean): boolean {
    const v = o?.[key];
    if (v === undefined || v === null) {
      if (fallback === undefined) this.fail(`missing "${path}${key}"`);
      return fallback;
    }
    if (typeof v !== 'boolean') this.fail(`"${path}${key}" must be true or false, got ${JSON.stringify(v)}`);
    return v;
  }

  /** The array behind descriptor `o[key]`, which must exist. */
  array(o: JsonObject | undefined, key: string, path: string): NdArray {
    const v = o?.[key];
    if (v === undefined || v === null) this.fail(`missing "${path}${key}"`);
    return this.container.array(v);
  }

  /** The array behind descriptor `o[key]`, or null when the field is absent or null. */
  optionalArray(o: JsonObject | undefined, key: string): NdArray | null {
    const v = o?.[key];
    return v === undefined || v === null ? null : this.container.array(v);
  }

  /** An array of ids that must exist, kept as an NdArray so callers can check its shape. A float array is a refusal. */
  intArray(o: JsonObject | undefined, key: string, path: string): NdArray {
    const a = this.array(o, key, path);
    if (a.isFloat) this.fail(`"${path}${key}" must hold integers, not ${a.dtype}`);
    return a;
  }

  /** Like `intArray`, but a missing or null field gives null. */
  optionalIntArray(o: JsonObject | undefined, key: string, path: string): NdArray | null {
    return o?.[key] === undefined || o[key] === null ? null : this.intArray(o, key, path);
  }

  /** An integer array that must exist. A float array in its place is a refusal. */
  ints(o: JsonObject | undefined, key: string, path: string): Int32Array {
    return this.asInts(this.array(o, key, path), `${path}${key}`);
  }

  optionalInts(o: JsonObject | undefined, key: string, path: string): Int32Array | null {
    const a = this.optionalArray(o, key);
    return a === null ? null : this.asInts(a, `${path}${key}`);
  }

  /** A numeric array that must exist, as float32. */
  floats(o: JsonObject | undefined, key: string, path: string): Float32Array {
    return this.array(o, key, path).float32();
  }

  optionalFloats(o: JsonObject | undefined, key: string): Float32Array | null {
    return this.optionalArray(o, key)?.float32() ?? null;
  }

  /** Per-token flags (stored as bytes) that must exist. */
  flags(o: JsonObject | undefined, key: string, path: string): Uint8Array {
    const a = this.array(o, key, path);
    if (a.isFloat) this.fail(`"${path}${key}" must hold integer flags, not ${a.dtype}`);
    return a.uint8();
  }

  private asInts(a: NdArray, where: string): Int32Array {
    if (a.isFloat) this.fail(`"${where}" must hold integers, not ${a.dtype}`);
    return a.int32();
  }
}

/** Token vocabulary: display labels, optional tag names and optional per-token scalar values. */
export class Vocab {
  readonly labels: string[];
  readonly tagNames: string[];
  /** Per-token numeric value; NaN or negative means "not a number token". */
  readonly values: Float32Array | null;

  constructor(labels: string[], tagNames: string[], values: Float32Array | null) {
    this.labels = labels;
    this.tagNames = tagNames;
    this.values = values;
  }

  get size(): number {
    return this.labels.length;
  }

  /** Display text of token `id`; throws for an id outside the vocabulary (loaders reject those up front). */
  label(id: number): string {
    if (!Number.isInteger(id) || id < 0 || id >= this.labels.length) {
      throw new RangeError(`token id ${id} outside vocabulary of ${this.labels.length}`);
    }
    return this.labels[id];
  }

  tagName(tagId: number): string {
    return tagId >= 0 && tagId < this.tagNames.length ? this.tagNames[tagId] : '';
  }

  /** Numeric value of token `id`, or -1 when the vocabulary has none. */
  valueOf(id: number): number {
    if (this.values === null || id < 0 || id >= this.values.length) return -1;
    return this.values[id];
  }

  hasValue(id: number): boolean {
    const v = this.valueOf(id);
    return !Number.isNaN(v) && v >= 0;
  }
}

export function parseVocab(r: Reader, head: JsonObject): Vocab {
  const vb = r.object(head, 'vocab', '');
  const labels = r.list(vb, 'labels', 'vocab.');
  for (let i = 0; i < labels.length; i++) {
    if (typeof labels[i] !== 'string') r.fail(`"vocab.labels[${i}]" must be a string`);
  }
  const tagNames = r.list(vb, 'tag_names', 'vocab.', false);
  return new Vocab(labels as string[], tagNames.map(String), r.optionalFloats(vb, 'values'));
}

export interface SpanInfo {
  id: number;
  /** First token of the span. */
  start: number;
  /** One past the last token. */
  end: number;
  /** Index of the enclosing span, -1 for a top-level span. */
  parent: number;
  label: string;
  kind: string;
  /** Named tint or #rrggbb, '' for none. */
  tint: string;
  meta: string;
  children: number[];
}

export function parseSpans(r: Reader, head: JsonObject): SpanInfo[] {
  const raw = r.list(head, 'spans', '', false);
  const spans = raw.map((js, i): SpanInfo => {
    const p = `spans[${i}].`;
    if (!isObject(js)) r.fail(`"spans[${i}]" must be an object`);
    return {
      id: r.int(js, 'id', p, i),
      start: r.int(js, 'start', p),
      end: r.int(js, 'end', p),
      parent: r.int(js, 'parent', p, -1),
      label: r.str(js, 'label', p),
      kind: r.str(js, 'kind', p, 'region'),
      tint: r.str(js, 'tint', p, ''),
      meta: r.str(js, 'meta', p, ''),
      children: [],
    };
  });
  spans.forEach((s, i) => {
    if (s.parent >= 0 && s.parent < spans.length && s.parent !== i) spans[s.parent].children.push(i);
  });
  return spans;
}

/** One column of the 3D network graph. */
export interface Stage {
  name: string;
  label: string;
  group: string;
  /** input | layer | latent | output */
  kind: string;
  /** none | source | target */
  tokenSource: string;
  /** Nodes in the column. */
  count: number;
  tokenOffset: number;
  /** Half height of the column in graph units. */
  spanY: number;
  zJitter: number;
  xJitter: number;
  /** Extra distance from the previous column. */
  gapBefore: number;
  /** Per-node activation magnitude. */
  act: Float32Array | null;
}

export function parseStages(r: Reader, list: unknown[], path: string): Stage[] {
  return list.map((js, s): Stage => {
    const p = `${path}[${s}].`;
    if (!isObject(js)) r.fail(`"${path}[${s}]" must be an object`);
    const name = r.str(js, 'name', p, `stage_${s}`);
    const label = r.str(js, 'label', p, name);
    const count = r.int(js, 'count', p);
    if (count < 0) r.fail(`"${p}count" must not be negative`);
    return {
      name,
      label,
      group: r.str(js, 'group', p, label),
      kind: r.str(js, 'kind', p, 'layer'),
      tokenSource: r.str(js, 'token_source', p, 'none'),
      count,
      tokenOffset: r.int(js, 'token_offset', p, 0),
      spanY: r.num(js, 'span_y', p, 150),
      zJitter: r.num(js, 'z_jitter', p, 44),
      xJitter: r.num(js, 'x_jitter', p, 5),
      gapBefore: r.num(js, 'gap_before', p, 0),
      act: r.optionalFloats(js, 'act'),
    };
  });
}

/**
 * Rejects a token id outside the vocabulary now, with its position, instead of failing mid-render.
 * @param what where the ids come from, for the message
 * @param allowNegative negative ids mean "none" (top-k padding)
 */
export function checkTokenIds(r: Reader, what: string, ids: ArrayLike<number>, vocabSize: number, allowNegative = false): void {
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    if (id >= vocabSize || (id < 0 && !allowNegative)) {
      r.fail(`${what}[${i}] is token id ${id}, outside the vocabulary of ${vocabSize}`);
    }
  }
}

/** Trace identity fields common to both file kinds. */
export interface TraceInfo {
  id: string;
  title: string;
  modelName: string;
  arch: string;
  precision: string;
  device: string;
  step: number | null;
  topK: number;
  deltaRedAt: number;
  created: string;
}

export function parseTraceInfo(r: Reader, head: JsonObject): TraceInfo {
  const tr = r.object(head, 'trace', '');
  const id = r.str(tr, 'id', 'trace.', 'trace');
  const rawStep = tr.step;
  if (rawStep !== undefined && rawStep !== null && !Number.isInteger(rawStep)) r.fail('"trace.step" must be an integer');
  return {
    id,
    title: r.str(tr, 'title', 'trace.', id),
    modelName: r.str(tr, 'model_name', 'trace.', 'PyTorch Model'),
    arch: r.str(tr, 'arch', 'trace.', 'custom'),
    precision: r.str(tr, 'precision', 'trace.', 'fp32'),
    device: r.str(tr, 'device', 'trace.', 'cpu'),
    step: (rawStep as number | null | undefined) ?? null,
    topK: r.int(tr, 'top_k', 'trace.', 5),
    deltaRedAt: r.num(tr, 'delta_red_at', 'trace.', 12),
    created: r.str(head, 'created', '', ''),
  };
}
