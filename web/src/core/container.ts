/**
 * Reader for the aligned binary trace container ("VITRO01"), the browser twin of python/vitro/container.py.
 *
 * Layout (little-endian):
 *   bytes 0..7    8-byte magic "VITRO01\n"
 *   bytes 8..15   unsigned 64-bit JSON header length
 *   bytes 16..    UTF-8 JSON header, zero-padded to a 16-byte boundary
 *   data          raw C-order tensors, each starting on a 16-byte boundary
 *
 * The header holds array descriptors {dtype, off, shape}; `Container.array()` turns one into an NdArray.
 * Every refusal names the file and what was wrong, in the same words as the Python reader, so a bad file
 * reads the same in the viewer and in `vitro info`.
 *
 * Sizes this must handle: headers of a few MB (a 150k-token vocabulary) and data sections of hundreds of MB.
 * Arrays are zero-copy views over the file buffer, except float16, which is widened to float32 on first use.
 * This module is free of the DOM so that Node can run it directly.
 */

export const MAGIC = 'VITRO01\n';
export const ALIGN = 16;
export const VERSION = 1;

/** Raised for every malformed container; the message is meant for the person who picked the file. */
export class ContainerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ContainerError';
  }
}

export type JsonObject = Record<string, unknown>;
export type DType = 'f4' | 'f2' | 'i4' | 'i2' | 'i1' | 'u1';
export type TypedData = Float32Array | Int32Array | Int16Array | Int8Array | Uint8Array;

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

interface DTypeInfo {
  bytes: number;
  /** A view over the file buffer; null for float16, which has no native view and is decoded instead. */
  view: ((buf: ArrayBuffer, off: number, n: number) => TypedData) | null;
}

const DTYPES: Record<DType, DTypeInfo> = {
  f4: { bytes: 4, view: (buf, off, n) => new Float32Array(buf, off, n) },
  f2: { bytes: 2, view: null },
  i4: { bytes: 4, view: (buf, off, n) => new Int32Array(buf, off, n) },
  i2: { bytes: 2, view: (buf, off, n) => new Int16Array(buf, off, n) },
  i1: { bytes: 1, view: (buf, off, n) => new Int8Array(buf, off, n) },
  u1: { bytes: 1, view: (buf, off, n) => new Uint8Array(buf, off, n) },
};

let halfTable: Float32Array | null = null;

/** float16 bit pattern -> float32 value, for all 65536 patterns (built once, 256 KB). */
function getHalfTable(): Float32Array {
  if (halfTable) return halfTable;
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const man = h & 0x3ff;
    if (exp === 0) table[h] = sign * man * 2 ** -24;
    else if (exp === 31) table[h] = man ? NaN : sign * Infinity;
    else table[h] = sign * (1 + man / 1024) * 2 ** (exp - 15);
  }
  halfTable = table;
  return table;
}

/** A C-order array of one dtype. `data` is a typed array (float16 arrives widened to Float32Array). */
export class NdArray {
  readonly dtype: DType;
  readonly shape: number[];
  readonly length: number;
  private load: (() => TypedData) | null;
  private cache: TypedData | null = null;

  /** @param load produces the typed array; called at most once */
  constructor(dtype: DType, shape: number[], load: () => TypedData) {
    this.dtype = dtype;
    this.shape = shape;
    this.length = shape.reduce((a, b) => a * b, 1);
    this.load = load;
  }

  get data(): TypedData {
    if (this.cache === null) {
      this.cache = this.load!();
      this.load = null;
    }
    return this.cache;
  }

  /** True for float32 and float16 arrays. */
  get isFloat(): boolean {
    return this.dtype === 'f4' || this.dtype === 'f2';
  }

  /** Size of dimension `i`; throws for a dimension the array does not have. */
  dim(i: number): number {
    if (i < 0 || i >= this.shape.length) throw new RangeError(`array of shape [${this.shape}] has no dimension ${i}`);
    return this.shape[i];
  }

  /** The values as float32: a view for f4, the widened copy for f2, a converted copy for integers. */
  float32(): Float32Array {
    const d = this.data;
    return d instanceof Float32Array ? d : Float32Array.from(d);
  }

  /** The values as int32: a view for i4, a widened copy for smaller integers. Callers check `isFloat` first. */
  int32(): Int32Array {
    const d = this.data;
    return d instanceof Int32Array ? d : Int32Array.from(d);
  }

  /** The values as bytes: a view for u1, a converted copy otherwise. Callers check `isFloat` first. */
  uint8(): Uint8Array {
    const d = this.data;
    return d instanceof Uint8Array ? d : Uint8Array.from(d);
  }
}

/** True for exactly the shape of a descriptor written by the Python writer: {dtype, off, shape}. */
export function isDescriptor(x: unknown): x is { dtype: unknown; off: unknown; shape: unknown } {
  if (x === null || typeof x !== 'object' || Array.isArray(x)) return false;
  const keys = Object.keys(x);
  return keys.length === 3 && 'dtype' in x && 'off' in x && 'shape' in x;
}

const pad = (n: number): number => (ALIGN - (n % ALIGN)) % ALIGN;

export class Container {
  readonly name: string;
  readonly buffer: ArrayBuffer;
  readonly header: JsonObject;
  readonly dataAt: number;
  readonly dataSize: number;

  /**
   * Validates `buffer` and parses its header.
   * @param buffer the whole file
   * @param name file name used in messages
   */
  constructor(buffer: ArrayBuffer, name = 'file') {
    if (!LITTLE_ENDIAN) throw new ContainerError(`${name}: Vitro traces are little-endian; this platform is not`);
    if (!(buffer instanceof ArrayBuffer)) throw new TypeError('Container needs an ArrayBuffer');
    this.name = name;
    this.buffer = buffer;
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 16) throw new ContainerError(`${name}: truncated file (less than 16 bytes)`);
    for (let i = 0; i < MAGIC.length; i++) {
      if (bytes[i] !== MAGIC.charCodeAt(i)) {
        throw new ContainerError(`${name}: not a Vitro trace file`);
      }
    }
    const headerLen = Number(new DataView(buffer).getBigUint64(8, true));
    if (headerLen === 0 || 16 + headerLen > bytes.length) {
      throw new ContainerError(`${name}: header says ${headerLen} bytes, file has ${bytes.length - 16}`);
    }
    let text: string;
    try {
      text = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(16, 16 + headerLen));
    } catch (e) {
      throw new ContainerError(`${name}: header is not valid UTF-8 (${(e as Error).message})`);
    }
    let header: unknown;
    try {
      header = JSON.parse(text);
    } catch (e) {
      throw new ContainerError(`${name}: header is not valid JSON (${(e as Error).message})`);
    }
    if (header === null || typeof header !== 'object' || Array.isArray(header)) {
      throw new ContainerError(`${name}: header must be a JSON object`);
    }
    this.header = header as JsonObject;
    if (this.header.version !== VERSION) {
      throw new ContainerError(`${name}: container version ${String(this.header.version)}, expected ${VERSION}`);
    }
    this.dataAt = 16 + headerLen + pad(headerLen);
    this.dataSize = bytes.length - this.dataAt;
    if (this.dataSize < 0) throw new ContainerError(`${name}: padding runs past end of file`);
  }

  /** Resolves an array descriptor into an NdArray over the file buffer. */
  array(desc: unknown): NdArray {
    if (!isDescriptor(desc)) throw new ContainerError(`${this.name}: ${JSON.stringify(desc)} is not an array descriptor`);
    const { dtype, off, shape } = desc;
    if (typeof dtype !== 'string' || !Object.hasOwn(DTYPES, dtype)) {
      throw new ContainerError(`${this.name}: unknown dtype ${JSON.stringify(dtype)}`);
    }
    const kind = dtype as DType;
    const spec = DTYPES[kind];
    if (typeof off !== 'number' || !Number.isInteger(off) || off < 0 || off % ALIGN !== 0) {
      throw new ContainerError(`${this.name}: offset ${String(off)} lies outside the data section (must be >= 0 and ${ALIGN}-aligned)`);
    }
    if (!Array.isArray(shape) || shape.some((s) => typeof s !== 'number' || !Number.isInteger(s) || s < 0)) {
      throw new ContainerError(`${this.name}: invalid shape ${JSON.stringify(shape)}`);
    }
    const dims = shape as number[];
    const count = dims.reduce((a, b) => a * b, 1);
    if (off + count * spec.bytes > this.dataSize) {
      throw new ContainerError(`${this.name}: array ${kind}[${dims}] at ${off} lies outside the data section (${this.dataSize} bytes)`);
    }
    const at = this.dataAt + off;
    if (spec.view === null) {
      return new NdArray(kind, dims, () => {
        const table = getHalfTable();
        const raw = new Uint16Array(this.buffer, at, count);
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = table[raw[i]];
        return out;
      });
    }
    const view = spec.view;
    return new NdArray(kind, dims, () => view(this.buffer, at, count));
  }
}
