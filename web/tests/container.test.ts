/** Container reader: layout, dtypes (float16 above all) and every refusal the Python reader makes. */

import { expect, test } from 'vitest';

import { Container, ContainerError } from '../src/core/container.ts';
import { TestWriter, buildFile } from './helpers.ts';

/** Fails unless `fn` throws a ContainerError whose message matches `pattern`. */
function refuses(fn: () => unknown, pattern: RegExp): void {
  let caught: unknown = null;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  expect(caught, `expected a ContainerError matching ${pattern}`).toBeInstanceOf(ContainerError);
  expect((caught as Error).message).toMatch(pattern);
}

test('reads every dtype back from 16-byte aligned offsets', () => {
  const w = new TestWriter();
  const header = {
    f4: w.f4([1.5, -2.25, 3]),
    i4: w.i4([1, -2, 300000]),
    i2: w.add('i2', Int16Array.from([-3, 4]), [2]),
    i1: w.add('i1', Int8Array.from([-1, 2, 3]), [3]),
    u1: w.u1([0, 255, 7]),
    m: w.f4([1, 2, 3, 4, 5, 6], [2, 3]),
  };
  const c = new Container(buildFile(header, w), 'mem.vtdump');
  expect([...c.array(c.header.f4).data]).toEqual([1.5, -2.25, 3]);
  expect([...c.array(c.header.i4).data]).toEqual([1, -2, 300000]);
  expect([...c.array(c.header.i2).data]).toEqual([-3, 4]);
  expect([...c.array(c.header.i1).data]).toEqual([-1, 2, 3]);
  expect([...c.array(c.header.u1).data]).toEqual([0, 255, 7]);
  const m = c.array(c.header.m);
  expect(m.shape).toEqual([2, 3]);
  expect(m.dim(1)).toBe(3);
  expect(() => m.dim(2)).toThrow(RangeError);
  for (const d of Object.values(header)) expect(d.off % 16).toBe(0);
});

test('widens float16 exactly, including subnormals, infinities and NaN', () => {
  const w = new TestWriter();
  const bits = [0x3c00, 0xc000, 0x0001, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x0000, 0x8000, 0x0400];
  const desc = w.f2bits(bits);
  const c = new Container(buildFile({ h: desc }, w), 'half.vtdump');
  const v = c.array(c.header.h).data;
  expect(v[0]).toBe(1);
  expect(v[1]).toBe(-2);
  expect(v[2]).toBe(2 ** -24);
  expect(v[3]).toBe(65504);
  expect(v[4]).toBe(Infinity);
  expect(v[5]).toBe(-Infinity);
  expect(Number.isNaN(v[6])).toBe(true);
  expect(v[7]).toBe(0);
  expect(Object.is(v[8], -0)).toBe(true);
  expect(v[9]).toBe(2 ** -14);
  expect(v).toBeInstanceOf(Float32Array);
});

/** float16 -> float32 by moving bits (no arithmetic on the value), an algorithm independent of the reader's table. */
function halfToFloatByBits(h: number): number {
  const sign = (h & 0x8000) << 16;
  let exp = (h >> 10) & 0x1f;
  let man = h & 0x3ff;
  let bits: number;
  if (exp === 0) {
    if (man === 0) bits = sign;
    else {
      exp = 113;
      while (!(man & 0x400)) {
        man <<= 1;
        exp--;
      }
      bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
    }
  } else if (exp === 31) bits = sign | 0x7f800000 | (man << 13);
  else bits = sign | ((exp + 112) << 23) | (man << 13);
  const f = new Float32Array(1);
  new Uint32Array(f.buffer)[0] = bits >>> 0;
  return f[0];
}

test('all 65536 float16 patterns decode to what a bit-level conversion gives', () => {
  const w = new TestWriter();
  const desc = w.f2bits(Array.from({ length: 65536 }, (_, i) => i));
  const v = new Container(buildFile({ h: desc }, w), 'all.vtdump').array(desc).data;
  let nan = 0;
  let firstWrong = -1;
  for (let h = 0; h < 65536; h++) {
    const want = halfToFloatByBits(h);
    const same = Number.isNaN(want) ? Number.isNaN(v[h]) : Object.is(v[h], want);
    if (Number.isNaN(want) && same) nan++;
    if (!same && firstWrong < 0) firstWrong = h;
  }
  expect(firstWrong, `first pattern that decodes differently (0x${firstWrong.toString(16)})`).toBe(-1);
  expect(nan).toBe(2 * 1023);
});

test('zero-length arrays are legal when the offset is inside the data section', () => {
  const w = new TestWriter();
  w.f4([1]);
  const c = new Container(buildFile({ e: { dtype: 'f4', off: 16, shape: [0, 3] } }, w), 'empty.vtdump');
  expect(c.array(c.header.e).data.length).toBe(0);
});

test('refuses files that are not containers, and says why', () => {
  const good = buildFile({ x: 1 });
  refuses(() => new Container(good.slice(0, 10), 'a.vtdump'), /a\.vtdump: truncated file \(less than 16 bytes\)/);
  refuses(() => new Container(buildFile({ x: 1 }, new TestWriter(), { magic: 'NOTVITRO' }), 'b.vtdump'), /b\.vtdump: not a Vitro trace file/);
  refuses(() => new Container(buildFile({ x: 1 }, new TestWriter(), { headerLen: 99999 }), 'c.vtdump'), /c\.vtdump: header says 99999 bytes, file has/);
  refuses(() => new Container(buildFile({ x: 1 }, new TestWriter(), { headerLen: 0 }), 'd.vtdump'), /header says 0 bytes/);
  refuses(() => new Container(buildFile('{"version":2}'), 'e.vtdump'), /e\.vtdump: container version 2, expected 1/);
  refuses(() => new Container(buildFile('{"x":'), 'f.vtdump'), /f\.vtdump: header is not valid JSON/);
  refuses(() => new Container(buildFile('[1,2]'), 'g.vtdump'), /header must be a JSON object/);
  refuses(() => new Container(new TextEncoder().encode('hello').buffer as ArrayBuffer, 'h.txt'), /h\.txt: truncated/);
});

test('refuses bad array descriptors with the Python wording', () => {
  const w = new TestWriter();
  const ok = w.f4([1, 2, 3, 4]);
  const c = new Container(buildFile({ ok }, w), 'arr.vtdump');
  refuses(() => c.array({ dtype: 'f4', off: 3, shape: [1] }), /offset 3 lies outside the data section \(must be >= 0 and 16-aligned\)/);
  refuses(() => c.array({ dtype: 'f4', off: -16, shape: [1] }), /offset -16 lies outside/);
  refuses(() => c.array({ dtype: 'f4', off: 0, shape: [1000] }), /array f4\[1000\] at 0 lies outside the data section \(16 bytes\)/);
  refuses(() => c.array({ dtype: 'f8', off: 0, shape: [1] }), /unknown dtype "f8"/);
  refuses(() => c.array({ dtype: 'f4', off: 0, shape: [-1] }), /invalid shape/);
  refuses(() => c.array({ dtype: 'f4', off: 0, shape: [1.5] }), /invalid shape/);
  refuses(() => c.array({ dtype: 'f4', off: 0 }), /is not an array descriptor/);
  refuses(() => c.array({ dtype: 'f4', off: 0, shape: [1], extra: 1 }), /is not an array descriptor/);
  refuses(() => c.array(null), /is not an array descriptor/);
  refuses(() => c.array({ dtype: 'constructor', off: 0, shape: [1] }), /unknown dtype "constructor"/);
});

test('rejects a non-ArrayBuffer input as a programming error', () => {
  expect(() => new Container(new Uint8Array(32) as unknown as ArrayBuffer, 'x')).toThrow(TypeError);
});
