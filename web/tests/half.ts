/**
 * Prints the SHA-256 of what the browser reader makes of a float16 array: every value as float32 bits, with
 * every NaN written as the canonical quiet NaN (the payload of a NaN carries no meaning).
 *
 * tests/test_web.py writes all 65536 float16 bit patterns with the Python writer and compares this digest with
 * the one NumPy gives for the same patterns: the two decoders agree on every pattern, not only on a few.
 *
 * Usage:  node web/tests/half.ts <file with header key "h" naming the array>
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { Container } from '../src/core/container.ts';

const path = process.argv[2];
if (!path) {
  process.stderr.write('usage: node half.ts <file>\n');
  process.exit(2);
}
const buf = readFileSync(path);
const c = new Container(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer, path);
const values = c.array(c.header.h).float32();
const bits = new Uint32Array(values.length);
const view = new Float32Array(1);
const as32 = new Uint32Array(view.buffer);
for (let i = 0; i < values.length; i++) {
  view[0] = values[i];
  bits[i] = Number.isNaN(values[i]) ? 0x7fc00000 : as32[0];
}
process.stdout.write(`${createHash('sha256').update(new Uint8Array(bits.buffer)).digest('hex')} ${values.length}\n`);
