/**
 * Prints what the browser reader says about a file: "OK", or "REFUSED " and the message of the refusal.
 *
 * tests/test_web.py runs this on broken containers and compares the message with the one the Python reader
 * raises for the same bytes, so that a bad file reads the same in `vitro info` and in the web app.
 *
 * Usage:  node web/tests/refuse.ts <file> [name used in the message, default: the path]
 */

import { readFileSync } from 'node:fs';

import { Container, ContainerError } from '../src/core/container.ts';

const [path, name = path] = process.argv.slice(2);
if (!path) {
  process.stderr.write('usage: node refuse.ts <file> [name]\n');
  process.exit(2);
}
const buf = readFileSync(path);
try {
  new Container(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer, name);
  process.stdout.write('OK\n');
} catch (e) {
  if (!(e instanceof ContainerError)) throw e;
  process.stdout.write(`REFUSED ${e.message}\n`);
}
