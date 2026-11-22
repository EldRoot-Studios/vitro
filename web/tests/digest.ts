/**
 * Prints a numeric digest of a trace file, read through the same code the web app uses.
 *
 * tests/test_web.py runs this on the sample traces and compares the digest with the one NumPy computes from the
 * raw container (`python_digest` in tests/test_viewer.py, `check_tape` in tests/test_demo.py). Two independent
 * readers agreeing to 1e-9 is the proof that the browser sees the numbers the exporter wrote.
 *
 * Node runs this file directly (it strips the types), so it imports the core modules with their .ts names:
 *   node web/tests/digest.ts <file.vtdump | file.vtdemo>
 */

import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { TraceDemo } from '../src/core/demo.ts';
import { openTrace } from '../src/core/open.ts';
import type { TraceDump } from '../src/core/trace.ts';

/** Sum of the entries, skipping NaN, like numpy.nansum on float64. */
export function nansum(data: ArrayLike<number> | null | undefined): number {
  let s = 0;
  if (data === null || data === undefined) return s;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v === v) s += v;
  }
  return s;
}

/** Digest of a TraceDump, with the keys of `python_digest`. */
export function dumpDigest(dump: TraceDump) {
  return {
    uid: dump.uid,
    labels: dump.vocab.size,
    spans: dump.spans.length,
    units: dump.units.map((u) => ({
      u: u.u,
      n: u.nTokens,
      free_n: u.freeN,
      diverge: u.diverge,
      sum_tok: nansum(u.tok),
      sum_p: nansum(u.p),
      sum_rank: nansum(u.rank),
      sum_top_p: nansum(u.topP),
      sum_entropy: nansum(u.entropy),
      sum_weights: u.links.reduce((a, lk) => a + nansum(lk.weights?.data), 0),
      sum_act: u.stages.reduce((a, st) => a + nansum(st.act), 0),
      sum_norms: nansum(u.norms.data?.data),
    })),
  };
}

/** Digest of a TraceDemo, with the keys of `check_tape`. */
export function demoDigest(demo: TraceDemo) {
  return {
    uid: demo.uid,
    segments: demo.segments.map((sg) => ({
      k: sg.k,
      m: sg.ftok.length,
      sum_ftok: nansum(sg.ftok),
      sum_fp: nansum(sg.fp),
      sum_top_p: nansum(sg.tape.topP.data),
      sum_self_w: nansum(sg.tape.selfW.data),
      sum_norm: nansum(sg.tape.norm.data),
    })),
  };
}

function main(): void {
  const path = process.argv[2];
  if (!path) {
    process.stderr.write('usage: node digest.ts <file.vtdump | file.vtdemo>\n');
    process.exit(2);
  }
  const buf = readFileSync(path);
  const trace = openTrace(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer, basename(path));
  const digest = trace.kind === 'dump' ? dumpDigest(trace) : demoDigest(trace);
  process.stdout.write(`${JSON.stringify(digest)}\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
