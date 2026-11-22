/** Trace models: the sample traces, one-defect-at-a-time refusals, and the maths the 3D scenes stand on. */

import { expect, test } from 'vitest';

import { attnMap, cellColor, describeToken, errorTokens, latentCos, nextError, spanStats, type ColorMode } from '../src/core/analysis.ts';
import { heat, hex, deltaError, luminance, parseTint, surprisal, textOn, TINTS } from '../src/core/colors.ts';
import { Container, ContainerError } from '../src/core/container.ts';
import { parseDemo } from '../src/core/demo.ts';
import { Camera3D, frameCamera, hash01, topK } from '../src/core/math3d.ts';
import { buildNet, describeNode, pickNode, projectNodes } from '../src/core/net.ts';
import { openTrace } from '../src/core/open.ts';
import { buildScene, composeFrame, stepPosition } from '../src/core/replay.ts';
import { parseDump } from '../src/core/trace.ts';
import { loadDemo, loadDump, miniDemo, miniDump } from './helpers.ts';

/** Fails unless opening the buffer made by `make` throws a ContainerError whose message matches `pattern`. */
function refuses(make: () => ArrayBuffer, pattern: RegExp): void {
  let caught: unknown = null;
  try {
    openTrace(make(), 'bad.vt');
  } catch (e) {
    caught = e;
  }
  expect(caught, `expected a refusal matching ${pattern}`).toBeInstanceOf(ContainerError);
  expect((caught as Error).message).toMatch(pattern);
}

test('T5 sample dump: structure the exporter wrote', () => {
  const d = loadDump('t5_small.vtdump');
  expect(d.kind).toBe('dump');
  expect(d.uid).toBe('t5_small_translation');
  expect(d.vocab.size).toBe(32128);
  expect(d.spans.length).toBe(3);
  const u = d.units[0];
  expect(u.nTokens).toBe(15);
  expect(u.nSource).toBe(14);
  expect(u.freeN).toBe(15);
  expect(u.stages.length).toBe(14);
  expect(u.attnFamilies.map((f) => [f.name, f.layers, f.heads, f.rows, f.cols, f.causal])).toEqual([
    ['Encoder Self-Attn', 6, 8, 14, 14, false],
    ['Decoder Self-Attn', 6, 8, 14, 14, true],
    ['Cross-Attn', 6, 8, 14, 14, false],
  ]);
  expect(u.norms.data?.shape).toEqual([6, 1, 14]);
  expect(d.label(u.tok[7])).toBe(' verre');
  expect(Math.abs(u.bits - 4.4835)).toBeLessThan(1e-3);
  expect(u.forcedExact && u.freeExact).toBe(true);
});

test('Qwen sample dump: a causal model with misses', () => {
  const d = loadDump('qwen3_0_6b.vtdump');
  const u = d.units[0];
  expect(u.attnFamilies.length).toBe(1);
  expect(u.attnFamilies[0].layers).toBe(28);
  expect(u.attnFamilies[0].causal).toBe(true);
  expect([...u.rank.slice(0, 6)]).toEqual([0, 106, 14, 5743, 2779, 32]);
  expect(errorTokens(u)).toEqual([1, 2, 3, 4, 5]);
  expect(nextError(u, -1, 1)).toBe(1);
  expect(nextError(u, 5, 1)).toBe(-1);
  expect(nextError(u, -1, -1)).toBe(5);
  expect(nextError(u, 3, -1)).toBe(2);
  expect(d.spansCovering(3).map((s) => s.label)).toEqual(['Prompt']);
  expect(d.spansCovering(8).map((s) => s.label)).toEqual(['Generated Tokens']);
  const s = describeToken(d, u, 3);
  expect(s).toMatch(/^token 3 " routes" \(Prompt\) - p=\d\.\d{3}, rank=5743 \(top-1: ".*" \d\.\d\d\)$/);
  expect(describeToken(d, u, 0).endsWith('context prefix')).toBe(true);
  const st = spanStats(u, 0, 6);
  expect(st.lo).toBe(0);
  expect(st.hi).toBe(6);
  expect(st.errors).toBe(5);
});

test('sample recordings: timeline, prefill, cross-attention', () => {
  const t5 = loadDemo('t5_small.vtdemo');
  expect(t5.kind).toBe('demo');
  expect(t5.total).toBe(22);
  const sg = t5.segments[0];
  expect(sg.readTicks).toBe(8);
  expect(sg.writeTicks).toBe(14);
  expect(sg.prefillLinks.length).toBe(6);
  expect(sg.crossStage).toBe(6);
  expect(sg.srcTok?.length).toBe(14);
  const qwen = loadDemo('qwen3_0_6b.vtdemo');
  expect(qwen.total).toBe(10);
  expect(qwen.segments[0].crossStage).toBe(-1);
  expect(qwen.segments[0].prefillLinks.length).toBe(0);
});

test('openTrace picks the reader from the header kind and refuses unknown kinds', () => {
  expect(openTrace(miniDump(), 'a').kind).toBe('dump');
  expect(openTrace(miniDemo(), 'b').kind).toBe('demo');
  refuses(() => miniDump(({ header }) => (header.kind = 'movie')), /unknown recording kind "movie"/);
  expect(() => parseDump(new Container(miniDemo(), 'x.vtdemo'))).toThrow(/x\.vtdemo: this is a 'demo' recording, not a trace dump/);
  expect(() => parseDemo(new Container(miniDump(), 'y.vtdump'))).toThrow(/y\.vtdump: this is a 'dump' trace, not a generation recording/);
});

test('mini dump parses and its derived maps are right', () => {
  const d = loadMiniDump();
  const u = d.units[0];
  expect(u.nTokens).toBe(4);
  expect(d.spanDepth(0)).toBe(0);
  expect(d.label(2)).toBe('cat');
  expect(() => d.label(8)).toThrow(RangeError);
  // one family, one layer, two heads: head 0 is all 0.5, head 1 is all 1.0
  const mean = attnMap(u, 0, -1, -1);
  expect(mean.length).toBe(16);
  expect([...mean].every((v) => v === 0.75)).toBe(true);
  expect([...attnMap(u, 0, 0, 0)].every((v) => v === 0.5)).toBe(true);
  expect([...attnMap(u, 0, 0, 1)].every((v) => v === 1)).toBe(true);
  expect([...attnMap(u, 0, -1, 1)].every((v) => v === 1), 'all layers of head 1 is head 1').toBe(true);
  expect(attnMap(u, 0, -1, -1), 'repeat calls hit the cache').toBe(mean);
  const cos = latentCos(u, 0);
  expect([cos[0], cos[4], cos[8]].map((v) => +v.toFixed(6))).toEqual([1, 1, 1]);
  expect(Math.abs(cos[1]), 'rows (1,0) and (0,1) are orthogonal').toBeLessThan(1e-6);
  expect(Math.abs(cos[2] - Math.SQRT1_2), 'rows (1,0) and (1,1) are 45 degrees apart').toBeLessThan(1e-6);
  const modes: ColorMode[] = ['delta', 'surprisal', 'rank', 'entropy'];
  expect(modes.map((m) => cellColor(d, u, 2, m).wrong)).toEqual([true, true, true, true]);
  expect(cellColor(d, u, 0, 'surprisal').wrong).toBe(false);
});

function loadMiniDump() {
  const d = openTrace(miniDump(), 'mini.vtdump');
  if (d.kind !== 'dump') throw new Error('the mini dump must open as a dump');
  return d;
}

test('a dump with defects is refused with a message that names the defect', () => {
  refuses(() => miniDump(({ w, unit }) => (unit.sequence.p = w.f4([1, 2, 3]))), /unit 0: per-token metric shapes do not match tokens length/);
  refuses(() => miniDump(({ w, unit }) => (unit.sequence.tokens = w.i4([0, 99, 2, 3]))), /unit 0 tokens\[1\] is token id 99, outside the vocabulary of 8/);
  refuses(() => miniDump(({ w, unit }) => (unit.sequence.free_tok = w.i4([0, 1, -3, 3]))), /unit 0 free rollout\[2\] is token id -3/);
  refuses(() => miniDump(({ header }) => (header.units = [])), /trace contains zero units/);
  refuses(() => miniDump(({ unit }) => (unit.t0 = 9)), /t0 9 lies outside the 4 tokens/);
  refuses(() => miniDump(({ unit }) => (unit.links[0].dst = 5)), /unit 0: link stage index out of range/);
  refuses(() => miniDump(({ header }) => (header.trace.top_k = 3)), /top-k arrays hold 8 ids for 4 tokens, expected top_k \(3\)/);
  refuses(() => miniDump(({ header }) => (header.trace.top_k = 2.5)), /"trace.top_k" must be an integer/);
  refuses(() => miniDump(({ header }) => delete header.vocab), /missing "vocab"/);
  refuses(() => miniDump(({ header }) => (header.vocab.labels = ['a', 3])), /"vocab.labels\[1\]" must be a string/);
  refuses(() => miniDump(({ unit }) => delete unit.sequence.tokens), /missing "units\[0\]\.sequence\.tokens"/);
  refuses(() => miniDump(({ w, unit }) => (unit.norms.data = w.f4([1, 2, 3, 4], [4]))), /norms data must be \(layers, components, tokens\)/);
  refuses(() => miniDump(({ w, unit }) => (unit.latents[0].data = w.f4([1, 2, 3, 4], [4]))), /latent map "State" must be 2D/);
  refuses(() => miniDump(({ w, unit }) => (unit.sequence.free_p = w.f4([1, 1]))), /free rollout tokens and probabilities do not match/);
  refuses(
    () =>
      miniDump(({ w, unit }) => {
        unit.links.push({ ...unit.links[1], name: 'Self L2', weights: w.f2bits(new Array(3 * 4 * 4).fill(0), [3, 4, 4]) });
      }),
    /attention family 'Self-Attn' has mismatched layer shapes/,
  );
  refuses(() => miniDump(({ w, unit }) => (unit.links[1].weights = w.f2bits([0, 0, 0, 0], [4]))), /unit 0: attention weights must be 2D or 3D/);
  refuses(() => miniDump(({ unit }) => (unit.links[1].weights = { dtype: 'f2', off: 0, shape: [1000, 1000] })), /array f2\[1000,1000\] at 0 lies outside the data section/);
  refuses(() => miniDump(({ w, unit }) => (unit.sequence.rank = w.f4([0, 1, 2, 1]))), /"units\[0\]\.sequence\.rank" must hold integers, not f4/);
});

test('a recording with defects is refused with a message that names the defect', () => {
  refuses(() => miniDemo(({ header }) => (header.segments = [])), /recording contains zero segments/);
  refuses(() => miniDemo(({ w, seg }) => (seg.free_tok = w.i4([0]))), /segment 0: no generated tokens past t0=1/);
  refuses(() => miniDemo(({ w, seg }) => (seg.free_p = w.f4([1, 1]))), /segment 0: free_p has 2 entries for 4 tokens/);
  refuses(() => miniDemo(({ seg }) => (seg.decode.out_stage = 9)), /segment 0: decode stage index out of range/);
  refuses(() => miniDemo(({ seg }) => (seg.decode.cross_stage = 1)), /segment 0: cross-attention stage given but the tape has no cross links/);
  refuses(() => miniDemo(({ w, seg }) => (seg.tape.top_id = w.i4([1, 2, 3, 4], [2, 2]))), /segment 0: tape top-k must be \(4, 2\); got \[2,2\]/);
  refuses(() => miniDemo(({ w, seg }) => (seg.tape.norm = w.f2bits([0, 0, 0, 0], [4, 2, 1]))), /segment 0: tape norm must be \(4, 1, channels\)/);
  refuses(() => miniDemo(({ w, seg }) => (seg.free_tok = w.i4([0, 1, 2, 77]))), /segment 0 free tokens\[3\] is token id 77/);
  refuses(
    () => miniDemo(({ w, seg }) => (seg.prefill_links = [{ src: 0, dst: 7, kind: 'self', tint: 'blue', ids: w.i4([0, 0], [2, 1]), weights: w.f4([1, 1], [2, 1]) }])),
    /prefill link stage index out of range/,
  );
  // ids are read as integers later; a float array in their place must be refused, not silently truncated
  refuses(() => miniDemo(({ w, seg }) => (seg.tape.top_id = w.f4([-1, -1, 1, 4, 2, 6, 3, 5], [4, 2]))), /"segments\[0\]\.tape\.top_id" must hold integers, not f4/);
  refuses(() => miniDemo(({ w, seg }) => (seg.tape.self_id = w.f4([0, 0, 0, 1, 0, 1, 1, 2], [4, 1, 2]))), /"segments\[0\]\.tape\.self_id" must hold integers, not f4/);
  refuses(
    () => miniDemo(({ w, seg }) => (seg.prefill_links = [{ src: 0, dst: 1, kind: 'self', tint: 'blue', ids: w.f4([0, 0], [2, 1]), weights: w.f4([1, 1], [2, 1]) }])),
    /"segments\[0\]\.prefill_links\[0\]\.ids" must hold integers, not f4/,
  );
});

test('hash01 matches the C++ original bit for bit', () => {
  // values printed by cpp/include/vitro/math3d.hpp hash01 (g++ -O2 -ffp-contract=off)
  const ref: [number, number, number][] = [
    [0, 0, 0.1716303825378418],
    [1, 0, 0.184284508228302],
    [0, 1, 0.41093218326568604],
    [3, 9001, 0.05402684211730957],
    [7, 9007, 0.52689766883850098],
    [14, 0, 0.40752381086349487],
    [29, 15, 0.35846453905105591],
    [12, 9012, 0.21990901231765747],
    [5, 12345, 0.14372193813323975],
    [4294967295, 2, 0.42897456884384155],
    [100, 50, 0.55947107076644897],
    [2, 9000, 0.32673531770706177],
  ];
  for (const [a, b, want] of ref) expect(hash01(a | 0, b), `hash01(${a}, ${b})`).toBe(want);
});

test('topK keeps the largest above the floor, largest first, ties by index', () => {
  const v = [0.1, 0.4, 0.4, 0.05, 0.9, 0.003];
  expect(topK(v, 0, 6, 3, 0.005)).toEqual([4, 1, 2]);
  expect(topK(v, 0, 6, 8, 0.005)).toEqual([4, 1, 2, 0, 3]);
  expect(topK(v, 0, 6, 1, 0.005)).toEqual([4]);
  expect(topK(v, 0, 6, 0, 0.005)).toEqual([]);
  expect(topK(v, 0, 0, 3, 0.005)).toEqual([]);
  expect(topK(v, 2, 3, 2, 0.005), 'window [2,5) holds 0.4, 0.05, 0.9: the two largest are at 2 and 0').toEqual([2, 0]);
  expect(topK([NaN, 0.2, NaN], 0, 3, 3, 0), 'NaN never becomes an edge').toEqual([1]);
});

test('graph of T5 token 7 matches the old viewer: 15 stages, 201 nodes, 911 edges', () => {
  const d = loadDump('t5_small.vtdump');
  const net = buildNet(d, d.units[0], 7, -1, 3);
  expect(net.nStages).toBe(15);
  expect(net.nodes.n).toBe(201);
  expect(net.edges.n).toBe(911);
  expect(net.nodes.stage[net.trueOut]).toBe(net.sOut);
  expect(d.label(net.nodes.idx[net.trueOut])).toBe(' verre');
  for (let e = 0; e < net.edges.n; e++) {
    expect(net.edges.a[e] >= 0 && net.edges.a[e] < net.nodes.n && net.edges.b[e] >= 0 && net.edges.b[e] < net.nodes.n).toBe(true);
    expect(net.edges.imp[e] >= 0 && net.edges.imp[e] <= 1).toBe(true);
  }
  expect([...net.hot].every((e) => net.edges.imp[e] >= 0.2)).toBe(true);
  expect(net.nodes.imp[net.first[net.sPred] + net.selP], 'the explained node carries all the influence').toBe(1);
  const k8 = buildNet(d, d.units[0], 7, -1, 8);
  expect(k8.edges.n, 'more incoming edges per node means more edges').toBeGreaterThan(net.edges.n);
  const head0 = buildNet(d, d.units[0], 7, 0, 3);
  expect(head0.edges.imp.join(), 'one head differs from the mean over heads').not.toBe(net.edges.imp.join());
});

test('graph of the Qwen sample is stable (regression pin of this implementation)', () => {
  const d = loadDump('qwen3_0_6b.vtdump');
  const net = buildNet(d, d.units[0], 9, -1, 3);
  expect([net.nStages, net.nodes.n, net.edges.n]).toEqual([30, 440, 1600]);
  expect(describeNode(d, d.units[0], net, net.outFirst).startsWith('output "')).toBe(true);
});

test('the framing camera keeps every node inside the view, whatever the angle', () => {
  const d = loadDump('t5_small.vtdump');
  const net = buildNet(d, d.units[0], 7, -1, 3);
  const view = { x: 20, y: 30, w: 900, h: 520 };
  const frame = (yaw: number, pitch: number, zoom: number) =>
    frameCamera({ yaw, pitch, zoom, halfSpan: net.halfSpan, view, xs: net.nodes.x, ys: net.nodes.y, zs: net.nodes.z, fitH: 0.74 });
  for (const [yaw, pitch] of [[0.3, 0.14], [0, 0], [1.2, -0.6], [-2.5, 1.0]]) {
    const s = projectNodes(net.nodes, frame(yaw, pitch, 1));
    for (let i = 0; i < net.nodes.n; i++) {
      const inside = s.x[i] >= view.x - 1 && s.x[i] <= view.x + view.w + 1 && s.y[i] >= view.y - 1 && s.y[i] <= view.y + view.h + 1;
      expect(inside, `yaw ${yaw} pitch ${pitch}: node ${i} at ${s.x[i]},${s.y[i]}`).toBe(true);
    }
    const i = 17;
    expect(pickNode(s, s.x[i], s.y[i])).toBeGreaterThanOrEqual(0);
    expect(pickNode(s, -500, -500)).toBe(-1);
  }
  const out = new Float32Array(3);
  const out2 = new Float32Array(3);
  frame(0.3, 0.14, 2).map(0, 0, 0, out);
  frame(0.3, 0.14, 1).map(0, 0, 0, out2);
  expect(Math.abs(out[2] / out2[2] - 2), 'zoom scales the perspective factor').toBeLessThan(1e-6);
  expect(new Camera3D().zoom).toBe(1);
});

test('heat ramp rises in luminance, and the other ramps keep their ends', () => {
  let prev = -1;
  for (let i = 0; i <= 200; i++) {
    const l = luminance(heat(i / 200));
    expect(l, `luminance must not fall at t=${i / 200}`).toBeGreaterThanOrEqual(prev - 1e-9);
    prev = l;
  }
  expect(luminance(heat(0)), 'the ramp starts at near-black').toBeLessThan(0.06);
  expect(luminance(heat(1)), 'and ends at white').toBeGreaterThan(0.99);
  expect(surprisal(0)).toEqual(heat(0));
  expect(surprisal(50)).toEqual(heat(1));
  expect(deltaError(0)).toEqual(hex('#275a52'));
  expect(deltaError(12, 24).map(Math.round), 'redAt 24 stretches the ramp twofold').toEqual(deltaError(6).map(Math.round));
  expect(parseTint('gold', [0, 0, 0])).toEqual(TINTS.gold);
  expect(parseTint('#102030', [0, 0, 0])).toEqual([16, 32, 48]);
  expect(parseTint('nonsense', [1, 2, 3])).toEqual([1, 2, 3]);
  expect(parseTint('', [1, 2, 3])).toEqual([1, 2, 3]);
  expect(parseTint('constructor', [1, 2, 3])).toEqual([1, 2, 3]);
  expect(textOn(hex('#000000'))).toEqual(hex('#f4f4f5'));
  expect(textOn(hex('#ffffff'))).toEqual(hex('#0b0c0e'));
  expect(() => hex('red')).toThrow(RangeError);
});

test('playback clock: locate, positionOfToken and stepping agree', () => {
  const d = loadDemo('t5_small.vtdemo');
  const sg = d.segments[0];
  expect(d.locate(0)).toEqual({ seg: 0, reading: true, readP: 0, token: sg.t0, frac: 0.5 });
  expect(d.locate(4).readP).toBe(0.5);
  const first = d.locate(8);
  expect([first.reading, first.token, first.frac]).toEqual([false, 1, 0]);
  for (let k = sg.t0; k < sg.ftok.length; k++) {
    const n = d.locate(d.positionOfToken(0, k));
    expect([n.reading, n.token]).toEqual([false, k]);
  }
  expect(d.locate(1e9).token, 'past the end clamps to the last token').toBe(sg.ftok.length - 1);
  expect(d.locate(-5).reading, 'before the start clamps to the start').toBe(true);
  expect(d.positionOfToken(0, 999)).toBe(d.positionOfToken(0, sg.ftok.length - 1));
  expect(stepPosition(d, 0, 1), 'stepping from reading lands one token after its end').toBe(8 + 1);
  expect(stepPosition(d, d.positionOfToken(0, 5), -1)).toBe(d.positionOfToken(0, 4));
  expect(stepPosition(d, 1e9, 10)).toBe(d.total - 1e-6);
  expect(stepPosition(d, 0, -10)).toBe(0);
});

test('two segments share one timeline', () => {
  const d = openTrace(
    miniDemo(({ header, w, seg }) => {
      const second = JSON.parse(JSON.stringify(seg));
      second.k = 1;
      second.has_read = false;
      second.free_tok = w.i4([0, 1, 2, 3]);
      second.free_p = w.f4([1, 1, 1, 1]);
      // the descriptors of the copied tape are valid (same arrays); only the segment record repeats
      header.segments.push(second);
    }),
    'two.vtdemo',
  );
  if (d.kind !== 'demo') throw new Error('the mini recording must open as a recording');
  expect(d.segments[1].start).toBe(d.segments[0].readTicks + d.segments[0].writeTicks);
  expect(d.total).toBe(4 + 3 + 3);
  expect(d.segmentAt(6.9)).toBe(0);
  expect(d.segmentAt(7)).toBe(1);
  expect(d.locate(7.2).seg).toBe(1);
});

test('replay frames: reading wave, then per-step links, pulses and candidate halos', () => {
  const d = loadDemo('t5_small.vtdemo');
  const sg = d.segments[0];
  const scene = buildScene(sg);
  expect(scene.nStages).toBe(15);
  expect(scene.nodes.n).toBe(14 * 14 + 5);
  const reading = composeFrame(d, sg, scene, d.locate(3));
  expect(reading.lines.length, 'only prefill links while reading').toBe(scene.edges.n);
  expect(reading.halos.length).toBe(0);
  expect(reading.caption.startsWith('T5-Small Rollout: encoding source context (37%)')).toBe(true);
  expect(reading.lines.every((l) => l.alpha > 0 && l.alpha <= 0.9)).toBe(true);

  const now = d.locate(d.positionOfToken(0, 7));
  const step = composeFrame(d, sg, scene, { ...now, frac: 0.95 });
  expect(step.caption).toBe('T5-Small Rollout: generating token 7 of 14');
  expect(step.lines.length).toBeGreaterThan(scene.edges.n);
  expect(step.pulses.length).toBeGreaterThan(0);
  const labels = step.halos.filter((h) => h.label !== undefined).map((h) => h.label);
  const top = Array.from({ length: 5 }, (_, o) => d.label(sg.tape.topId.data[7 * 5 + o]));
  expect(labels, 'one halo per candidate, in rank order').toEqual(top);
  expect(step.halos.filter((h) => h.label !== undefined && !h.labelDim).length, 'exactly one candidate was chosen').toBe(1);
  for (const l of step.lines) {
    expect(l.a >= 0 && l.a < scene.nodes.n && l.b >= 0 && l.b < scene.nodes.n && l.alpha >= 0 && l.alpha <= 1).toBe(true);
  }
  for (const p of step.pulses) expect(p.t >= 0 && p.t < 1).toBe(true);
  expect(step.nodes.filter((n) => n === null).length, 'the candidate column is drawn only as halos').toBe(5);
  const early = composeFrame(d, sg, scene, { ...now, frac: 0.0 });
  expect(early.lines.length, 'layers are revealed progressively during a step').toBeLessThan(step.lines.length);
});
