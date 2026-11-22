/** The state layer: address-bar round trips, clamping, opening and closing files, the library list, the store. */

import { expect, test } from 'vitest';

import { clampPatch, closeSlotPatch, currentUnit, defaultView, openTracePatch, sameRun, selectTokenPatch, selectUnitPatch, viewAvailable } from '../src/state/actions.ts';
import { groupLibrary, parseLibrary, type LibraryEntry } from '../src/state/library.ts';
import { INITIAL_STATE, type AppState, type FileInfo } from '../src/state/state.ts';
import { createStore } from '../src/state/store.ts';
import { formatHash, parseHash } from '../src/state/urlState.ts';
import { loadDemo, loadDump } from './helpers.ts';

const file = (name: string, id: number | null = null): FileInfo => ({ name, size: 100, id });

function openedT5(): AppState {
  let s: AppState = { ...INITIAL_STATE };
  for (const [trace, info] of [
    [loadDump('t5_small.vtdump'), file('t5_small.vtdump', 0)],
    [loadDemo('t5_small.vtdemo'), file('t5_small.vtdemo', 1)],
  ] as const) {
    s = { ...s, ...openTracePatch(s, trace, info).patch };
  }
  return s;
}

test('the address bar round-trips every mirrored field and leaves defaults out', () => {
  expect(formatHash(INITIAL_STATE, [])).toBe('');
  const s: AppState = {
    ...INITIAL_STATE,
    view: 'attention',
    unit: 1,
    token: 7,
    fam: 2,
    layer: 3,
    head: 4,
    grid: true,
    mode: 'rank',
    latent: 1,
    span: 2,
    netK: 6,
    netPath: true,
    netYaw: 1.25,
    netPitch: -0.3,
    netZoom: 2,
    netPlay: false,
    netTime: 3.5,
    pos: 12.3456,
    playing: false,
    speed: 4,
    loop: false,
    ghost: false,
    yaw: 0.1,
    pitch: 0.2,
    zoom: 1.5,
  };
  const hash = formatHash(s, [0, 1]);
  const back = parseHash(hash);
  expect(back.open).toEqual([0, 1]);
  expect(back.ignored).toEqual([]);
  const { patch } = back;
  expect(patch).toMatchObject({ view: 'attention', unit: 1, token: 7, fam: 2, layer: 3, head: 4, grid: true, mode: 'rank', netK: 6, netPath: true, netPlay: false, playing: false, loop: false, ghost: false });
  expect(patch.pos).toBeCloseTo(12.3456, 4);
  expect(patch.netPitch).toBeCloseTo(-0.3, 4);
  // only the fields that differ from the defaults are written
  expect(formatHash({ ...INITIAL_STATE, token: 5 }, [])).toBe('#token=5');
});

test('what the address bar says that cannot be used is reported, not guessed', () => {
  const r = parseHash('#view=net&token=abc&speed=&nope=1&open=0,x,2&grid=true&nyaw=1e3');
  expect(r.patch).toMatchObject({ view: 'net', grid: true, netYaw: 1000 });
  expect(r.patch.token).toBeUndefined();
  expect(r.patch.speed).toBeUndefined();
  expect(r.open).toEqual([0, 2]);
  expect([...r.ignored].sort()).toEqual(['nope', 'open=0,x,2', 'speed=', 'token=abc'].sort());
});

test('opening the dump and the recording of one run keeps both; another run replaces the other slot', () => {
  const s = openedT5();
  expect(s.dump?.uid).toBe('t5_small_translation');
  expect(s.demo?.uid).toBe('t5_small_translation');
  expect(s.screen).toBe('app');
  expect(sameRun(s.dump!, s.demo!)).toBe(true);

  const qwen = openTracePatch(s, loadDemo('qwen3_0_6b.vtdemo'), file('qwen3_0_6b.vtdemo', 3));
  expect(qwen.notice).toMatch(/qwen3_0_6b\.vtdemo is a different run than t5_small\.vtdump; the inspector was closed\./);
  expect(qwen.patch.dump).toBeNull();
  expect(qwen.patch.view).toBe('replay');

  const qdump = openTracePatch(s, loadDump('qwen3_0_6b.vtdump'), file('qwen3_0_6b.vtdump', 2));
  expect(qdump.notice).toMatch(/is a different run than t5_small\.vtdemo; the replay was closed\./);
  expect(qdump.patch.demo).toBeNull();
  expect(qdump.patch.view).toBe('tokens');
});

test('the first file picks the first tab; a later one keeps the open tab when it can', () => {
  const first = openTracePatch(INITIAL_STATE, loadDemo('t5_small.vtdemo'), file('d.vtdemo'));
  expect(first.patch).toMatchObject({ view: 'replay', screen: 'app', filesOpen: false });
  const s = { ...INITIAL_STATE, ...first.patch };
  const second = openTracePatch({ ...s, view: 'replay' }, loadDump('t5_small.vtdump'), file('d.vtdump'));
  expect(second.patch.view).toBe('replay');
  expect(second.notice).toBeNull();
});

test('closing the last file returns to the start screen with the library and panel kept', () => {
  const s = { ...openedT5(), railOpen: false, library: { status: 'ready' as const, files: [], error: '' } };
  const one = closeSlotPatch({ ...s, view: 'replay' }, 'demo');
  expect(one).toMatchObject({ demo: null, demoFile: null, view: 'tokens' });
  const none = closeSlotPatch({ ...s, ...one } as AppState, 'dump');
  expect(none.screen).toBe('home');
  expect(none.dump).toBeNull();
  expect(none.library).toBe(s.library);
  expect(none.railOpen).toBe(false);
});

test('clampPatch brings every index back into the loaded data', () => {
  const s = openedT5();
  expect(clampPatch(s)).toEqual({});
  const wild = clampPatch({ ...s, unit: 9, token: 99, fam: 7, layer: 40, head: 20, latent: 3, latentRow: 50, span: 8, pos: 1e9, mode: 'nope' as AppState['mode'], netK: 5, speed: 3, view: 'x' as AppState['view'] });
  expect(wild).toMatchObject({ unit: 0, token: 14, fam: 2, layer: 5, head: 7, netK: 3, speed: 2, mode: 'delta', view: 'tokens', span: 2 });
  expect(wild.pos).toBeLessThan(s.demo!.total);
  // a token inside the context prefix moves to the first predicted one
  expect(clampPatch({ ...s, token: 0 }).token).toBe(s.dump!.units[0].t0);
  expect(selectTokenPatch(s, 0).token).toBe(s.dump!.units[0].t0);
  expect(selectTokenPatch(s, 999).token).toBe(s.dump!.units[0].nTokens - 1);
  expect(selectTokenPatch(INITIAL_STATE, 3)).toEqual({});
  expect(selectUnitPatch({ ...s, token: 5, layer: 2 }, 0)).toMatchObject({ unit: 0, token: -1, layer: -1, head: -1 });
  expect(currentUnit(s)).toBe(s.dump!.units[0]);
  expect(currentUnit(INITIAL_STATE)).toBeNull();
});

test('a view is available when the file it reads is open', () => {
  const s = openedT5();
  expect(viewAvailable(s, 'tokens')).toBe(true);
  expect(viewAvailable({ ...s, demo: null }, 'replay')).toBe(false);
  expect(viewAvailable({ ...s, dump: null }, 'net')).toBe(false);
  expect(defaultView({ dump: null })).toBe('replay');
  expect(defaultView(s)).toBe('tokens');
});

const entry = (over: Partial<LibraryEntry>): LibraryEntry => ({ id: 0, name: 'a.vtdump', kind: 'dump', size: 10, traceId: 't', title: 'T', model: 'm', arch: 'a', source: 'sample', error: '', ...over });

test('the library list is checked at the boundary and grouped by run', () => {
  const body = {
    files: [
      { id: 0, name: 'a.vtdump', kind: 'dump', size: 10, traceId: 't', title: 'Title', model: 'm', arch: 'decoder_only', source: 'sample' },
      { id: 1, name: 'a.vtdemo', kind: 'demo', size: 20, traceId: 't', title: 'Title', model: 'm', arch: 'decoder_only', source: 'sample' },
      { id: 2, name: 'broken.vtdump', source: 'argument', error: 'broken.vtdump: truncated file' },
    ],
  };
  const files = parseLibrary(body);
  expect(files.map((f) => f.id)).toEqual([0, 1, 2]);
  const groups = groupLibrary(files);
  expect(groups.length).toBe(2);
  expect(groups[0].dump?.id).toBe(0);
  expect(groups[0].demo?.id).toBe(1);
  expect(groups[1].error).toBe('broken.vtdump: truncated file');
  expect(groups[1].dump).toBeNull();
  expect(groupLibrary([entry({ id: 0 }), entry({ id: 1, traceId: 'other', kind: 'demo' })]).length).toBe(2);
  // a file given by name is listed before the samples and wins over a sample of the same run
  const dup = groupLibrary([entry({ id: 0, name: 'mine.vtdump', source: 'argument' }), entry({ id: 1, name: 'sample.vtdump', source: 'sample' })]);
  expect(dup.length).toBe(1);
  expect(dup[0].dump?.name).toBe('mine.vtdump');

  expect(() => parseLibrary({})).toThrow('the answer has no file list');
  expect(() => parseLibrary({ files: [3] })).toThrow('files[0] is not an object');
  expect(() => parseLibrary({ files: [{ id: -1, name: 'x' }] })).toThrow('files[0].id is not a file number');
  expect(() => parseLibrary({ files: [{ id: 0, name: 5 }] })).toThrow('files[0].name is not a string');
  expect(() => parseLibrary({ files: [{ id: 0, name: 'x', kind: 'movie', size: 1 }] })).toThrow('files[0].kind is "movie"');
  expect(() => parseLibrary({ files: [{ id: 0, name: 'x', kind: 'dump', size: 'big' }] })).toThrow('files[0].size is not a number');
  expect(() => parseLibrary({ files: [{ id: 0, name: 'x', kind: 'dump', size: 1, title: 7 }] })).toThrow('files[0].title is not a string');
});

test('the store tells watchers only about real changes, and names them', () => {
  const store = createStore({ a: 1, b: 'x', c: [1] });
  const seen: string[][] = [];
  let pulses = 0;
  const off = store.watch((_s, _p, changed) => seen.push(changed.map(String)));
  const offReact = store.subscribe(() => pulses++);
  store.set({ a: 1 });
  expect(seen).toEqual([]);
  store.set({ a: 2, b: 'x' });
  expect(seen).toEqual([['a']]);
  expect(store.get().a).toBe(2);
  const arr = store.get().c;
  store.set({ c: arr });
  expect(seen.length).toBe(1);
  store.set({ c: [1] });
  expect(seen).toEqual([['a'], ['c']]);
  expect(pulses).toBe(2);
  off();
  offReact();
  store.set({ a: 3 });
  expect(seen.length).toBe(2);
  expect(pulses).toBe(2);
});
