/**
 * Pure helpers that turn the state and a request into a state patch, keeping every index inside the loaded
 * data. Port of VitroApp::clamp() and the selection rules of the old viewer, plus the rules for putting a file
 * into the session. Nothing here touches the page, so the tests run it as it is.
 */

import { COLOR_MODES } from '../core/analysis.ts';
import { NET_KS } from '../core/net.ts';
import type { Trace } from '../core/open.ts';
import { SPEEDS } from '../core/replay.ts';
import type { UnitTrace } from '../core/trace.ts';
import { INITIAL_STATE, VIEWS, type AppState, type FileInfo, type ViewId } from './state.ts';

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

/** The unit shown by the inspector views, or null. */
export function currentUnit(state: Pick<AppState, 'dump' | 'unit'>): UnitTrace | null {
  return state.dump ? state.dump.units[state.unit] : null;
}

/** Whether the file a view reads is loaded. */
export function viewAvailable(state: Pick<AppState, 'dump' | 'demo'>, id: ViewId): boolean {
  const view = VIEWS.find((v) => v.id === id);
  return view !== undefined && (view.needs === 'dump' ? state.dump !== null : state.demo !== null);
}

/** The first tab that has something to show: the inspector when a dump is loaded, else the replay. */
export function defaultView(state: Pick<AppState, 'dump'>): ViewId {
  return state.dump ? 'tokens' : 'replay';
}

/** The patch that brings out-of-range indices back into range for the loaded data; empty when all is well. */
export function clampPatch(state: AppState): Partial<AppState> {
  const patch: Record<string, unknown> = {};
  const set = <K extends keyof AppState>(key: K, value: AppState[K]): void => {
    if (!Object.is(state[key], value)) patch[key] = value;
  };
  const { dump, demo } = state;
  if (dump) {
    const unit = clamp(state.unit, 0, dump.units.length - 1);
    const u = dump.units[unit];
    let token = clamp(state.token, -1, u.nTokens - 1);
    if (token >= 0 && token < u.t0 && u.nTokens > u.t0) token = u.t0;
    let fam = 0;
    let layer = -1;
    let head = -1;
    if (u.attnFamilies.length) {
      fam = clamp(state.fam, 0, u.attnFamilies.length - 1);
      const af = u.attnFamilies[fam];
      layer = clamp(state.layer, -1, af.layers - 1);
      head = clamp(state.head, -1, af.heads - 1);
    }
    let latent = 0;
    let latentRow = -1;
    if (u.latents.length) {
      latent = clamp(state.latent, 0, u.latents.length - 1);
      latentRow = clamp(state.latentRow, -1, u.latents[latent].data.shape[0] - 1);
    }
    set('unit', unit);
    set('token', token);
    set('fam', fam);
    set('layer', layer);
    set('head', head);
    set('latent', latent);
    set('latentRow', latentRow);
    set('span', dump.spans.length ? clamp(state.span, -1, dump.spans.length - 1) : -1);
  }
  if (demo) set('pos', clamp(state.pos, 0, Math.max(0, demo.total - 1e-6)));
  if (!COLOR_MODES.includes(state.mode)) set('mode', 'delta');
  if (!(NET_KS as readonly number[]).includes(state.netK)) set('netK', 3);
  if (!(SPEEDS as readonly number[]).includes(state.speed)) {
    set('speed', SPEEDS.reduce((a, b) => (Math.abs(b - state.speed) < Math.abs(a - state.speed) ? b : a)));
  }
  if (!VIEWS.some((v) => v.id === state.view)) set('view', defaultView(state));
  return patch as Partial<AppState>;
}

/**
 * `patch` together with whatever it takes to keep every index inside the data after the patch is applied. One
 * `set` with this keeps components from ever drawing a state whose indices point outside the loaded file.
 */
export function withClamp(state: AppState, patch: Partial<AppState>): Partial<AppState> {
  return { ...patch, ...clampPatch({ ...state, ...patch }) };
}

/** Selecting a unit resets everything that belongs to the previous one. */
export function selectUnitPatch(state: AppState, unit: number): Partial<AppState> {
  const count = state.dump ? state.dump.units.length : 1;
  return {
    unit: clamp(unit, 0, count - 1),
    token: -1,
    fam: 0,
    layer: -1,
    head: -1,
    latent: 0,
    latentRow: -1,
  };
}

/** Selects target token `j`; a token of the given context prefix becomes the first predicted token instead. */
export function selectTokenPatch(state: AppState, j: number): Partial<AppState> {
  const u = currentUnit(state);
  if (!u) return {};
  let token = clamp(j, -1, u.nTokens - 1);
  if (token >= 0 && token < u.t0 && u.nTokens > u.t0) token = u.t0;
  return { token };
}

/** True when two parsed traces describe the same run (same id, model and vocabulary size). */
export function sameRun(a: Trace, b: Trace): boolean {
  return a.uid === b.uid && a.info.modelName === b.info.modelName && a.vocab.size === b.vocab.size;
}

/** What opening a file does: the patch, and a notice for the person when it replaced the other file. */
export interface OpenResult {
  patch: Partial<AppState>;
  notice: string | null;
}

/** Selections that belong to the dump, back to their start. */
const DUMP_SELECTION: Partial<AppState> = { unit: 0, token: -1, fam: 0, layer: -1, head: -1, latent: 0, latentRow: -1, span: -1 };

/**
 * Puts a parsed trace into its slot of the session. A dump and a recording of the same run share the session;
 * a file of another run replaces the other slot, and the result says so.
 */
export function openTracePatch(state: AppState, trace: Trace, info: FileInfo): OpenResult {
  let patch: Partial<AppState>;
  let notice: string | null = null;
  if (trace.kind === 'dump') {
    const keepDemo = state.demo !== null && sameRun(trace, state.demo);
    if (state.demo && !keepDemo) notice = `${info.name} is a different run than ${state.demoFile?.name ?? 'the open recording'}; the replay was closed.`;
    patch = {
      ...DUMP_SELECTION,
      dump: trace,
      dumpFile: info,
      demo: keepDemo ? state.demo : null,
      demoFile: keepDemo ? state.demoFile : null,
      netTime: 0,
    };
    if (!keepDemo) patch.pos = 0;
  } else {
    const keepDump = state.dump !== null && sameRun(trace, state.dump);
    if (state.dump && !keepDump) notice = `${info.name} is a different run than ${state.dumpFile?.name ?? 'the open dump'}; the inspector was closed.`;
    patch = {
      demo: trace,
      demoFile: info,
      dump: keepDump ? state.dump : null,
      dumpFile: keepDump ? state.dumpFile : null,
      pos: 0,
      playing: true,
    };
    if (!keepDump) Object.assign(patch, DUMP_SELECTION);
  }
  const next = { ...state, ...patch };
  const wanted: ViewId = state.screen === 'app' ? state.view : trace.kind === 'dump' ? 'tokens' : 'replay';
  patch.view = viewAvailable(next, wanted) ? wanted : defaultView(next);
  patch.screen = 'app';
  patch.filesOpen = false;
  return { patch, notice };
}

/** Closes one slot of the session; with nothing left, the app returns to the start screen with its defaults. */
export function closeSlotPatch(state: AppState, kind: 'dump' | 'demo'): Partial<AppState> {
  const patch: Partial<AppState> = kind === 'dump' ? { dump: null, dumpFile: null } : { demo: null, demoFile: null };
  const next = { ...state, ...patch };
  if (!next.dump && !next.demo) {
    return { ...INITIAL_STATE, library: state.library, railOpen: state.railOpen, filesOpen: state.filesOpen, screen: 'home' };
  }
  patch.view = viewAvailable(next, state.view) ? state.view : defaultView(next);
  return patch;
}
