/**
 * Opening files: from the file picker, from drag and drop, or from the server that serves the samples. A dump
 * and a recording of the same run share one session; opening a file of a different run replaces the other file
 * and says so. Every failure is shown to the person, with the parser's own message naming the file.
 */

import type { Trace } from '../core/open.ts';
import { openTrace } from '../core/open.ts';
import { AUTOPLAY, app } from './app.ts';
import { closeSlotPatch, openTracePatch, withClamp } from './actions.ts';
import { parseLibrary, type LibraryEntry, type RunGroup } from './library.ts';
import type { FileInfo } from './state.ts';
import { toasts } from './toasts.ts';

interface Parsed {
  trace: Trace;
  info: FileInfo;
}

interface Source {
  name: string;
  id: number | null;
  load: () => Promise<ArrayBuffer | null>;
}

const reason = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** Parses a file; on failure tells the person why and returns null. */
function parse(buffer: ArrayBuffer, name: string): Trace | null {
  try {
    return openTrace(buffer, name);
  } catch (e) {
    toasts.show(reason(e), 'err');
    return null;
  }
}

/** Puts a parsed trace into the session and brings the indices back into range. */
function apply({ trace, info }: Parsed): void {
  const state = app.get();
  const { patch, notice } = openTracePatch(state, trace, info);
  if (notice) toasts.show(notice, 'warn');
  // a recording starts playing when it is opened, unless the person asked for less motion
  if (!AUTOPLAY && patch.playing) patch.playing = false;
  app.set(withClamp(state, patch));
}

/** Applies parsed traces, dumps first so that a dump and its recording end up in one session. */
function applyAll(parsed: Parsed[]): number {
  for (const p of [...parsed].sort((a, b) => Number(a.trace.kind === 'demo') - Number(b.trace.kind === 'demo'))) apply(p);
  return parsed.length;
}

/** Reads and parses, in parallel, the given sources; failed ones are reported and left out. */
async function parseAll(sources: Source[]): Promise<Parsed[]> {
  const out = await Promise.all(
    sources.map(async ({ name, id, load }): Promise<Parsed | null> => {
      const buffer = await load();
      const trace = buffer ? parse(buffer, name) : null;
      return buffer && trace ? { trace, info: { name, size: buffer.byteLength, id } } : null;
    }),
  );
  return out.filter((p): p is Parsed => p !== null);
}

async function readFile(file: File): Promise<ArrayBuffer | null> {
  try {
    return await file.arrayBuffer();
  } catch (e) {
    toasts.show(`${file.name}: cannot read the file (${reason(e)})`, 'err');
    return null;
  }
}

async function fetchEntry(entry: LibraryEntry): Promise<ArrayBuffer | null> {
  try {
    const res = await fetch(new URL(`files/${entry.id}`, document.baseURI));
    if (!res.ok) throw new Error(`the server answered ${res.status}`);
    return await res.arrayBuffer();
  } catch (e) {
    toasts.show(`${entry.name}: cannot load the file (${reason(e)})`, 'err');
    return null;
  }
}

/** Opens picked or dropped files. Returns how many were opened. */
export async function openFiles(files: Iterable<File>): Promise<number> {
  return applyAll(await parseAll([...files].map((f) => ({ name: f.name, id: null, load: () => readFile(f) }))));
}

/** Opens library entries served by `vitro web`. Returns how many were opened. */
export async function openServerFiles(entries: readonly (LibraryEntry | null)[]): Promise<number> {
  const real = entries.filter((e): e is LibraryEntry => e !== null);
  return applyAll(await parseAll(real.map((e) => ({ name: e.name, id: e.id, load: () => fetchEntry(e) }))));
}

/** Opens the dump and the recording of one library card. */
export function openGroup(group: RunGroup): Promise<number> {
  return openServerFiles([group.dump, group.demo]);
}

/** Closes one slot of the session; with nothing left, returns to the start screen. */
export function closeSlot(kind: 'dump' | 'demo'): void {
  app.set(closeSlotPatch(app.get(), kind));
}

/** Asks the server for the list of samples and given files. */
export async function loadLibrary(): Promise<void> {
  try {
    const res = await fetch(new URL('api/library', document.baseURI));
    if (!res.ok) throw new Error(`the server answered ${res.status}`);
    app.set({ library: { status: 'ready', files: parseLibrary(await res.json()), error: '' } });
  } catch (e) {
    app.set({ library: { status: 'unavailable', files: [], error: reason(e) } });
  }
}
