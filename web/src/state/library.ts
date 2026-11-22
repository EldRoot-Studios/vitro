/**
 * The list of traces the server knows (`GET api/library`): the sample runs and the files named on the
 * command line. A dump and a recording of one run are shown as one card, because together they open every tab.
 */

/** One entry of the list: a readable trace file, or a file the server could not read (`error` is set). */
export interface LibraryEntry {
  id: number;
  name: string;
  kind: 'dump' | 'demo' | '';
  size: number;
  traceId: string;
  title: string;
  model: string;
  arch: string;
  source: string;
  error: string;
}

/** The files of one run, as one card. */
export interface RunGroup {
  key: string;
  title: string;
  model: string;
  arch: string;
  dump: LibraryEntry | null;
  demo: LibraryEntry | null;
  error: string;
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * Checks the body of `api/library` and turns it into entries. A body that does not look right is an error that
 * names the field, not an empty list: an empty list would say "no traces" about a server that answered badly.
 */
export function parseLibrary(body: unknown): LibraryEntry[] {
  if (!isRecord(body) || !Array.isArray(body.files)) throw new Error('the answer has no file list');
  return body.files.map((raw: unknown, i: number): LibraryEntry => {
    const where = `files[${i}]`;
    if (!isRecord(raw)) throw new Error(`${where} is not an object`);
    if (typeof raw.id !== 'number' || !Number.isInteger(raw.id) || raw.id < 0) throw new Error(`${where}.id is not a file number`);
    if (typeof raw.name !== 'string') throw new Error(`${where}.name is not a string`);
    const text = (key: string): string => {
      const v = raw[key];
      if (v === undefined || v === null) return '';
      if (typeof v !== 'string') throw new Error(`${where}.${key} is not a string`);
      return v;
    };
    const error = text('error');
    const kind = text('kind');
    if (error === '' && kind !== 'dump' && kind !== 'demo') throw new Error(`${where}.kind is "${kind}", expected "dump" or "demo"`);
    const size = raw.size;
    if (error === '' && (typeof size !== 'number' || !Number.isFinite(size) || size < 0)) throw new Error(`${where}.size is not a number`);
    return {
      id: raw.id,
      name: raw.name,
      kind: kind === 'dump' || kind === 'demo' ? kind : '',
      size: typeof size === 'number' ? size : 0,
      traceId: text('traceId'),
      title: text('title') || raw.name,
      model: text('model'),
      arch: text('arch'),
      source: text('source'),
      error,
    };
  });
}

/**
 * Groups entries into runs. Entries the server could not read stay visible as their own cards with their error.
 */
export function groupLibrary(files: readonly LibraryEntry[]): RunGroup[] {
  const groups = new Map<string, RunGroup>();
  for (const f of files) {
    const key = f.error ? `error:${f.id}` : `${f.traceId}|${f.model}`;
    let g = groups.get(key);
    if (!g) {
      g = { key, title: f.title, model: f.model, arch: f.arch, dump: null, demo: null, error: '' };
      groups.set(key, g);
    }
    // the first file of a kind wins: files given by name are listed before the samples, so they are the ones opened
    if (f.error) g.error = f.error;
    else if (f.kind === 'demo') g.demo ??= f;
    else g.dump ??= f;
  }
  return [...groups.values()];
}
