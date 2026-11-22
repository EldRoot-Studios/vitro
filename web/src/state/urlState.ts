/**
 * State that is mirrored in the address bar as `#key=value&...`, so a reload or a pasted link returns to the same
 * view. The files themselves are named by the server's number (`open=0,1`); picked files cannot be, so a link
 * only restores what the server has.
 */

import { INITIAL_STATE, type AppState } from './state.ts';

type UrlType = 's' | 'i' | 'f' | 'b';
type UrlKey = 'view' | 'unit' | 'token' | 'fam' | 'layer' | 'head' | 'grid' | 'mode' | 'latent' | 'span' | 'netK' | 'netPath' | 'netYaw' | 'netPitch' | 'netZoom' | 'netPlay' | 'netTime' | 'pos' | 'playing' | 'speed' | 'loop' | 'ghost' | 'yaw' | 'pitch' | 'zoom';

/** [url key, state key, type]: s string, i integer, f number, b boolean written as 1 or 0. */
export const URL_FIELDS: readonly (readonly [string, UrlKey, UrlType])[] = [
  ['view', 'view', 's'],
  ['unit', 'unit', 'i'],
  ['token', 'token', 'i'],
  ['fam', 'fam', 'i'],
  ['layer', 'layer', 'i'],
  ['head', 'head', 'i'],
  ['grid', 'grid', 'b'],
  ['mode', 'mode', 's'],
  ['latent', 'latent', 'i'],
  ['span', 'span', 'i'],
  ['k', 'netK', 'i'],
  ['path', 'netPath', 'b'],
  ['nyaw', 'netYaw', 'f'],
  ['npitch', 'netPitch', 'f'],
  ['nzoom', 'netZoom', 'f'],
  ['nplay', 'netPlay', 'b'],
  ['ntime', 'netTime', 'f'],
  ['pos', 'pos', 'f'],
  ['playing', 'playing', 'b'],
  ['speed', 'speed', 'f'],
  ['loop', 'loop', 'b'],
  ['ghost', 'ghost', 'b'],
  ['ryaw', 'yaw', 'f'],
  ['rpitch', 'pitch', 'f'],
  ['rzoom', 'zoom', 'f'],
];

export interface ParsedHash {
  patch: Partial<AppState>;
  /** Server file numbers named by `open=`. */
  open: number[];
  /** Keys and values that were not understood; the caller tells the person. */
  ignored: string[];
}

/** Parses the address-bar fragment into a state patch plus the ids of server files to open. */
export function parseHash(hash: string): ParsedHash {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const patch: Record<string, unknown> = {};
  const ignored: string[] = [];
  const known = new Set(['open', ...URL_FIELDS.map(([k]) => k)]);
  for (const key of params.keys()) if (!known.has(key)) ignored.push(key);
  for (const [urlKey, stateKey, type] of URL_FIELDS) {
    const raw = params.get(urlKey);
    if (raw === null) continue;
    let value: string | number | boolean | undefined;
    if (type === 's') value = raw;
    else if (type === 'b') value = raw === '1' || raw === 'true';
    else if (type === 'i') value = /^-?\d+$/.test(raw) ? Number(raw) : undefined;
    else value = raw.trim() !== '' && Number.isFinite(Number(raw)) ? Number(raw) : undefined;
    if (value === undefined) ignored.push(`${urlKey}=${raw}`);
    else patch[stateKey] = value;
  }
  const open = (params.get('open') ?? '')
    .split(',')
    .filter((s) => s !== '')
    .map((s) => (/^\d+$/.test(s) ? Number(s) : NaN));
  if (open.some(Number.isNaN)) ignored.push(`open=${params.get('open')}`);
  return { patch: patch as Partial<AppState>, open: open.filter((n) => !Number.isNaN(n)), ignored };
}

/**
 * Serialises the mirrored state, leaving out values equal to the defaults.
 * @param open ids of the server files in the session
 */
export function formatHash(state: AppState, open: readonly number[]): string {
  const params = new URLSearchParams();
  if (open.length) params.set('open', open.join(','));
  for (const [urlKey, stateKey, type] of URL_FIELDS) {
    const value = state[stateKey];
    if (value === INITIAL_STATE[stateKey]) continue;
    const text = type === 'b' ? (value ? '1' : '0') : type === 'f' ? String(Math.round(Number(value) * 10000) / 10000) : String(value);
    params.set(urlKey, text);
  }
  const s = params.toString();
  return s ? `#${s}` : '';
}
