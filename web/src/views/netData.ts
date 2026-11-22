/**
 * What the three parts of the 3D Net view (the picture, the panel and the controls) share: the graph of the
 * selected token, the clock of the travelling wave, and the text about the node under the pointer.
 */

import { OK, WHITE, hex, type RGB } from '../core/colors.ts';
import { buildNet, selectedToken, type Net } from '../core/net.ts';
import type { TraceDump, UnitTrace } from '../core/trace.ts';
import { app } from '../state/app.ts';
import { NET_CAMERA } from '../state/state.ts';
import { createStore } from '../state/store.ts';

/** Colours of the net: a quiet grey-white for what the prediction did not lean on, white for what it did. */
export const NET_BASE: RGB = hex('#d9d7cd');
export const NET_OUT_OTHER: RGB = hex('#ecebe4');
export const NET_OK: RGB = OK;
export const NET_WHITE: RGB = WHITE;

let last: { key: unknown[]; net: Net } | null = null;

/**
 * The graph for a token of a unit. The picture and the panel ask with the same arguments in one render, so the
 * most recent graph is kept and built only once.
 */
export function netFor(dump: TraceDump, unit: UnitTrace, token: number, head: number, k: number): Net {
  const j = selectedToken(unit, token);
  const key = [dump, unit, j, head, k];
  if (last && last.key.every((v, i) => v === key[i])) return last.net;
  const net = buildNet(dump, unit, j, head, k);
  last = { key, net };
  return net;
}

/** Seconds of the wave clock; the picture advances it while the wave plays and commits it to the state on pause. */
export const netClock = { time: 0 };

/** The description of the node under the pointer, shown in the panel. */
export const netHover = createStore({ text: '' });

export function toggleNetPlay(): void {
  if (app.get().netPlay) app.set({ netPlay: false, netTime: netClock.time });
  else app.set({ netPlay: true });
}

export function resetNetCamera(): void {
  app.set({ netYaw: NET_CAMERA.yaw, netPitch: NET_CAMERA.pitch, netZoom: NET_CAMERA.zoom });
}
