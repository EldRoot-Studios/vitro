/**
 * The one store of the app and the hooks that read it. Components never copy state: `useApp` subscribes to the
 * part of it they show, and a change to any other part does not redraw them.
 */

import { useSyncExternalStore } from 'react';

import { INITIAL_STATE, type AppState } from './state.ts';
import { createStore } from './store.ts';

/** Windows narrower than this start with the left panel closed, so the picture is not squeezed. */
export const NARROW_PX = 1100;

/** False when the person asked the system for less motion: recordings and the wave then start paused. */
export const AUTOPLAY = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export const app = createStore<AppState>({ ...INITIAL_STATE, railOpen: window.innerWidth >= NARROW_PX, playing: AUTOPLAY, netPlay: AUTOPLAY });

/**
 * The value `select` picks from the state. `select` runs on every state change, so it must return a primitive or
 * an object that already exists in the state; a freshly built object would look like a change every time.
 */
export function useApp<T>(select: (state: AppState) => T): T {
  return useSyncExternalStore(app.subscribe, () => select(app.get()));
}
