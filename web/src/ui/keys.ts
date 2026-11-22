/**
 * What the open view can register with the window: a key handler (which says whether it used the key) and a
 * canvas to save as a picture. The window-wide handler in App.tsx asks the view after its own keys.
 */

import { useEffect, useLayoutEffect, useRef } from 'react';

export type KeyHandler = (e: KeyboardEvent) => boolean;

let viewKeys: KeyHandler | null = null;
let snapshotSource: (() => HTMLCanvasElement | null) | null = null;

/** Gives the open view first refusal of a key; returns true if it used it. */
export function handleViewKey(e: KeyboardEvent): boolean {
  return viewKeys?.(e) ?? false;
}

/** The canvas of the open view, or null when it has no picture to save. */
export function viewSnapshot(): HTMLCanvasElement | null {
  return snapshotSource?.() ?? null;
}

/** Registers the keys of the view for as long as it is mounted; the latest handler is the one called. */
export function useViewKeys(handler: KeyHandler): void {
  const ref = useRef(handler);
  useLayoutEffect(() => {
    ref.current = handler;
  });
  useEffect(() => {
    const mine: KeyHandler = (e) => ref.current(e);
    viewKeys = mine;
    return () => {
      if (viewKeys === mine) viewKeys = null;
    };
  }, []);
}

/** Registers the canvas of the view as the thing the camera button saves. */
export function useSnapshotSource(get: () => HTMLCanvasElement | null): void {
  const ref = useRef(get);
  useLayoutEffect(() => {
    ref.current = get;
  });
  useEffect(() => {
    const mine = (): HTMLCanvasElement | null => ref.current();
    snapshotSource = mine;
    return () => {
      if (snapshotSource === mine) snapshotSource = null;
    };
  }, []);
}
