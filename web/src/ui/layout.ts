/**
 * Where a canvas view may draw its content. The pictures fill the whole window and the panels lie on them as
 * glass, so what is visible and not covered is the "free" rectangle: the `.free` element of the shell, whose
 * insets the stylesheet computes from which panels are open. Views frame their content inside it.
 */

import type { Rect } from '../core/math3d.ts';

/** The free rectangle in the coordinates of `canvas` (CSS pixels, origin at its top-left corner). */
export function freeRect(canvas: HTMLCanvasElement): Rect {
  const free = canvas.closest('.app')?.querySelector<HTMLElement>(':scope > .free');
  if (!free) throw new Error('a scene canvas must live inside the app shell, which provides the free area');
  const c = canvas.getBoundingClientRect();
  const f = free.getBoundingClientRect();
  return { x: f.left - c.left, y: f.top - c.top, w: f.width, h: f.height };
}
