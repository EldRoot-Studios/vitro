/** Hooks the views share. */

import { useEffect, type CSSProperties, type RefObject } from 'react';

import type { TraceDemo } from '../core/demo.ts';
import type { Rect } from '../core/math3d.ts';
import type { TraceDump, UnitTrace } from '../core/trace.ts';
import { useApp } from '../state/app.ts';

/** The open dump. The shell mounts a dump view only when one is open, so a missing dump is a programming error. */
export function useDump(): TraceDump {
  const dump = useApp((s) => s.dump);
  if (!dump) throw new Error('this view needs a trace dump, and none is open');
  return dump;
}

/** The open recording; see `useDump`. */
export function useDemo(): TraceDemo {
  const demo = useApp((s) => s.demo);
  if (!demo) throw new Error('this view needs a generation recording, and none is open');
  return demo;
}

/** The open dump and the unit the inspector views show. */
export function useUnit(): { dump: TraceDump; unit: UnitTrace } {
  const dump = useDump();
  const index = useApp((s) => s.unit);
  return { dump, unit: dump.units[index] };
}

/** Redraws a canvas when its box changes size. */
export function useResizeRedraw(canvas: RefObject<HTMLElement | null>, request: () => void): void {
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const resize = new ResizeObserver(request);
    resize.observe(el);
    return () => resize.disconnect();
  }, [canvas, request]);
}

/** Room on the right for the labels of the output candidates, which stand beside the last column of a 3D graph. */
export const LABEL_ROOM = 72;

/** The rectangle a 3D graph is framed in: the free area less the room its output labels need. */
export function withLabelRoom(free: Rect): Rect {
  return { ...free, w: Math.max(1, free.w - LABEL_ROOM) };
}

/** CSS custom properties as a style object (React's CSSProperties does not know them). */
export function vars(values: Record<string, string | number>): CSSProperties {
  return values as CSSProperties;
}

/** The text of a label on one line, for a place with no room for whitespace. */
export const compact = (label: string): string => label.replace(/\s+/g, ' ').trim() || '·';
