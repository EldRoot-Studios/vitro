/**
 * The clock of the replay and what moves it. The picture advances `clock.pos` every frame; the rest of the page
 * sees the position only when it is committed to the state: when the token changes, on pause and on seek. That
 * keeps the panels from being redrawn sixty times a second. The transport and the panel that show the live
 * position subscribe to `clock.frame` and move their elements by hand.
 */

import { stepPosition } from '../core/replay.ts';
import { app } from '../state/app.ts';
import type { AppState } from '../state/state.ts';
import { clamp } from '../ui/format.ts';

export const clock = {
  /** Live position on the timeline in ticks. */
  pos: 0,
  /** Seconds the camera has been swaying, for the slow turn while playing. */
  sway: 0,
  /** Called after every frame the picture draws. */
  frame: new Set<() => void>(),
  /** Asks the picture for a frame; set while the picture is mounted. */
  request: null as (() => void) | null,
};

/** Commits the live position, with any other change that belongs with it, and asks for a frame. */
export function publish(extra: Partial<AppState> = {}): void {
  app.set({ pos: clock.pos, ...extra });
  clock.request?.();
}

export function togglePlay(): void {
  const s = app.get();
  const demo = s.demo;
  if (!demo) return;
  if (!s.playing && clock.pos >= demo.total - 1 && !s.loop) clock.pos = 0;
  publish({ playing: !s.playing });
}

export function stepBy(by: number): void {
  const demo = app.get().demo;
  if (!demo) return;
  clock.pos = stepPosition(demo, clock.pos, by);
  publish({ playing: false });
}

export function jumpSegment(dir: number): void {
  const demo = app.get().demo;
  if (!demo) return;
  const seg = clamp(demo.locate(clock.pos).seg + dir, 0, demo.segments.length - 1);
  clock.pos = demo.segments[seg].start;
  publish();
}

/** Moves to a fraction (0..1) of the whole recording. */
export function seekFraction(fraction: number): void {
  const demo = app.get().demo;
  if (!demo) return;
  clock.pos = clamp(fraction, 0, 1) * Math.max(0, demo.total - 1e-6);
  publish();
}

/** Moves to the first token where free generation left the reference, if there is one. */
export function seekDivergence(): void {
  const demo = app.get().demo;
  if (!demo) return;
  const seg = demo.locate(clock.pos).seg;
  const sg = demo.segments[seg];
  if (sg.diverge >= sg.t0 && sg.diverge < sg.ftok.length) {
    clock.pos = demo.positionOfToken(seg, sg.diverge);
    publish({ playing: false });
  }
}

export function jumpTo(pos: number, extra: Partial<AppState> = {}): void {
  clock.pos = pos;
  publish(extra);
}
