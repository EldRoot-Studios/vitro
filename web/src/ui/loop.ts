/**
 * A redraw loop that runs only while something moves. `draw(now, dt)` paints one frame and returns true when
 * another frame is needed (an animation is running); `request()` asks for a frame after a change of state.
 * Frames stop by themselves in a hidden tab, because the browser stops animation frames there.
 */

import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';

/** Paints one frame; `dt` is the time since the previous frame in seconds (0 for the first, at most 0.1). */
export type DrawFn = (now: number, dt: number) => boolean | void;

export interface Loop {
  request(): void;
  destroy(): void;
}

export function createLoop(draw: DrawFn): Loop {
  let raf = 0;
  let last: number | null = null;
  let dead = false;

  const tick = (now: number): void => {
    raf = 0;
    if (dead) return;
    const dt = last === null ? 0 : Math.min(0.1, (now - last) / 1000);
    last = now;
    if (draw(now, dt)) schedule();
    else last = null;
  };
  function schedule(): void {
    if (!raf && !dead) raf = requestAnimationFrame(tick);
  }
  return {
    request: schedule,
    destroy() {
      dead = true;
      cancelAnimationFrame(raf);
    },
  };
}

/**
 * Runs `draw` in a loop that lives as long as the component. Returns a stable `request` function; the first
 * frame is requested when the component mounts. `draw` may close over the latest props and state: the loop
 * always calls the version from the last render.
 */
export function useLoop(draw: DrawFn): () => void {
  const drawRef = useRef(draw);
  useLayoutEffect(() => {
    drawRef.current = draw;
  });
  const loopRef = useRef<Loop | null>(null);
  useEffect(() => {
    const loop = createLoop((now, dt) => drawRef.current(now, dt));
    loopRef.current = loop;
    loop.request();
    return () => {
      loop.destroy();
      loopRef.current = null;
    };
  }, []);
  return useCallback(() => loopRef.current?.request(), []);
}

/** True when the person asked the system for less motion. */
export function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
