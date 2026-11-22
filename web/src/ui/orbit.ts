/**
 * Pieces both 3D scenes share: the orbit controls (drag to rotate, wheel to zoom, click to pick) and the
 * brackets with group names under the graph.
 */

import { useEffect, useLayoutEffect, useRef, type RefObject } from 'react';

import type { Camera3D } from '../core/math3d.ts';
import { PITCH_LIMIT } from '../core/math3d.ts';
import type { CameraPose } from '../state/state.ts';
import { FONT } from './canvas.ts';
import { clamp } from './format.ts';

/** Zoom limits of the scene cameras. */
export const ZOOM_MIN = 0.3;
export const ZOOM_MAX = 5;

export interface OrbitHandlers {
  /** The current camera. */
  get(): CameraPose;
  set(patch: Partial<CameraPose>): void;
  /** A press that did not move, in canvas pixels. */
  click?(x: number, y: number, e: PointerEvent): void;
  /** The pointer moving without a button. */
  hover?(x: number, y: number, e: PointerEvent): void;
  leave?(): void;
}

/**
 * Orbit controls on a canvas: drag turns the scene, the wheel zooms, a press without a move is a click. The
 * listeners are attached once; the handlers may change on every render and the latest ones are used.
 */
export function useOrbit(canvasRef: RefObject<HTMLCanvasElement | null>, handlers: OrbitHandlers): void {
  const ref = useRef(handlers);
  useLayoutEffect(() => {
    ref.current = handlers;
  });
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let down: { x: number; y: number; last: [number, number]; moved: boolean } | null = null;
    const local = (e: PointerEvent): [number, number] => {
      const box = canvas.getBoundingClientRect();
      return [e.clientX - box.left, e.clientY - box.top];
    };
    const onDown = (e: PointerEvent): void => {
      if (e.button !== 0) return;
      canvas.setPointerCapture(e.pointerId);
      down = { x: e.clientX, y: e.clientY, last: [e.clientX, e.clientY], moved: false };
    };
    const onMove = (e: PointerEvent): void => {
      const h = ref.current;
      if (!down) {
        h.hover?.(...local(e), e);
        return;
      }
      if (!down.moved && Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y) > 4) {
        down.moved = true;
        canvas.dataset.dragging = '';
      }
      if (down.moved) {
        const cam = h.get();
        h.set({ yaw: cam.yaw + (e.clientX - down.last[0]) * 0.008, pitch: clamp(cam.pitch + (e.clientY - down.last[1]) * 0.008, -PITCH_LIMIT, PITCH_LIMIT) });
      }
      down.last = [e.clientX, e.clientY];
    };
    const onUp = (e: PointerEvent): void => {
      if (!down) return;
      const wasClick = !down.moved;
      down = null;
      delete canvas.dataset.dragging;
      if (wasClick) ref.current.click?.(...local(e), e);
    };
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const h = ref.current;
      h.set({ zoom: clamp(h.get().zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1), ZOOM_MIN, ZOOM_MAX) });
    };
    const onLeave = (): void => {
      if (!down) ref.current.leave?.();
    };
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('pointerleave', onLeave);
    // not passive: the wheel must zoom the scene, not scroll the page
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      canvas.removeEventListener('pointerleave', onLeave);
      canvas.removeEventListener('wheel', onWheel);
      delete canvas.dataset.dragging;
    };
  }, [canvasRef]);
}

/** Places labels on up to `rows` lines so that neighbours do not overlap. */
export class StaggeredLabels {
  private readonly rows: [number, number][][];

  constructor(rows = 2) {
    this.rows = Array.from({ length: rows }, () => []);
  }

  /** Line index for a label centred at `cx` with width `w`, keeping `gap` pixels from its neighbours. */
  place(cx: number, w: number, gap: number): number {
    const left = cx - w / 2;
    const right = cx + w / 2;
    for (let k = 0; k < this.rows.length; k++) {
      const free = this.rows[k].every(([a, b]) => !(left - gap < b && right + gap > a));
      if (free || k + 1 === this.rows.length) {
        this.rows[k].push([left, right]);
        return k;
      }
    }
    return 0;
  }
}

/** Draws the bracket under each run of stages and its name; lit groups are drawn in white. */
export function drawBrackets(ctx: CanvasRenderingContext2D, cam: Camera3D, groups: readonly { a: number; b: number; label: string }[], stageX: Float32Array, lit: readonly boolean[]): void {
  const labels = new StaggeredLabels();
  const p = new Float32Array(3);
  const q = new Float32Array(3);
  ctx.font = FONT.sans(12);
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.lineWidth = 1;
  groups.forEach((g, i) => {
    cam.map(stageX[g.a], -162, 0, p);
    cam.map(stageX[g.b], -162, 0, q);
    ctx.strokeStyle = lit[i] ? 'rgba(255,255,255,0.9)' : 'rgba(244,244,245,0.26)';
    ctx.fillStyle = lit[i] ? '#ffffff' : 'rgba(244,244,245,0.42)';
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
    ctx.stroke();
    const tw = ctx.measureText(g.label).width;
    const cx = (p[0] + q[0]) / 2;
    const row = labels.place(cx, tw, 8);
    ctx.fillText(g.label, cx - tw / 2, (p[1] + q[1]) / 2 + 17 + row * 15);
  });
}
