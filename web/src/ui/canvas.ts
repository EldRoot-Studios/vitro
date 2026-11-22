/**
 * Canvas helpers: high-DPI sizing, additive "light" drawing (glow sprites and lines) for the 3D scenes, and
 * matrix painting for heat maps.
 *
 * The 3D scenes draw with the `lighter` composite mode, so overlapping lines and glows add up like light on a
 * black stage; the callers set that mode once per pass and restore it afterwards.
 */

import { css, heat } from '../core/colors.ts';

type Colour = readonly number[];

export interface Fitted {
  ctx: CanvasRenderingContext2D;
  /** Size of the canvas in CSS pixels. */
  w: number;
  h: number;
  dpr: number;
}

/**
 * Sizes the canvas backing store to its CSS box at the device pixel ratio (capped at 2) and returns a context
 * whose units are CSS pixels.
 */
export function fitCanvas(canvas: HTMLCanvasElement): Fitted {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth));
  const h = Math.max(1, Math.round(canvas.clientHeight));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('this browser cannot draw on a canvas');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h, dpr };
}

const sprites = new Map<number, HTMLCanvasElement>();

/** A soft round light of one colour: a radial falloff that is brightest at the centre, cached per colour. */
function glowSprite(color: Colour): HTMLCanvasElement {
  const key = ((color[0] >> 3) << 10) | ((color[1] >> 3) << 5) | (color[2] >> 3);
  let s = sprites.get(key);
  if (!s) {
    s = document.createElement('canvas');
    s.width = s.height = 64;
    const g = s.getContext('2d');
    if (!g) throw new Error('this browser cannot draw on a canvas');
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    for (const [at, a] of [[0, 1], [0.25, 0.8], [0.5, 0.3], [0.75, 0.07], [1, 0]]) grad.addColorStop(at, css(color, a));
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
    sprites.set(key, s);
  }
  return s;
}

/** Draws a glow of `radius` CSS pixels centred at (x, y); the caller has set the composite mode. */
export function glow(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number, color: Colour, alpha: number): void {
  if (alpha <= 0.003 || radius <= 0.2) return;
  ctx.globalAlpha = Math.min(1, alpha);
  ctx.drawImage(glowSprite(color), x - radius, y - radius, radius * 2, radius * 2);
  ctx.globalAlpha = 1;
}

/** A single bright point, the core of a node. */
export function spark(ctx: CanvasRenderingContext2D, x: number, y: number, color: Colour, alpha: number): void {
  if (alpha <= 0.003) return;
  ctx.fillStyle = css(color, Math.min(1, alpha));
  ctx.fillRect(Math.round(x) - 0.5, Math.round(y) - 0.5, 1.4, 1.4);
}

const styles = new Map<number, string>();

function lineStyle(color: Colour, alpha: number): string {
  const q = Math.round(Math.min(1, alpha) * 127);
  const key = (((color[0] | 0) << 16) | ((color[1] | 0) << 8) | (color[2] | 0)) * 128 + q;
  let s = styles.get(key);
  if (s === undefined) {
    if (styles.size > 6000) styles.clear();
    s = css(color, q / 127);
    styles.set(key, s);
  }
  return s;
}

/** Draws one hairline of light from (x0, y0) to (x1, y1). */
export function addLine(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: Colour, alpha: number): void {
  if (alpha <= 0.004) return;
  ctx.strokeStyle = lineStyle(color, alpha);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

/** Heat colours at 256 levels, as RGBA bytes: index = round(255 * t). */
const HEAT_LUT: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const c = heat(i / 255);
    lut.set([c[0], c[1], c[2], 255], i * 4);
  }
  return lut;
})();

/** Colour of a cell with no value (NaN): a hair above the background so the matrix keeps its shape. */
const EMPTY_CELL: readonly number[] = [29, 29, 27];

/**
 * RGBA pixels for a matrix on the heat ramp: value / vmax, raised to `gamma` to lift weak values.
 * NaN cells are painted as empty.
 * @param values rows * cols
 * @param vmax value that maps to the top of the ramp
 */
export function heatPixels(values: ArrayLike<number>, vmax: number, gamma = 0.55): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v !== v) {
      out.set([...EMPTY_CELL, 255], i * 4);
      continue;
    }
    const t = Math.pow(Math.min(Math.max(v / vmax, 0), 1), gamma);
    const k = ((t * 255 + 0.5) | 0) * 4;
    out[i * 4] = HEAT_LUT[k];
    out[i * 4 + 1] = HEAT_LUT[k + 1];
    out[i * 4 + 2] = HEAT_LUT[k + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** RGBA pixels for a matrix coloured by `colorOf(value)`; NaN cells are painted as empty. */
export function colorPixels(values: ArrayLike<number>, colorOf: (v: number) => Colour): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    const c = values[i] !== values[i] ? EMPTY_CELL : colorOf(values[i]);
    out[i * 4] = c[0];
    out[i * 4 + 1] = c[1];
    out[i * 4 + 2] = c[2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

let scratch: HTMLCanvasElement | null = null;

/**
 * Paints a rows x cols RGBA image into the rectangle (x, y, w, h): crisp cells when enlarged, smoothed when
 * shrunk below one pixel per cell.
 */
export function paintPixels(ctx: CanvasRenderingContext2D, pixels: Uint8ClampedArray<ArrayBuffer>, rows: number, cols: number, x: number, y: number, w: number, h: number): void {
  if (rows <= 0 || cols <= 0 || w <= 0 || h <= 0) return;
  scratch ??= document.createElement('canvas');
  scratch.width = cols;
  scratch.height = rows;
  const g = scratch.getContext('2d');
  if (!g) throw new Error('this browser cannot draw on a canvas');
  g.putImageData(new ImageData(pixels, cols, rows), 0, 0);
  ctx.imageSmoothingEnabled = w < cols || h < rows;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(scratch, x, y, w, h);
}

/** Text no wider than `maxWidth`, cut with an ellipsis; measures with the context's current font. */
export function ellipsize(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return lo > 0 ? `${text.slice(0, lo)}…` : '';
}

/** Fonts used on canvases; the faces are declared in styles/tokens.css. */
export const FONT = {
  sans: (px: number, weight = 400): string => `${weight} ${px}px Geist, ui-sans-serif, system-ui, "Segoe UI", sans-serif`,
  mono: (px: number, weight = 400): string => `${weight} ${px}px "Geist Mono", ui-monospace, "Cascadia Mono", Consolas, monospace`,
};

/**
 * Canvases are drawn before a web font has arrived, in the fallback font. This calls `redraw` once the faces the
 * canvases use are loaded, so the first picture is replaced by the right one; it returns a function that stops it.
 */
export function onFontsLoaded(redraw: () => void): () => void {
  let live = true;
  void Promise.all([document.fonts.load(FONT.sans(13)), document.fonts.load(FONT.mono(12))]).then(() => {
    if (live) redraw();
  });
  return () => {
    live = false;
  };
}
