/**
 * Colour ramps and named tints.
 *
 * Colours are [r, g, b] triples with channels 0..255. The interface has no colour of its own: it is black and
 * white glass. Colour lives only in the data, and even there it is quiet: one grey ramp ("heat") carries every
 * "how much" quantity (attention weight, surprisal, activation), muted status colours say right or wrong, and a
 * single cool grey-blue is the "other side" (negative values, cross-attention).
 */

export type RGB = [number, number, number];
type Colour = readonly number[];

/** Parses "#rrggbb" into [r, g, b]. */
export function hex(s: string): RGB {
  if (!/^#[0-9a-fA-F]{6}$/.test(s)) throw new RangeError(`not a #rrggbb colour: ${JSON.stringify(s)}`);
  return [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
}

/** Linear blend of two colours; t is clamped to 0..1. */
export function mix(a: Colour, b: Colour, t: number): RGB {
  const k = Math.min(Math.max(t, 0), 1);
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

/** CSS colour string for `c` with opacity `a`. */
export function css(c: Colour, a = 1): string {
  return `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
}

/** Piecewise-linear ramp over increasing stop positions; NaN and values below the first stop give the first colour. */
export function ramp(stops: readonly (readonly [number, Colour])[], t: number): RGB {
  if (Number.isNaN(t) || t <= stops[0][0]) return [...stops[0][1]] as RGB;
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      return mix(c0, c1, (t - t0) / (t1 - t0));
    }
  }
  return [...stops[stops.length - 1][1]] as RGB;
}

export const WHITE: RGB = [255, 255, 255];
export const INK: RGB = hex('#f4f4f5');
export const OK: RGB = hex('#7fcf9f');
export const ERR: RGB = hex('#e87b7b');
export const WARN: RGB = hex('#d9b36c');
/** The one cool tint: negative values, the other side of a comparison, cross-attention. */
export const STEEL: RGB = hex('#a2b8d4');

const HEAT: readonly RGB[] = ['#0b0c0e', '#2a2c31', '#6d7078', '#b9bcc4', '#ffffff'].map(hex);
const HEAT_STOPS: readonly (readonly [number, RGB])[] = HEAT.map((c, i) => [i / (HEAT.length - 1), c]);

/** The one "how much" ramp, black to white in grey; t in 0..1. Luminance rises with t. */
export function heat(t: number): RGB {
  return ramp(HEAT_STOPS, t);
}

/** CSS gradient of the heat ramp, for the legend of a data scale. */
export function heatGradient(n = 10): string {
  return `linear-gradient(90deg, ${Array.from({ length: n }, (_, i) => css(heat(i / (n - 1)))).join(', ')})`;
}

/** Token surprisal in bits (-log2 p): near-black for the expected, bright for the surprising. */
export function surprisal(bits: number): RGB {
  return heat(Math.pow(Math.min(Math.max(bits, 0) / 10, 1), 0.6));
}

const DELTA_STOPS: readonly (readonly [number, RGB])[] = [
  [0, hex('#275a52')],
  [1, hex('#3f7d6a')],
  [3, hex('#93ad4f')],
  [11, hex('#d9a63e')],
  [12, hex('#d8503a')],
  [200, hex('#ff3a5c')],
];

/**
 * Distance between a predicted and a true numeric value, in steps; `redAt` is where the crimson ceiling
 * starts (12 by default) and scales the whole ramp.
 */
export function deltaError(steps: number, redAt = 12): RGB {
  const k = Math.max(1, redAt) / 12;
  return ramp(
    DELTA_STOPS.map(([at, c]) => [at * k, c] as const),
    steps,
  );
}

const NEUTRAL: RGB = hex('#15161a');

/** Signed values in -1..1: steel below zero, near-black at zero, white above. */
export function diverging(t: number): RGB {
  const v = Math.min(Math.max(t, -1), 1);
  return v < 0 ? mix(NEUTRAL, STEEL, -v) : mix(NEUTRAL, INK, v);
}

/** Cosine similarity in -1..1: the heat ramp for aligned vectors, steel for opposite ones. */
export function similarity(c: number): RGB {
  const v = Math.min(Math.max(c, -1), 1);
  return v >= 0 ? heat(v) : mix(hex('#0b0c0e'), STEEL, -v);
}

/** Readable text colour for a given background. */
export function textOn(bg: Colour): RGB {
  const lum = (bg[0] * 299 + bg[1] * 587 + bg[2] * 114) / 1000;
  return lum > 140 ? hex('#0b0c0e') : INK;
}

/** Relative luminance 0..1 (Rec. 709 weights), used to check that a ramp is monotone. */
export function luminance(c: Colour): number {
  return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
}

/** Colours the trace author can name in a SpanLabel or LinkSpec tint, softened to sit on black glass. */
export const TINTS: Record<string, RGB> = {
  violet: hex('#a79fe0'),
  gold: hex('#d8c07e'),
  magenta: hex('#c897d6'),
  blue: hex('#8fb2e0'),
  cyan: hex('#8fd0d8'),
  emerald: hex('#86cba5'),
  coral: hex('#e19a88'),
};

/** Used for a span or link that names no tint. */
export const PLAIN_TINT: RGB = hex('#c9c9cf');

/** Resolves a tint name or "#rrggbb"; anything else gives `fallback`. */
export function parseTint(s: string, fallback: Colour): RGB {
  if (!s) return [...fallback] as RGB;
  if (/^#[0-9a-fA-F]{6}$/.test(s)) return hex(s);
  return Object.hasOwn(TINTS, s) ? TINTS[s] : ([...fallback] as RGB);
}

/** Component colours of the activation-norm bands (a single component uses the heat ramp instead). */
export const COMPONENT_COLORS: readonly RGB[] = [INK, STEEL, hex('#dccaa6'), hex('#d6a0aa')];
