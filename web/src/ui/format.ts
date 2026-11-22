/** Number and size formatting shared by the panels. */

export const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

/** 1536 -> "1.5 KB"; sizes below 1 KB stay in bytes. */
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(n < 10 * 1024 * 1024 ? 2 : 1)} MB`;
}

/** Probability with enough digits to tell 0.0004 from 0.04: three decimals above 0.001, else exponent. */
export function formatProb(p: number): string {
  if (!(p > 0)) return '0';
  return p >= 0.001 ? p.toFixed(3) : p.toExponential(1);
}

/** Integer with thin grouping: 151936 -> "151 936". */
export function formatInt(n: number): string {
  return Math.round(n)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/** Makes whitespace in a token label visible enough to click: an all-space label becomes a middle dot. */
export function visibleLabel(label: string): string {
  return label.trim() === '' && label !== '' ? '·' : label;
}

/** A label on one line for a canvas: runs of whitespace become one space, an empty result a middle dot. */
export function oneLine(label: string): string {
  return label.replace(/\s+/g, ' ').trim() || '·';
}
