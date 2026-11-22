/**
 * Activations: the L2 norm of the hidden state per layer, component and token as bands of colour, and - when the
 * trace carries them - 2D latent maps with the cosine similarity between their rows.
 * Port of cpp/src/view_norms.cpp.
 */

import { useEffect, useRef, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from 'react';

import { latentCos, maxAbs, tokenText } from '../core/analysis.ts';
import { ACCENT, COMPONENT_COLORS, css, diverging, hex, mix, similarity } from '../core/colors.ts';
import type { NdArray } from '../core/container.ts';
import type { NormBlock, UnitTrace } from '../core/trace.ts';
import { selectTokenPatch, selectUnitPatch } from '../state/actions.ts';
import { app, useApp } from '../state/app.ts';
import { FONT, colorPixels, ellipsize, fitCanvas, heatPixels, onFontsLoaded, paintPixels } from '../ui/canvas.ts';
import { clamp } from '../ui/format.ts';
import { useSnapshotSource, useViewKeys } from '../ui/keys.ts';
import { freeRect } from '../ui/layout.ts';
import { useLoop } from '../ui/loop.ts';
import { Facts, Heading, Seg } from '../ui/parts.tsx';
import { tooltip } from '../ui/tooltip.ts';
import { compact, useResizeRedraw, useUnit } from './hooks.ts';

const INK = 'rgba(250,249,245,0.86)';
const INK3 = 'rgba(250,249,245,0.5)';
const INK4 = 'rgba(250,249,245,0.32)';
const EDGE = 'rgba(255,255,255,0.14)';
const BAND_BG = hex('#1d1d1b');
const TITLE_H = 26;

const maxCache = new WeakMap<NdArray, number>();

/** Largest absolute value of an array, never below `floor`; computed once per array. */
function largest(nd: NdArray, floor: number): number {
  let v = maxCache.get(nd);
  if (v === undefined) {
    v = maxAbs(nd.data, floor);
    maxCache.set(nd, v);
  }
  return v;
}

interface NormHits {
  part: 'norm';
  x: number;
  y: number;
  left: number;
  top: number;
  availW: number;
  rowH: number;
  subH: number;
  L: number;
  C: number;
  R: number;
  nb: NormBlock;
  d: ArrayLike<number>;
}

interface LatentHits {
  part: 'latent';
  left: number;
  top: number;
  mapW: number;
  mapH: number;
  cosX: number;
  cosSize: number;
  S: number;
  D: number;
  data: ArrayLike<number>;
  cos: Float32Array;
  rowLabel: string;
  colLabel: string;
}

type Hover = { part: 'norm'; t: number; l: number; c: number } | { part: 'map' | 'cos'; r: number; c: number } | null;

export function ActivationsStage() {
  const { unit } = useUnit();
  if (!unit.norms.data && unit.latents.length === 0) {
    return (
      <div className="empty glass">
        <h2>No activations</h2>
        <p>No activation norms or latent maps were recorded for this unit.</p>
      </div>
    );
  }
  return <ActivationsCanvas />;
}

function ActivationsCanvas() {
  const { dump, unit } = useUnit();
  const token = useApp((s) => s.token);
  const latent = useApp((s) => s.latent);
  const latentRow = useApp((s) => s.latentRow);
  const canvas = useRef<HTMLCanvasElement>(null);
  const hover = useRef<Hover>(null);
  const hitNorm = useRef<NormHits | null>(null);
  const hitLat = useRef<LatentHits | null>(null);

  const request = useLoop(() => {
    const c = canvas.current;
    if (!c) return false;
    const { ctx, w, h } = fitCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const free = freeRect(c);
    const hasNorms = unit.norms.data !== null;
    const hasLat = unit.latents.length > 0;
    hitNorm.current = null;
    hitLat.current = null;
    // both parts share the free area: the norms above, the latent maps below
    const split = hasNorms && hasLat ? Math.round(free.h * 0.54) : free.h;
    if (hasNorms) drawNorms(ctx, { x: free.x, y: free.y, w: free.w, h: split });
    if (hasLat) drawLatent(ctx, hasNorms ? { x: free.x, y: free.y + split, w: free.w, h: free.h - split } : free);
    return false;
  });

  function drawNorms(ctx: CanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }): void {
    const nb = unit.norms;
    if (!nb.data) return;
    const [L, C, R] = nb.data.shape;
    const d = nb.data.float32();
    const vmax = largest(nb.data, 1e-6);
    const names = R <= 48;
    const left = 96;
    const top = TITLE_H + (names ? 76 : 20);
    const bottom = C > 1 ? 34 : 14;
    const availW = r.w - left - 24;
    const availH = r.h - top - bottom;
    const rowH = clamp(availH / L, 5, 44);
    const subH = Math.max(2, (rowH - 4) / C);
    const hv = hover.current;
    hitNorm.current = { part: 'norm', x: r.x, y: r.y, left, top, availW, rowH, subH, L, C, R, nb, d };
    ctx.save();
    ctx.translate(r.x, r.y);
    ctx.font = FONT.sans(12.5);
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK3;
    ctx.fillText(`${nb.name} — ${L} layers × ${C} ${C === 1 ? 'component' : 'components'} × ${R} tokens`, 8, 16);
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.font = FONT.sans(11);
    for (let l = 0; l < L; l++) {
      const y = top + l * rowH;
      if (rowH >= 12) {
        ctx.fillStyle = hv && hv.part === 'norm' && hv.l === l ? INK : INK3;
        ctx.fillText(ellipsize(ctx, nb.layers[l] ?? `L${l + 1}`, left - 14), left - 10, y + rowH / 2);
      }
      for (let c = 0; c < C; c++) {
        const slice = d.subarray((l * C + c) * R, (l * C + c + 1) * R);
        const px = C === 1 ? heatPixels(slice, vmax, 0.5) : colorPixels(slice, (v) => mix(BAND_BG, COMPONENT_COLORS[c % COMPONENT_COLORS.length], Math.sqrt(clamp(v / vmax, 0, 1))));
        paintPixels(ctx, px, 1, R, left, y + c * subH, availW, Math.max(1, Math.floor(subH) - 1));
      }
    }
    ctx.strokeStyle = EDGE;
    ctx.lineWidth = 1;
    ctx.strokeRect(left - 0.5, top - 0.5, availW + 1, L * rowH + 1);
    const colW = availW / R;
    if (names && colW >= 11) {
      ctx.font = FONT.mono(11);
      ctx.textAlign = 'left';
      for (let t = 0; t < R; t++) {
        const j = nb.rowOffset + t;
        ctx.save();
        ctx.translate(left + (t + 0.5) * colW, top - 8);
        ctx.rotate(-Math.PI * 0.28);
        ctx.fillStyle = j === token || (hv && hv.part === 'norm' && hv.t === t) ? INK : INK3;
        ctx.fillText(ellipsize(ctx, compact(tokenText(dump, unit, j)), 90), 0, 0);
        ctx.restore();
      }
    } else {
      ctx.font = FONT.mono(10.5);
      ctx.textAlign = 'left';
      ctx.fillStyle = INK4;
      const tick = R <= 32 ? 4 : R <= 120 ? 10 : 25;
      ctx.textBaseline = 'alphabetic';
      for (let t = 0; t < R; t += tick) ctx.fillText(String(nb.rowOffset + t), left + t * colW, top - 8);
    }
    const sel = token - nb.rowOffset;
    if (sel >= 0 && sel < R) {
      ctx.strokeStyle = css(ACCENT);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(left + sel * colW - 0.5, top - 3.5, Math.max(1, colW) + 1, L * rowH + 7);
      ctx.lineWidth = 1;
    }
    if (hv && hv.part === 'norm') {
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(left + hv.t * colW + 0.5, top + hv.l * rowH + hv.c * subH + 0.5, Math.max(1, colW - 1), Math.max(1, Math.floor(subH) - 1));
      ctx.lineWidth = 1;
    }
    if (C > 1) {
      ctx.font = FONT.sans(12);
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      let lx = left;
      for (let c = 0; c < C; c++) {
        ctx.fillStyle = css(COMPONENT_COLORS[c % COMPONENT_COLORS.length]);
        ctx.fillRect(lx, r.h - 22, 11, 11);
        ctx.fillStyle = INK3;
        const name = nb.components[c] ?? `component ${c}`;
        ctx.fillText(name, lx + 17, r.h - 16);
        lx += 17 + ctx.measureText(name).width + 22;
      }
    }
    ctx.restore();
  }

  function drawLatent(ctx: CanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }): void {
    const lm = unit.latents[latent];
    const [S, D] = lm.data.shape;
    const data = lm.data.float32();
    const vmax = largest(lm.data, 1e-6);
    const left = 60;
    const top = 44;
    const mapW = Math.max(40, Math.round((r.w - 30) * 0.58 - left));
    const mapH = Math.max(40, r.h - top - 18);
    const cosX = left + mapW + 40;
    const cosSize = Math.max(40, Math.min(r.w - cosX - 24, mapH));
    const cos = latentCos(unit, latent);
    const hv = hover.current;
    hitLat.current = { part: 'latent', left: left + r.x, top: top + r.y, mapW, mapH, cosX: cosX + r.x, cosSize, S, D, data, cos, rowLabel: lm.rowLabel, colLabel: lm.colLabel };
    ctx.save();
    ctx.translate(r.x, r.y);
    paintPixels(ctx, colorPixels(Float32Array.from(data, (v) => v / vmax), diverging), S, D, left, top, mapW, mapH);
    paintPixels(ctx, colorPixels(cos, similarity), S, S, cosX, top, cosSize, cosSize);
    ctx.strokeStyle = EDGE;
    ctx.lineWidth = 1;
    ctx.strokeRect(left - 0.5, top - 0.5, mapW + 1, mapH + 1);
    ctx.strokeRect(cosX - 0.5, top - 0.5, cosSize + 1, cosSize + 1);
    ctx.font = FONT.sans(12);
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK3;
    ctx.fillText(`${lm.name}: ${lm.colLabel} (${S} × ${D}, max |x| ${vmax.toFixed(2)})`, left, top - 14);
    ctx.fillText(`${lm.rowLabel} cosine similarity`, cosX, top - 14);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = FONT.mono(10.5);
    const every = Math.max(1, Math.ceil(14 / (mapH / S)));
    for (let row = 0; row < S; row += every) {
      ctx.fillStyle = INK4;
      ctx.fillText(String(row), left - 8, top + ((row + 0.5) * mapH) / S);
    }
    if (latentRow >= 0 && latentRow < S) {
      ctx.strokeStyle = css(ACCENT);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(left - 0.5, top + (latentRow * mapH) / S, mapW + 1, Math.max(2, mapH / S));
      ctx.strokeRect(cosX - 0.5, top + (latentRow * cosSize) / S, cosSize + 1, Math.max(2, cosSize / S));
      ctx.lineWidth = 1;
    }
    if (hv && (hv.part === 'map' || hv.part === 'cos')) {
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      const [x, y, cw, ch] = hv.part === 'map' ? [left, top, mapW / D, mapH / S] : [cosX, top, cosSize / S, cosSize / S];
      ctx.strokeRect(x + hv.c * cw + 0.5, y + hv.r * ch + 0.5, Math.max(1, cw - 1), Math.max(1, ch - 1));
    }
    ctx.restore();
  }

  useEffect(() => request(), [request, dump, unit, token, latent, latentRow]);
  useEffect(() => onFontsLoaded(request), [request]);
  useResizeRedraw(canvas, request);
  useSnapshotSource(() => canvas.current);

  useViewKeys((e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const s = app.get();
    switch (e.key) {
      case 'ArrowLeft':
      case 'ArrowRight': {
        const at = s.token < 0 ? unit.t0 : s.token + (e.key === 'ArrowLeft' ? -1 : 1);
        app.set(selectTokenPatch(s, clamp(at, unit.t0, unit.nTokens - 1)));
        return true;
      }
      case '[':
        app.set(selectUnitPatch(s, Math.max(0, s.unit - 1)));
        return true;
      case ']':
        app.set(selectUnitPatch(s, Math.min(dump.units.length - 1, s.unit + 1)));
        return true;
      default:
        return false;
    }
  });

  const local = (e: { clientX: number; clientY: number }): { x: number; y: number } => {
    const box = canvas.current!.getBoundingClientRect();
    return { x: e.clientX - box.left, y: e.clientY - box.top };
  };

  /** Which cell of which part is under the pointer. */
  function cellAt(e: { clientX: number; clientY: number }): Hover {
    const { x, y } = local(e);
    const g = hitNorm.current;
    if (g) {
      const lx = x - g.x - g.left;
      const ly = y - g.y - g.top;
      const t = Math.floor((lx * g.R) / g.availW);
      const l = Math.floor(ly / g.rowH);
      const c = Math.floor((ly % g.rowH) / g.subH);
      if (lx >= 0 && ly >= 0 && t >= 0 && t < g.R && l >= 0 && l < g.L && c < g.C) return { part: 'norm', t, l, c };
    }
    const m = hitLat.current;
    if (m) {
      if (x >= m.left && x < m.left + m.mapW && y >= m.top && y < m.top + m.mapH) {
        return { part: 'map', r: Math.floor(((y - m.top) * m.S) / m.mapH), c: Math.floor(((x - m.left) * m.D) / m.mapW) };
      }
      if (x >= m.cosX && x < m.cosX + m.cosSize && y >= m.top && y < m.top + m.cosSize) {
        return { part: 'cos', r: Math.floor(((y - m.top) * m.S) / m.cosSize), c: Math.floor(((x - m.cosX) * m.S) / m.cosSize) };
      }
    }
    return null;
  }

  const onMove = (e: ReactPointerEvent): void => {
    const cell = cellAt(e);
    hover.current = cell;
    request();
    if (!cell) return tooltip.hide();
    if (cell.part === 'norm') {
      const g = hitNorm.current!;
      const j = g.nb.rowOffset + cell.t;
      tooltip.show(`${g.nb.layers[cell.l] ?? `layer ${cell.l + 1}`} · ${g.nb.components[cell.c] ?? `component ${cell.c}`}\ntoken ${j} "${tokenText(dump, unit, j).trim() || '·'}"\nnorm ${g.d[(cell.l * g.C + cell.c) * g.R + cell.t].toFixed(3)}`, e.clientX, e.clientY);
      return;
    }
    const g = hitLat.current!;
    const text = cell.part === 'map' ? `${g.rowLabel} ${cell.r}, ${g.colLabel} ${cell.c}: ${g.data[cell.r * g.D + cell.c].toFixed(4)}` : `cosine(${g.rowLabel} ${cell.r}, ${g.rowLabel} ${cell.c}) = ${g.cos[cell.r * g.S + cell.c].toFixed(3)}`;
    tooltip.show(text, e.clientX, e.clientY);
  };
  const onClick = (e: ReactMouseEvent): void => {
    const cell = cellAt(e);
    if (!cell) return;
    if (cell.part === 'norm') app.set(selectTokenPatch(app.get(), hitNorm.current!.nb.rowOffset + cell.t));
    else app.set({ latentRow: cell.r });
  };

  return (
    <canvas
      ref={canvas}
      className="scene"
      role="img"
      aria-label="Activation norms per layer and token, and latent maps"
      style={{ cursor: 'crosshair' }}
      onPointerMove={onMove}
      onPointerLeave={() => {
        hover.current = null;
        tooltip.hide();
        request();
      }}
      onClick={onClick}
    />
  );
}

/** Norm of every layer for the selected token, one line per component. */
function Profile({ unit, token }: { unit: UnitTrace; token: number }) {
  const nb = unit.norms;
  if (!nb.data) return null;
  const [L, C, R] = nb.data.shape;
  const t = token - nb.rowOffset;
  if (t < 0 || t >= R) return null;
  const d = nb.data.float32();
  const vmax = largest(nb.data, 1e-6);
  const W = 280;
  const H = 84;
  return (
    <svg className="profile" viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Norm by layer for the selected token">
      <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="10" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" />
      {Array.from({ length: C }, (_, c) => (
        <polyline
          key={c}
          fill="none"
          stroke={css(COMPONENT_COLORS[c % COMPONENT_COLORS.length])}
          strokeWidth="1.5"
          strokeLinejoin="round"
          points={Array.from({ length: L }, (_, l) => `${((L === 1 ? 0.5 : l / (L - 1)) * (W - 8) + 4).toFixed(1)},${(H - 6 - (d[(l * C + c) * R + t] / vmax) * (H - 14)).toFixed(1)}`).join(' ')}
        />
      ))}
    </svg>
  );
}

export function ActivationsInspector() {
  const { unit } = useUnit();
  const token = useApp((s) => s.token);
  const latent = useApp((s) => s.latent);
  const latentRow = useApp((s) => s.latentRow);
  const nb = unit.norms;
  return (
    <>
      <div className="title">Activations</div>
      <div className="sub">{nb.data ? nb.name : 'latent maps'}</div>
      {nb.data && (
        <>
          <Facts
            rows={[
              ['Layers', String(nb.data.shape[0])],
              ['Components', nb.components.join(', ') || String(nb.data.shape[1])],
              ['Tokens', String(nb.data.shape[2])],
              ['Largest norm', largest(nb.data, 1e-6).toFixed(2)],
            ]}
          />
          {token - nb.rowOffset >= 0 && token - nb.rowOffset < nb.data.shape[2] ? (
            <>
              <Heading>Token {token} across layers</Heading>
              <Profile unit={unit} token={token} />
              <p className="note">Lines show the norm at each layer, first to last, scaled to the largest norm of the trace.</p>
            </>
          ) : (
            <p className="note">Click a column of the norm map, or pick a token in Tokens, to see its norm across layers.</p>
          )}
        </>
      )}
      {unit.latents.length > 0 && (
        <>
          <Heading>Latent map</Heading>
          {unit.latents.length > 1 && (
            <div className="block">
              <Seg label="Latent map" full value={latent} onChange={(i) => app.set({ latent: i, latentRow: -1 })} options={unit.latents.map((lm, i) => ({ value: i, label: lm.name }))} />
            </div>
          )}
          <Facts
            rows={[
              ['Map', unit.latents[latent].name],
              ['Shape', `${unit.latents[latent].data.shape[0]} × ${unit.latents[latent].data.shape[1]}`],
              ['Selected row', latentRow < 0 ? 'none' : String(latentRow)],
            ]}
          />
        </>
      )}
    </>
  );
}
