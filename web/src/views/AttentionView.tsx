/**
 * Attention: one matrix of attention weights at a time - for a chosen family (encoder self, decoder self,
 * cross...), layer and head, or averaged over layers and heads - or all layers x heads on one page. Clicking a
 * row selects its token; the inspector lists the strongest keys of that row.
 * Port of cpp/src/view_attn.cpp.
 */

import { useEffect, useRef, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';

import { attnAxes, attnMap, maxOf } from '../core/analysis.ts';
import { ACCENT, css, heatGradient } from '../core/colors.ts';
import { selectTokenPatch, selectUnitPatch } from '../state/actions.ts';
import { app, useApp } from '../state/app.ts';
import { FONT, ellipsize, fitCanvas, heatPixels, onFontsLoaded, paintPixels } from '../ui/canvas.ts';
import { clamp } from '../ui/format.ts';
import { useSnapshotSource, useViewKeys } from '../ui/keys.ts';
import { freeRect } from '../ui/layout.ts';
import { useLoop } from '../ui/loop.ts';
import { Chips, Facts, Heading, Seg } from '../ui/parts.tsx';
import { tooltip } from '../ui/tooltip.ts';
import { compact, useResizeRedraw, useUnit, vars } from './hooks.ts';

const INK = 'rgba(250,249,245,0.86)';
const INK3 = 'rgba(250,249,245,0.5)';
const INK4 = 'rgba(250,249,245,0.32)';
const EDGE = 'rgba(255,255,255,0.14)';

const layerText = (layer: number): string => (layer < 0 ? 'layer mean' : `layer ${layer + 1}`);
const headText = (head: number): string => (head < 0 ? 'head mean' : `head ${head + 1}`);

/** Sizes of one map inside a box: square cells for squarish matrices, else as much as fits. */
function cellSize(availW: number, availH: number, R: number, C: number): { cw: number; ch: number } {
  let cw = clamp(availW / C, 1, 44);
  let ch = clamp(availH / R, 1, 44);
  if (Math.max(R, C) <= 128 && Math.max(R, C) / Math.min(R, C) <= 1.6) cw = ch = Math.min(cw, ch);
  return { cw: cw >= 2 ? Math.floor(cw) : cw, ch: ch >= 2 ? Math.floor(ch) : ch };
}

type Axes = ReturnType<typeof attnAxes>;

/** Geometry of the last drawing, in canvas pixels, for hit testing. */
type Hits =
  | { kind: 'single'; x0: number; y0: number; mapW: number; mapH: number; R: number; C: number; m: Float32Array; axes: Axes }
  | { kind: 'grid'; left: number; top: number; gap: number; cellW: number; cellH: number; layers: number; heads: number };

type Hover = { r: number; c: number } | { layer: number; head: number } | null;

/** The unit's attention matrices, or an empty state when none were recorded. */
export function AttentionStage() {
  const { unit } = useUnit();
  if (unit.attnFamilies.length === 0) {
    return (
      <div className="empty glass">
        <h2>No attention</h2>
        <p>No attention matrices were recorded for this unit.</p>
      </div>
    );
  }
  return <AttentionCanvas />;
}

function AttentionCanvas() {
  const { dump, unit } = useUnit();
  const fam = useApp((s) => s.fam);
  const layer = useApp((s) => s.layer);
  const head = useApp((s) => s.head);
  const grid = useApp((s) => s.grid);
  const token = useApp((s) => s.token);
  const canvas = useRef<HTMLCanvasElement>(null);
  const hover = useRef<Hover>(null);
  const hits = useRef<Hits | null>(null);
  const af = unit.attnFamilies[fam];

  const request = useLoop(() => {
    const c = canvas.current;
    if (!c) return false;
    const { ctx, w, h } = fitCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const free = freeRect(c);
    ctx.save();
    ctx.translate(free.x, free.y);
    if (grid) drawGrid(ctx, free.w, free.h, free.x, free.y);
    else drawSingle(ctx, free.w, free.h, free.x, free.y);
    ctx.restore();
    return false;
  });

  function drawSingle(ctx: CanvasRenderingContext2D, w: number, h: number, ox: number, oy: number): void {
    const axes = attnAxes(dump, unit, af);
    const R = af.rows;
    const C = af.cols;
    const m = attnMap(unit, fam, layer, head);
    const vmax = maxOf(m);
    const rowNames = R <= 64;
    const colNames = C <= 48;
    const left = rowNames ? 112 : 46;
    const top = colNames ? 92 : 36;
    const right = colNames ? 64 : 22;
    const { cw, ch } = cellSize(w - left - right, h - top - 22, R, C);
    const mapW = Math.round(cw * C);
    const mapH = Math.round(ch * R);
    const x0 = Math.round(left + Math.max(0, (w - left - right - mapW) / 2));
    const y0 = top;
    hits.current = { kind: 'single', x0: x0 + ox, y0: y0 + oy, mapW, mapH, R, C, m, axes };
    const hv = hover.current;

    paintPixels(ctx, heatPixels(m, vmax), R, C, x0, y0, mapW, mapH);
    ctx.strokeStyle = EDGE;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 - 0.5, y0 - 0.5, mapW + 1, mapH + 1);

    const selRow = token - af.rowOffset;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.font = FONT.mono(11);
    if (rowNames && ch >= 11) {
      for (let r = 0; r < R; r++) {
        ctx.fillStyle = r === selRow || (hv && 'r' in hv && hv.r === r) ? INK : INK3;
        ctx.fillText(ellipsize(ctx, compact(axes.row(r)), left - 18), x0 - 9, y0 + (r + 0.5) * ch);
      }
    } else {
      ctx.font = FONT.mono(10.5);
      const every = Math.max(1, Math.ceil(14 / ch));
      for (let r = 0; r < R; r += every) {
        ctx.fillStyle = INK4;
        ctx.fillText(String(af.rowOffset + r), x0 - 9, y0 + (r + 0.5) * ch);
      }
    }
    if (colNames && cw >= 11) {
      ctx.font = FONT.mono(11);
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      for (let c = 0; c < C; c++) {
        ctx.save();
        ctx.translate(x0 + (c + 0.5) * cw, y0 - 8);
        ctx.rotate(-Math.PI * 0.28);
        ctx.fillStyle = hv && 'c' in hv && hv.c === c ? INK : INK3;
        ctx.fillText(ellipsize(ctx, compact(axes.col(c)), 96), 0, 0);
        ctx.restore();
      }
    } else {
      ctx.font = FONT.mono(10.5);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      const tick = C <= 32 ? 4 : C <= 120 ? 10 : C <= 300 ? 25 : 100;
      for (let c = 0; c < C; c += tick) {
        ctx.fillStyle = INK4;
        ctx.fillText(String(c), x0 + c * cw, y0 - 10);
        ctx.fillRect(x0 + c * cw, y0 - 6, 1, 5);
      }
    }
    if (selRow >= 0 && selRow < R) {
      const y = y0 + selRow * ch;
      ctx.strokeStyle = css(ACCENT);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 - 1.5, y - 1.5, mapW + 3, ch + 3);
    }
    if (hv && 'r' in hv) {
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 + hv.c * cw + 0.5, y0 + hv.r * ch + 0.5, Math.max(1, cw - 1), Math.max(1, ch - 1));
      ctx.lineWidth = 1;
    }
  }

  function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number, ox: number, oy: number): void {
    const R = af.rows;
    const C = af.cols;
    const left = 62;
    const top = 40;
    const gap = 4;
    const cellW = (w - left - 16 - gap * af.heads) / (af.heads + 1);
    const cellH = (h - top - 16 - gap * af.layers) / (af.layers + 1);
    hits.current = { kind: 'grid', left: left + ox, top: top + oy, gap, cellW, cellH, layers: af.layers, heads: af.heads };
    const hv = hover.current;
    ctx.textBaseline = 'middle';
    ctx.font = FONT.mono(10.5);
    for (let l = 0; l <= af.layers; l++) {
      const ly = l === af.layers ? -1 : l;
      const y = top + l * (cellH + gap);
      ctx.textAlign = 'left';
      ctx.fillStyle = hv && 'layer' in hv && hv.layer === ly ? INK : INK3;
      ctx.fillText(ly < 0 ? 'mean' : `L${l + 1}`, 10, y + cellH / 2);
      for (let k = 0; k <= af.heads; k++) {
        const hy = k === af.heads ? -1 : k;
        const x = left + k * (cellW + gap);
        if (l === 0) {
          ctx.textAlign = 'center';
          ctx.fillStyle = hv && 'head' in hv && hv.head === hy ? INK : INK3;
          ctx.fillText(hy < 0 ? 'mean' : `H${k + 1}`, x + cellW / 2, top - 16);
        }
        const m = attnMap(unit, fam, ly, hy);
        paintPixels(ctx, heatPixels(m, maxOf(m)), R, C, x, y, cellW, cellH);
        const current = ly === layer && hy === head;
        const over = hv && 'layer' in hv && hv.layer === ly && hv.head === hy;
        ctx.strokeStyle = current ? css(ACCENT) : over ? 'rgba(255,255,255,0.7)' : EDGE;
        ctx.lineWidth = current ? 2 : 1;
        ctx.strokeRect(x - 0.5, y - 0.5, cellW + 1, cellH + 1);
      }
    }
    ctx.lineWidth = 1;
  }

  useEffect(() => request(), [request, dump, unit, fam, layer, head, grid, token]);
  useEffect(() => onFontsLoaded(request), [request]);
  useResizeRedraw(canvas, request);
  useSnapshotSource(() => canvas.current);

  // the wheel steps through the layers; it must not scroll the page, so it is not a passive listener
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const onWheel = (e: WheelEvent): void => {
      const s = app.get();
      if (s.grid) return;
      e.preventDefault();
      app.set({ layer: clamp(s.layer + (e.deltaY > 0 ? 1 : -1), -1, af.layers - 1) });
    };
    c.addEventListener('wheel', onWheel, { passive: false });
    return () => c.removeEventListener('wheel', onWheel);
  }, [af.layers]);

  useViewKeys((e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const s = app.get();
    switch (e.key) {
      case 'l':
      case 'L': {
        const L = af.layers;
        app.set({ layer: e.shiftKey ? (s.layer <= -1 ? L - 1 : s.layer - 1) : s.layer >= L - 1 ? -1 : s.layer + 1 });
        return true;
      }
      case 'h':
      case 'H': {
        const H = af.heads;
        app.set({ head: e.shiftKey ? (s.head <= -1 ? H - 1 : s.head - 1) : s.head >= H - 1 ? -1 : s.head + 1 });
        return true;
      }
      case 'f':
      case 'F':
        if (unit.attnFamilies.length < 2) return false;
        app.set({ fam: (s.fam + 1) % unit.attnFamilies.length, layer: -1, head: -1 });
        return true;
      case 'g':
      case 'G':
        app.set({ grid: !s.grid });
        return true;
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

  function cellAt(e: { clientX: number; clientY: number }): Hover {
    const g = hits.current;
    const c = canvas.current;
    if (!g || !c) return null;
    const box = c.getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    if (g.kind === 'single') {
      if (x < g.x0 || y < g.y0 || x >= g.x0 + g.mapW || y >= g.y0 + g.mapH) return null;
      return { r: Math.min(g.R - 1, Math.floor(((y - g.y0) * g.R) / g.mapH)), c: Math.min(g.C - 1, Math.floor(((x - g.x0) * g.C) / g.mapW)) };
    }
    const k = Math.floor((x - g.left) / (g.cellW + g.gap));
    const l = Math.floor((y - g.top) / (g.cellH + g.gap));
    if (x < g.left || y < g.top || k < 0 || l < 0 || k > g.heads || l > g.layers) return null;
    return { layer: l === g.layers ? -1 : l, head: k === g.heads ? -1 : k };
  }

  const onMove = (e: ReactPointerEvent): void => {
    const cell = cellAt(e);
    hover.current = cell;
    request();
    const g = hits.current;
    if (!cell || !g) return tooltip.hide();
    if ('layer' in cell) {
      tooltip.show(`${layerText(cell.layer)}, ${headText(cell.head)} - click to focus`, e.clientX, e.clientY);
      return;
    }
    if (g.kind !== 'single') return;
    const weight = g.m[cell.r * g.C + cell.c];
    tooltip.show(`${g.axes.row(cell.r).trim() || '·'}  (query ${af.rowOffset + cell.r})\nattends to ${g.axes.col(cell.c).trim() || '·'}  (key ${cell.c})\nweight ${weight.toFixed(4)}`, e.clientX, e.clientY);
  };
  const onClick = (e: ReactMouseEvent): void => {
    const cell = cellAt(e);
    if (!cell) return;
    if ('layer' in cell) app.set({ layer: cell.layer, head: cell.head, grid: false });
    else app.set(selectTokenPatch(app.get(), af.rowOffset + cell.r));
  };

  return (
    <canvas
      ref={canvas}
      className="scene"
      role="img"
      aria-label="Attention weights as a heat map"
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

export function AttentionInspector() {
  const { dump, unit } = useUnit();
  const fam = useApp((s) => s.fam);
  const layer = useApp((s) => s.layer);
  const head = useApp((s) => s.head);
  const grid = useApp((s) => s.grid);
  const token = useApp((s) => s.token);
  if (unit.attnFamilies.length === 0) {
    return (
      <>
        <div className="title">Attention</div>
        <p className="sub">This unit has no attention matrices.</p>
      </>
    );
  }
  const af = unit.attnFamilies[fam];
  const axes = attnAxes(dump, unit, af);
  const selRow = token - af.rowOffset;
  const m = attnMap(unit, fam, layer, head);
  const vmax = maxOf(m);
  let looks: ReactNode;
  if (selRow >= 0 && selRow < af.rows) {
    const base = selRow * af.cols;
    const order = Array.from({ length: af.cols }, (_, c) => c)
      .sort((a, b) => m[base + b] - m[base + a] || a - b)
      .slice(0, 12);
    const top = Math.max(1e-9, m[base + order[0]]);
    looks = (
      <>
        <Heading>
          {af.rowLabel} · position {token}
        </Heading>
        <p className="who">
          “<b>{compact(axes.row(selRow))}</b>” attends most to
        </p>
        <div className="cands">
          {order.map((c) => (
            <div className="cand" key={c}>
              <span className="n">{c}</span>
              <span className="t" title={axes.col(c)}>
                {compact(axes.col(c))}
              </span>
              <span className="meter">
                <i style={{ width: `${(m[base + c] / top) * 100}%` }} />
              </span>
              <span className="v">{m[base + c].toFixed(3)}</span>
            </div>
          ))}
        </div>
      </>
    );
  } else {
    looks = <p className="note">Click a row of the map, or pick a token in Tokens, to see which positions it attends to most.</p>;
  }
  return (
    <>
      <div className="title">{af.name}</div>
      <div className="sub">
        {layer < 0 ? 'mean over layers' : `layer ${layer + 1}`} · {head < 0 ? 'mean over heads' : `head ${head + 1}`}
      </div>
      {unit.attnFamilies.length > 1 && (
        <div className="block">
          <Seg label="Attention family" full value={fam} onChange={(i) => app.set({ fam: i, layer: -1, head: -1 })} options={unit.attnFamilies.map((f, i) => ({ value: i, label: f.name }))} />
        </div>
      )}
      <div className="block">
        <Seg
          label="Layout"
          full
          value={grid ? 'all' : 'one'}
          onChange={(v) => app.set({ grid: v === 'all' })}
          options={[
            { value: 'one', label: 'Single map' },
            { value: 'all', label: 'Layers × heads' },
          ]}
        />
      </div>
      <Heading>Layer</Heading>
      <Chips label="Layer" value={layer} onChange={(v) => app.set({ layer: v })} options={[{ value: -1, label: 'mean', title: 'Average over all layers' }, ...Array.from({ length: af.layers }, (_, l) => ({ value: l, label: String(l + 1), title: `Layer ${l + 1}` }))]} />
      <Heading>Head</Heading>
      <Chips label="Head" value={head} onChange={(v) => app.set({ head: v })} options={[{ value: -1, label: 'mean', title: 'Average over all heads' }, ...Array.from({ length: af.heads }, (_, k) => ({ value: k, label: String(k + 1), title: `Head ${k + 1}` }))]} />
      <Facts
        rows={[
          ['Layer', layer < 0 ? `mean of ${af.layers}` : `${layer + 1} of ${af.layers}`],
          ['Head', head < 0 ? `mean of ${af.heads}` : `${head + 1} of ${af.heads}`],
          ['Matrix', `${af.rows} × ${af.cols}${af.causal ? ', causal' : ''}`],
        ]}
      />
      {looks}
      <div className="scale" style={vars({ '--ramp': heatGradient() })}>
        0<i />
        {vmax.toFixed(3)}
      </div>
    </>
  );
}
