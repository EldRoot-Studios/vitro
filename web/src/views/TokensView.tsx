/**
 * Tokens: the target sequence as coloured cells (surprisal, rank, entropy or numeric error), the model's free
 * rollout under each cell, labelled spans as coloured ticks, and an inspector with the probabilities of the
 * selected token. Port of cpp/src/view_tokens.cpp.
 */

import { memo, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from 'react';

import { COLOR_MODES, bitsOf, cellColor, describeToken, freeText, isScalarToken, nextError, tokenText, topId, type ColorMode } from '../core/analysis.ts';
import { PLAIN_TINT, css, deltaError, parseTint, surprisal, textOn, type RGB } from '../core/colors.ts';
import type { TraceDump, UnitTrace } from '../core/trace.ts';
import { selectTokenPatch, selectUnitPatch } from '../state/actions.ts';
import { app, useApp } from '../state/app.ts';
import { clamp, formatInt, formatProb, visibleLabel } from '../ui/format.ts';
import { useViewKeys } from '../ui/keys.ts';
import { CandidateList, Facts, Heading, Seg } from '../ui/parts.tsx';
import { tooltip } from '../ui/tooltip.ts';
import { useUnit, vars } from './hooks.ts';

const MODE_LABELS: Record<ColorMode, string> = { delta: 'Δ / bits', surprisal: 'Surprisal', rank: 'Rank', entropy: 'Entropy' };
const CELL_WIDTHS = [96, 78, 64, 56, 48, 40, 34, 28];
const CELL_GAP = 4;
/** Height of one row of cells: index gap, span tick, cell, free-rollout chip, row gap. */
const ROW_HEIGHT = 22 + 6 + 28 + 18 + 6;

function gradient(colorAt: (t: number) => RGB): string {
  return `linear-gradient(90deg, ${Array.from({ length: 25 }, (_, i) => css(colorAt(i / 24))).join(', ')})`;
}

interface CellData {
  j: number;
  label: string;
  free: 'same' | 'differs' | 'none' | null;
  freeLabel: string;
  wrong: boolean;
  context: boolean;
  diverge: boolean;
  style: CSSProperties;
}

function buildCells(dump: TraceDump, unit: UnitTrace, mode: ColorMode): CellData[] {
  const hasSpans = dump.spans.length > 0;
  const cells: CellData[] = [];
  for (let j = 0; j < unit.nTokens; j++) {
    const { color, wrong } = cellColor(dump, unit, j, mode);
    const cov = hasSpans ? dump.spansCovering(j) : [];
    const tint = cov.length ? css(parseTint(cov[cov.length - 1].tint, PLAIN_TINT)) : 'transparent';
    const predicted = j >= unit.t0;
    let free: CellData['free'] = null;
    if (predicted && unit.freeN > 0) free = j < unit.freeN ? (unit.ftok![j] === unit.tok[j] ? 'same' : 'differs') : 'none';
    cells.push({
      j,
      label: visibleLabel(tokenText(dump, unit, j)),
      free,
      freeLabel: free === null || free === 'none' ? '' : visibleLabel(freeText(dump, unit, j)),
      wrong,
      context: !predicted,
      diverge: j === unit.diverge,
      style: vars({ '--bg': css(color), '--fg': css(textOn(color)), '--tint': tint }),
    });
  }
  return cells;
}

const Cell = memo(function Cell({ c, selected, rowStart }: { c: CellData; selected: boolean; rowStart: boolean }) {
  return (
    <button type="button" role="listitem" className="cell" aria-current={selected} data-j={c.j} data-wrong={c.wrong || undefined} data-context={c.context || undefined} data-diverge={c.diverge || undefined} data-rowstart={rowStart || undefined} style={c.style}>
      <span className="tick" />
      <span className="main">{c.label}</span>
      {c.free && <span className={`free ${c.free}`}>{c.freeLabel}</span>}
    </button>
  );
});

/** Cell width in pixels: the user's choice, or the widest that shows every row without scrolling. */
function fitCellWidth(box: { w: number; h: number }, tokens: number): number {
  const usable = Math.max(200, box.w - 36);
  for (const w of CELL_WIDTHS) {
    const per = Math.max(1, Math.floor(usable / (w + CELL_GAP)));
    if (Math.ceil(tokens / per) * ROW_HEIGHT <= Math.max(120, box.h - 12)) return w;
  }
  return CELL_WIDTHS[CELL_WIDTHS.length - 1];
}

function Legend({ unit, dump, shown }: { unit: UnitTrace; dump: TraceDump; shown: number }) {
  const ramp = (name: string, colorAt: (t: number) => RGB, ticks: number[], span: number) => (
    <div className="ramp">
      <span className="cap">{name}</span>
      <div className="rbar">
        <div className="bar-ramp" style={{ background: gradient(colorAt) }} />
        <div className="ticks">
          {ticks.map((b) => (
            <span key={b} style={{ left: `${(b / span) * 100}%` }}>
              {b}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
  const cellW = useApp((s) => s.cellW);
  const current = cellW || shown;
  return (
    <div className="tv-legend">
      {ramp('surprisal, bits', (t) => surprisal(10 * t), [0, 1, 3, 6, 10], 10)}
      {unit.valMean && ramp('numeric error', (t) => deltaError(15 * t, dump.info.deltaRedAt), [0, 3, 12], 15)}
      <div className="miss-key">
        <i />
        top-1 ≠ target
      </div>
      <div className="zoom">
        <button type="button" className="chip" aria-label="Smaller cells" onClick={() => app.set({ cellW: Math.max(24, current - 6) })}>
          −
        </button>
        <button type="button" className="chip" aria-label="Larger cells" onClick={() => app.set({ cellW: Math.min(140, current + 6) })}>
          +
        </button>
        <button type="button" className="chip" aria-label="Fit cell size to the window" onClick={() => app.set({ cellW: 0 })}>
          fit
        </button>
      </div>
    </div>
  );
}

export function TokensStage() {
  const { dump, unit } = useUnit();
  const mode = useApp((s) => s.mode);
  const cellW = useApp((s) => s.cellW);
  const token = useApp((s) => s.token);
  const scroller = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const resize = new ResizeObserver(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      setBox((b) => (Math.abs(w - b.w) < 2 && Math.abs(h - b.h) < 2 ? b : { w, h }));
    });
    resize.observe(el);
    return () => resize.disconnect();
  }, []);

  const width = cellW > 0 ? cellW : fitCellWidth(box, unit.nTokens);
  const perRow = Math.max(1, Math.floor((Math.max(200, box.w - 36) + CELL_GAP) / (width + CELL_GAP)));
  const cells = useMemo(() => buildCells(dump, unit, mode), [dump, unit, mode]);
  const misses = useMemo(() => {
    let n = 0;
    for (let j = unit.t0; j < unit.nTokens; j++) if (unit.rank[j] > 1) n++;
    return n;
  }, [unit]);

  // keep the selected cell in view (the keyboard moves the selection while the grid scrolls)
  useEffect(() => {
    scroller.current?.querySelector('[aria-current="true"]')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [token, cells, perRow]);

  useViewKeys((e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const s = app.get();
    const at = s.token < 0 ? unit.t0 : s.token;
    const select = (j: number): void => app.set(selectTokenPatch(s, clamp(j, unit.t0, unit.nTokens - 1)));
    switch (e.key) {
      case 'ArrowLeft':
        select(s.token < 0 ? unit.t0 : at - 1);
        return true;
      case 'ArrowRight':
        select(s.token < 0 ? unit.t0 : at + 1);
        return true;
      case 'ArrowUp':
        select(at - perRow);
        return true;
      case 'ArrowDown':
        select(at + perRow);
        return true;
      case 'e':
      case 'E': {
        const j = nextError(unit, s.token, e.shiftKey ? -1 : 1);
        if (j >= 0) select(j);
        return true;
      }
      case 'd':
      case 'D':
        if (unit.diverge >= 0 && unit.diverge < unit.nTokens) select(unit.diverge);
        return true;
      case 'c':
      case 'C':
        app.set({ mode: COLOR_MODES[(COLOR_MODES.indexOf(s.mode) + 1) % COLOR_MODES.length] });
        return true;
      case '+':
      case '=':
        app.set({ cellW: Math.min(140, (s.cellW || width) + 6) });
        return true;
      case '-':
      case '_':
        app.set({ cellW: Math.max(24, (s.cellW || width) - 6) });
        return true;
      case '0':
        app.set({ cellW: 0 });
        return true;
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

  const cellOf = (e: { target: EventTarget }): { cell: HTMLElement; j: number } | null => {
    const cell = (e.target as HTMLElement).closest<HTMLElement>('.cell');
    return cell ? { cell, j: Number(cell.dataset.j) } : null;
  };
  const onMove = (e: PointerEvent): void => {
    const hit = cellOf(e);
    if (!hit) return tooltip.hide();
    let text = describeToken(dump, unit, hit.j);
    if ((e.target as HTMLElement).closest('.free') && hit.j < unit.freeN) {
      const same = unit.ftok![hit.j] === unit.tok[hit.j];
      text = `free rollout "${freeText(dump, unit, hit.j)}" (p=${unit.fp![hit.j].toFixed(3)}) - ${same ? 'matches the target' : 'differs from the target'}`;
    }
    tooltip.show(text, e.clientX, e.clientY);
  };

  return (
    <div className="free-scroll tv">
      <header className="view-head">
        <div>
          <h2>{unit.label}</h2>
          <p>
            {unit.nTokens} tokens ({Math.max(0, unit.nTokens - unit.t0)} predicted), {misses} top-1 {misses === 1 ? 'miss' : 'misses'}, {unit.bits.toFixed(1)} total bits
          </p>
        </div>
        <Seg label="Colouring" value={mode} onChange={(m) => app.set({ mode: m })} options={COLOR_MODES.map((m) => ({ value: m, label: MODE_LABELS[m] }))} />
      </header>
      <Legend unit={unit} dump={dump} shown={width} />
      <div className="tv-scroll" ref={scroller}>
        <div
          className="tv-grid"
          role="list"
          style={vars({ '--cell': `${width}px`, '--cols': perRow })}
          onClick={(e) => {
            const hit = cellOf(e);
            if (hit) app.set(selectTokenPatch(app.get(), hit.j));
          }}
          onPointerMove={onMove}
          onPointerLeave={() => tooltip.hide()}
        >
          {cells.map((c) => (
            <Cell key={c.j} c={c} selected={c.j === token} rowStart={c.j % perRow === 0} />
          ))}
        </div>
      </div>
    </div>
  );
}

export function TokensInspector() {
  const { dump, unit } = useUnit();
  const demo = useApp((s) => s.demo);
  const unitIndex = useApp((s) => s.unit);
  const j = useApp((s) => s.token);
  if (j < unit.t0) {
    return (
      <>
        <div className="title">Token</div>
        <p className="sub">Click a token, or use ← → and ↑ ↓. E jumps to the next miss, D to the first divergence.</p>
        <p className="note">The first tokens of a sequence are context: the model is not asked to predict them, so they have no probability.</p>
      </>
    );
  }
  const p = unit.p[j];
  const rank = unit.rank[j];
  const allowed = unit.nAllowed ? unit.nAllowed[j] : dump.vocab.size;
  const rows: [string, ReactNode][] = [
    ['Target probability', `${formatProb(p)} · ${bitsOf(p).toFixed(2)} bits`],
    ['Rank', <span key="rank" data-state={rank > 1 ? 'bad' : undefined}>{`${formatInt(rank)} of ${formatInt(allowed)}`}</span>],
    ['Entropy', `${unit.entropy[j].toFixed(2)} bits`],
  ];
  if (isScalarToken(dump, unit, j) && unit.valMean && unit.valStd) {
    const top = topId(dump, unit, j);
    const vt = dump.vocab.valueOf(unit.tok[j]);
    const vp = dump.vocab.hasValue(top) ? dump.vocab.valueOf(top) : unit.valMean[j];
    const bad = Math.abs(vp - vt) >= dump.info.deltaRedAt;
    rows.push(['Numeric value', <span key="num" data-state={bad ? 'bad' : undefined}>{`target ${+vt.toPrecision(4)}, top ${+vp.toPrecision(4)}`}</span>]);
    rows.push(['Mean ± std', `${unit.valMean[j].toFixed(2)} ± ${unit.valStd[j].toFixed(2)}`]);
  }
  if (j < unit.freeN && unit.ftok && unit.fp) {
    const same = unit.ftok[j] === unit.tok[j];
    rows.push(['Free rollout', <span key="free" data-state={same ? 'good' : 'bad'}>{`"${freeText(dump, unit, j)}" (p=${unit.fp[j].toFixed(3)}) ${same ? '✓' : '≠ target'}`}</span>]);
  }
  const tag = unit.tokenTags ? dump.vocab.tagName(unit.tokenTags[j]) : '';
  const spans = dump.spansCovering(j);
  const segIdx = demo ? Math.min(unitIndex, demo.segments.length - 1) : 0;
  const sg = demo ? demo.segments[segIdx] : null;
  const replayable = demo !== null && sg !== null && j >= sg.t0 && j < sg.ftok.length;
  return (
    <>
      <div className="big">
        {j} <span className="mono tok">{tokenText(dump, unit, j)}</span>
      </div>
      {(tag || spans.length > 0) && (
        <div className="tags">
          {tag && <span className="badge">{tag}</span>}
          {spans.map((s, i) => (
            <span className="badge" key={i}>
              <i className="dot" style={{ background: css(parseTint(s.tint, PLAIN_TINT)) }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <Facts rows={rows} />
      <Heading>Top-{dump.topK} candidates</Heading>
      <CandidateList dump={dump} unit={unit} j={j} />
      {replayable && (
        <div className="actions">
          <button type="button" className="btn ghost" onClick={() => app.set({ view: 'replay', pos: demo.positionOfToken(segIdx, j), playing: false })}>
            Replay this step
          </button>
        </div>
      )}
    </>
  );
}
