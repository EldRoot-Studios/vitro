/** Small pieces several screens share: the drop area, segmented controls, chips, key-value lists, probability bars. */

import { useState, type ReactNode } from 'react';

import type { TraceDump, UnitTrace } from '../core/trace.ts';
import { openGroup } from '../state/session.ts';
import { groupLibrary } from '../state/library.ts';
import type { LibraryState } from '../state/state.ts';
import { formatBytes, formatProb, visibleLabel } from './format.ts';
import { Icon } from './icons.tsx';
import { pickFiles } from './picker.ts';

/** The drop area. It only lights up while files are dragged over it; the drop itself is handled for the whole window. */
export function Pick() {
  const [over, setOver] = useState(false);
  return (
    <div className="drop" data-over={over || undefined} onDragOver={() => setOver(true)} onDragLeave={() => setOver(false)} onDrop={() => setOver(false)}>
      <div>
        <b>Drop a trace here</b>
        <br />a .vtdump, a .vtdemo, or both from one run
      </div>
      <button type="button" className="btn" onClick={pickFiles}>
        <Icon name="open" size={15} />
        Choose files
      </button>
    </div>
  );
}

/** A row of mutually exclusive choices. */
export function Seg<T extends string | number>({ label, value, options, onChange, full }: { label: string; value: T; options: readonly { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; full?: boolean }) {
  return (
    <div className={full ? 'seg full' : 'seg'} role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" aria-pressed={o.value === value} title={o.title} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Small square choices, such as layer or head numbers. */
export function Chips<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: readonly { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="chips" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" className="chip" aria-pressed={o.value === value} title={o.title} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Name and value pairs in two columns. */
export function Facts({ rows }: { rows: readonly (readonly [string, ReactNode])[] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} className="kv-row">
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Heading of a group in a side panel. */
export function Heading({ children }: { children: ReactNode }) {
  return <h4>{children}</h4>;
}

/** The probability bars shared by the inspector of the Tokens view and of the 3D net: the model's top candidates for token `j`. */
export function CandidateList({ dump, unit, j }: { dump: TraceDump; unit: UnitTrace; j: number }) {
  const k = dump.topK;
  const target = unit.tok[j];
  const rows: ReactNode[] = [];
  let shown = false;
  for (let c = 0; c < k; c++) {
    const id = unit.topId[j * k + c];
    if (id < 0) break;
    const p = unit.topP[j * k + c];
    const isTarget = id === target;
    shown ||= isTarget;
    rows.push(<Candidate key={c} rank={c + 1} label={dump.label(id)} p={p} ok={isTarget} />);
  }
  // the target was not among the top candidates: it is listed last, so the person sees how far down it was
  if (!shown) rows.push(<Candidate key="miss" rank={null} label={dump.label(target)} p={unit.p[j]} ok={false} miss />);
  return <div className="cands">{rows}</div>;
}

/** One candidate: its place, the token, a bar for its probability and the number. */
export function Candidate({ rank, label, p, ok, miss }: { rank: number | null; label: string; p: number; ok: boolean; miss?: boolean }) {
  return (
    <div className="cand" data-ok={ok || undefined} data-miss={miss || undefined}>
      <span className="n">{rank ?? '·'}</span>
      <span className="t" title={label}>
        {visibleLabel(label)}
      </span>
      <span className="meter">
        <i style={{ width: `${Math.min(100, p * 100)}%` }} />
      </span>
      <span className="v">{formatProb(p)}</span>
    </div>
  );
}

/** One card per run the server knows, or a note saying why there are none. */
export function RunList({ library }: { library: LibraryState }) {
  if (library.status === 'loading') return <div className="library-note">Looking for sample traces…</div>;
  if (library.status === 'unavailable') {
    return (
      <div className="library-note">
        Sample traces are listed when the app is served by <span className="mono">vitro web</span>. Drop your own .vtdump or .vtdemo files above. ({library.error})
      </div>
    );
  }
  const groups = groupLibrary(library.files);
  if (groups.length === 0) return <div className="library-note">No traces found yet. Export one with vitro.export_dump(...) and drop it here.</div>;
  return (
    <div className="runs">
      {groups.map((g) => {
        const label = g.dump && g.demo ? 'Open run' : g.dump ? 'Open inspector' : 'Open replay';
        const files = [g.dump && `inspector ${formatBytes(g.dump.size)}`, g.demo && `replay ${formatBytes(g.demo.size)}`].filter(Boolean).join(' + ');
        return (
          <div className="run" key={g.key}>
            <div>
              <b>{g.title}</b>
              <small>{[g.model, g.arch.replace(/_/g, ' '), files].filter(Boolean).join(' · ')}</small>
            </div>
            {g.error ? (
              <span className="badge err" title={g.error}>
                cannot be read
              </span>
            ) : (
              <button type="button" className="btn ghost" onClick={() => void openGroup(g)}>
                {label}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
