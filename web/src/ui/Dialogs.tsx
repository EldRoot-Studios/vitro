/**
 * The two dialogs: opening traces (drop area, what is open now, the samples) and the keyboard reference.
 * Both are shown while the `filesOpen` and `helpOpen` flags of the state are set.
 */

import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react';

import { app, useApp } from '../state/app.ts';
import { closeSlot } from '../state/session.ts';
import type { FileInfo } from '../state/state.ts';
import { formatBytes } from './format.ts';
import { Icon } from './icons.tsx';
import { Pick, RunList } from './parts.tsx';

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

/** A modal sheet: takes focus when it opens, keeps Tab inside, gives focus back when it closes. */
function Modal({ label, title, sub, onClose, children }: { label: string; title: string; sub?: string; onClose: () => void; children: ReactNode }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    box.current?.focus();
    return () => before?.focus?.();
  }, []);
  const trap = (e: KeyboardEvent<HTMLDivElement>): void => {
    if (e.key !== 'Tab' || !box.current) return;
    const items = [...box.current.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => !el.hasAttribute('disabled'));
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog glass" role="dialog" aria-modal="true" aria-label={label} tabIndex={-1} ref={box} onKeyDown={trap}>
        <div className="dialog-head">
          <div>
            <h2>{title}</h2>
            {sub && <p>{sub}</p>}
          </div>
          <button type="button" className="ib" aria-label="Close" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function FilesDialog() {
  const dump = useApp((s) => s.dump);
  const demo = useApp((s) => s.demo);
  const dumpFile = useApp((s) => s.dumpFile);
  const demoFile = useApp((s) => s.demoFile);
  const library = useApp((s) => s.library);
  const rows: { label: string; file: FileInfo; kind: 'dump' | 'demo' }[] = [];
  if (dump && dumpFile) rows.push({ label: 'Inspector', file: dumpFile, kind: 'dump' });
  if (demo && demoFile) rows.push({ label: 'Replay', file: demoFile, kind: 'demo' });
  return (
    <Modal label="Open traces" title="Open traces" sub="A dump (.vtdump) powers the inspector tabs, a recording (.vtdemo) the replay. Open both files of one run to use every tab." onClose={() => app.set({ filesOpen: false })}>
      <Pick />
      {rows.length > 0 && (
        <div className="open-files">
          {rows.map((r) => (
            <div className="open-file" key={r.kind}>
              <span className="badge">{r.label}</span>
              <span className="name">{r.file.name}</span>
              <em>{formatBytes(r.file.size)}</em>
              <button type="button" className="ib" aria-label={`Close ${r.file.name}`} onClick={() => closeSlot(r.kind)}>
                <Icon name="close" size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
      <h4>Samples</h4>
      <RunList library={library} />
    </Modal>
  );
}

const KEYS: readonly (readonly [string, readonly (readonly [readonly string[], string])[]])[] = [
  [
    'Everywhere',
    [
      [['1', '–', '6'], 'Tokens, Attention, Activations, Spans, 3D Net, Replay'],
      [['O'], 'Open traces'],
      [['B'], 'Start screen'],
      [['S'], 'Save the picture of this view as PNG'],
      [['?'], 'This help'],
      [['Esc'], 'Close a dialog'],
    ],
  ],
  [
    'Tokens',
    [
      [['←', '→'], 'Previous / next token'],
      [['↑', '↓'], 'Token one row up / down'],
      [['E'], 'Next top-1 miss (Shift: previous)'],
      [['D'], 'First point where free generation diverges'],
      [['C'], 'Cycle colouring'],
      [['+', '−', '0'], 'Cell size, auto'],
      [['[', ']'], 'Previous / next unit'],
    ],
  ],
  [
    'Attention',
    [
      [['L'], 'Next layer (Shift: previous); wraps through the layer mean'],
      [['H'], 'Next head (Shift: previous); wraps through the head mean'],
      [['F'], 'Next attention family'],
      [['G'], 'All layers × heads on one page'],
      [['←', '→'], 'Previous / next token'],
    ],
  ],
  [
    '3D Net',
    [
      [['drag'], 'Orbit'],
      [['wheel'], 'Zoom'],
      [['click'], 'Pick the token behind a node'],
      [['Space'], 'Play / pause the wave'],
      [['P'], 'Rollout path only'],
      [['K'], 'Edges per node'],
      [['R'], 'Reset camera'],
    ],
  ],
  [
    'Replay',
    [
      [['Space'], 'Play / pause'],
      [['←', '→'], 'Step one token (Shift: ten)'],
      [['↑', '↓'], 'Faster / slower'],
      [['[', ']'], 'Previous / next segment'],
      [['Home', 'End'], 'Start / end'],
      [['L', 'G', 'D', 'R'], 'Loop, reference, divergence, camera'],
    ],
  ],
];

export function HelpDialog() {
  return (
    <Modal label="Keyboard shortcuts" title="Keys and mouse" sub="Everything the old viewer and player could do, in one window." onClose={() => app.set({ helpOpen: false })}>
      <div className="helpgrid">
        {KEYS.map(([title, rows]) => (
          <section key={title}>
            <h4>{title}</h4>
            <ul>
              {rows.map(([keys, what]) => (
                <li key={what}>
                  <span>
                    {keys.map((k) => (
                      <kbd key={k}>{k}</kbd>
                    ))}
                  </span>
                  <span>{what}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </Modal>
  );
}
