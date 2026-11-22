/**
 * The left panel. It always starts with the facts of the trace; below it comes what can be chosen in the open
 * view: the units of a dump for the inspector tabs, the segments and spans of a recording for the replay.
 */

import { PLAIN_TINT, css, parseTint } from '../core/colors.ts';
import type { TraceDemo } from '../core/demo.ts';
import { segmentTone } from '../core/replay.ts';
import type { TraceDump } from '../core/trace.ts';
import { app, useApp } from '../state/app.ts';
import { selectUnitPatch } from '../state/actions.ts';
import type { FileInfo } from '../state/state.ts';
import { formatBytes, formatInt } from './format.ts';
import { Facts, Heading } from './parts.tsx';

const TONES = { plain: 'var(--ink-3)', exact: 'var(--ok)', differs: 'var(--err)' } as const;

function TraceCard({ dump, demo, dumpFile, demoFile }: { dump: TraceDump | null; demo: TraceDemo | null; dumpFile: FileInfo | null; demoFile: FileInfo | null }) {
  const trace = (dump ?? demo)!;
  const info = trace.info;
  const rows: [string, string][] = [['Type', info.arch.replace(/_/g, ' ')], ['Device', `${info.device} / ${info.precision}`]];
  if (info.step !== null) rows.push(['Step', String(info.step)]);
  if (dump) {
    rows.push(['Tokens', formatInt(dump.units.reduce((n, u) => n + u.nTokens, 0))]);
    rows.push(['Units', String(dump.units.length)]);
    rows.push(['Attention groups', String(Math.max(...dump.units.map((u) => u.attnFamilies.length)))]);
  }
  if (demo) {
    rows.push(['Segments', String(demo.segments.length)]);
    rows.push(['Generated tokens', formatInt(demo.totalTokens)]);
  }
  rows.push(['Vocabulary', formatInt(trace.vocab.size)]);
  if (trace.spans.length) rows.push(['Spans', String(trace.spans.length)]);
  if (info.created) rows.push(['Recorded', info.created]);
  const files = [dump && dumpFile, demo && demoFile].filter((f): f is FileInfo => Boolean(f));
  rows.push(['Files', files.map((f) => formatBytes(f.size)).join(' + ')]);
  return (
    <section>
      <div className="title">{info.title}</div>
      <div className="sub">{info.modelName}</div>
      <Facts rows={rows} />
    </section>
  );
}

function Units({ dump, current }: { dump: TraceDump; current: number }) {
  return (
    <section>
      <Heading>Sequence units</Heading>
      {dump.units.map((u, i) => (
        <button key={i} type="button" className="row" aria-current={i === current} onClick={() => app.set(selectUnitPatch(app.get(), i))}>
          <span className="pair" title={u.freeN ? 'forced top-1 exact | free generation exact' : 'forced top-1 exact'}>
            <i data-state={u.forcedExact ? 'ok' : 'bad'} />
            <i data-state={u.freeN === 0 ? 'none' : u.freeExact ? 'ok' : 'bad'} />
          </span>
          <span className="name">{u.label}</span>
          <em>{u.nTokens} tok</em>
        </button>
      ))}
      <p className="hint">The two marks of a unit: forced top-1 exact above, free generation exact below. Green is exact.</p>
    </section>
  );
}

function Segments({ demo, current }: { demo: TraceDemo; current: number }) {
  const sg = demo.segments[current];
  return (
    <>
      <section>
        <Heading>Segments</Heading>
        {demo.segments.map((s, i) => (
          <button key={i} type="button" className="row" aria-current={i === current} onClick={() => app.set({ pos: s.start })}>
            <i style={{ background: TONES[segmentTone(s)] }} />
            <span className="name">{s.label}</span>
            <em>{s.ftok.length} tok</em>
          </button>
        ))}
      </section>
      {demo.spans.length > 0 && (
        <section>
          <Heading>Annotated spans</Heading>
          {demo.spans.map((sp, i) => (
            <button key={i} type="button" className="row" onClick={() => app.set({ pos: demo.positionOfToken(current, Math.max(sp.start, sg.t0)), playing: false })}>
              <i style={{ background: css(parseTint(sp.tint, PLAIN_TINT)) }} />
              <span className="name">{sp.label}</span>
              <em>
                {sp.start}–{sp.end - 1}
              </em>
            </button>
          ))}
        </section>
      )}
    </>
  );
}

export function Rail() {
  const dump = useApp((s) => s.dump);
  const demo = useApp((s) => s.demo);
  const dumpFile = useApp((s) => s.dumpFile);
  const demoFile = useApp((s) => s.demoFile);
  const replay = useApp((s) => s.view === 'replay');
  const unit = useApp((s) => s.unit);
  const seg = useApp((s) => (s.demo ? s.demo.locate(s.pos).seg : 0));
  if (!dump && !demo) return null;
  return (
    <>
      <TraceCard dump={dump} demo={demo} dumpFile={dumpFile} demoFile={demoFile} />
      {replay && demo ? <Segments demo={demo} current={seg} /> : dump ? <Units dump={dump} current={unit} /> : null}
    </>
  );
}
