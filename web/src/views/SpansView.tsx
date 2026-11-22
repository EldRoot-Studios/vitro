/**
 * Spans: the labelled intervals of the sequence (prompt, answer, tree parts...) as a nested list, each with
 * its place in the sequence and what it cost the model. Clicking one opens its first token in Tokens.
 * Port of cpp/src/view_spans.cpp.
 */

import { spanStats } from '../core/analysis.ts';
import { PLAIN_TINT, css, parseTint } from '../core/colors.ts';
import { selectTokenPatch } from '../state/actions.ts';
import { app, useApp } from '../state/app.ts';
import { useUnit, vars } from './hooks.ts';

export function SpansStage() {
  const { dump, unit } = useUnit();
  const selected = useApp((s) => s.span);
  return (
    <div className="free-scroll">
      <header className="view-head">
        <div>
          <h2>Annotated spans</h2>
          <p>
            {dump.spans.length} {dump.spans.length === 1 ? 'span' : 'spans'} across {unit.nTokens} tokens. Click one to open it in Tokens.
          </p>
        </div>
      </header>
      {dump.spans.length === 0 ? (
        <p className="note">No spans were attached to this trace. Pass SpanLabel(...) in the Python TraceSpec to organise prompts, code blocks or tree parts.</p>
      ) : (
        <div className="span-list" role="list">
          {dump.spans.map((sp, i) => {
            const { lo, hi, errors, bits } = spanStats(unit, sp.start, sp.end);
            const total = Math.max(1, unit.nTokens);
            return (
              <button
                key={i}
                type="button"
                role="listitem"
                className="span-card"
                aria-current={i === selected}
                style={vars({ '--tint': css(parseTint(sp.tint, PLAIN_TINT)), '--depth': dump.spanDepth(i) })}
                onClick={() => app.set({ span: i, view: 'tokens', ...selectTokenPatch(app.get(), Math.max(unit.t0, lo)) })}
              >
                <span className="sc-bar" />
                <span className="sc-main">
                  <span className="sc-title">
                    <span className="sc-kind">{sp.kind}</span>
                    {sp.label}
                  </span>
                  <span className="sc-sub">{sp.meta || `tokens ${sp.start}–${sp.end - 1} (${Math.max(0, hi - lo)} tok)`}</span>
                </span>
                <span className="sc-track" aria-hidden="true">
                  <i style={{ left: `${(lo / total) * 100}%`, width: `${Math.max(1.2, ((hi - lo) / total) * 100)}%` }} />
                </span>
                <span className="sc-stat" data-state={errors ? 'warn' : 'ok'}>
                  {errors} {errors === 1 ? 'miss' : 'misses'} · {bits.toFixed(1)} bits
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
