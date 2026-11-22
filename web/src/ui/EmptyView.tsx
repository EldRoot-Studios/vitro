/** A view of a kind of file that is not open: says what to open, and offers to do it. */

import { app } from '../state/app.ts';
import type { ViewInfo } from '../state/state.ts';
import { Icon } from './icons.tsx';

export function EmptyView({ view }: { view: ViewInfo }) {
  return (
    <div className="empty glass">
      <h2>
        {view.label} needs a {view.needs === 'dump' ? 'trace dump' : 'recording'}
      </h2>
      <p>{view.empty}</p>
      <button type="button" className="btn" onClick={() => app.set({ filesOpen: true })}>
        <Icon name="open" size={15} />
        Open files
      </button>
    </div>
  );
}
