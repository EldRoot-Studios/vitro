/** Entry point: styles, the errors nobody catches, and the app. */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import './styles/tokens.css';
import './styles/components.css';
import './styles/shell.css';
import './styles/views.css';
import { toasts } from './state/toasts.ts';
import { App } from './ui/App.tsx';

// A failure nobody catches must still be seen: show it, never swallow it.
window.addEventListener('error', (e) => toasts.show(`Unexpected error: ${e.message}`, 'err'));
window.addEventListener('unhandledrejection', (e) => toasts.show(`Unexpected error: ${e.reason instanceof Error ? e.reason.message : String(e.reason)}`, 'err'));

const root = document.getElementById('root');
if (!root) throw new Error('index.html has no #root element');
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
