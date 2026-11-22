/** Icons on a 24 x 24 grid, drawn as outlines unless marked filled. The cube is the mark of the app. */

import type { ReactNode } from 'react';

export type IconName =
  | 'open'
  | 'help'
  | 'camera'
  | 'fullscreen'
  | 'play'
  | 'pause'
  | 'segmentBack'
  | 'stepBack'
  | 'stepForward'
  | 'segmentForward'
  | 'loop'
  | 'eye'
  | 'close'
  | 'arrow'
  | 'rail'
  | 'reset';

const FILL = { fill: 'currentColor', stroke: 'none' } as const;

const ICONS: Record<IconName, ReactNode> = {
  open: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.2h7.5A2.5 2.5 0 0 1 21 9.7v7.8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z" />,
  help: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.6 9.4a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1.1 1-1.1 1.8" />
      <circle cx="12" cy="17" r="0.6" fill="currentColor" />
    </>
  ),
  camera: (
    <>
      <path d="M4 8h3l1.4-2h7.2L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" />
      <circle cx="12" cy="13" r="3.4" />
    </>
  ),
  fullscreen: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
  play: <path d="M8 5v14l11-7z" {...FILL} />,
  pause: <path d="M6 5h4v14H6zM14 5h4v14h-4z" {...FILL} />,
  segmentBack: <path d="M5 5h2v14H5zM20 6v12L9 12z" {...FILL} />,
  stepBack: <path d="M17 6v12l-9-6z" {...FILL} />,
  stepForward: <path d="M7 6v12l9-6z" {...FILL} />,
  segmentForward: <path d="M17 5h2v14h-2zM4 6v12l11-6z" {...FILL} />,
  loop: <path d="M17 3l3 3-3 3M4 11V9.5A3.5 3.5 0 0 1 7.5 6H20M7 21l-3-3 3-3M20 13v1.5a3.5 3.5 0 0 1-3.5 3.5H4" />,
  eye: (
    <>
      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
  close: <path d="M6 6l12 12M18 6L6 18" />,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  rail: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="3" />
      <path d="M9.5 4.5v15" />
    </>
  ),
  reset: <path d="M4 12a8 8 0 1 0 2.6-5.9M4 4v4.5h4.5" />,
};

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {ICONS[name]}
    </svg>
  );
}

/** The mark: an outlined cube. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true" focusable="false">
      <path d="M12 3 19.8 7.5 19.8 16.5 12 21 4.2 16.5 4.2 7.5Z M4.2 7.5 12 12 19.8 7.5 M12 12V21" fill="none" stroke="#faf9f5" strokeWidth="1.3" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
