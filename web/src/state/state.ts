/**
 * The state of the whole app. There is one session: at most one trace dump (for the inspector views) and at most
 * one recording (for the replay), normally two files of the same run. `view` says which tab is open.
 *
 * This module holds only data and constants, no behaviour, so that the tests can load it without a browser.
 */

import type { ColorMode } from '../core/analysis.ts';
import type { TraceDemo } from '../core/demo.ts';
import type { TraceDump } from '../core/trace.ts';
import type { LibraryEntry } from './library.ts';

export type ViewId = 'tokens' | 'attention' | 'activations' | 'spans' | 'net' | 'replay';

export interface ViewInfo {
  id: ViewId;
  label: string;
  /** The file kind the view reads. */
  needs: 'dump' | 'demo';
  /** What to say when that file is not open. */
  empty: string;
}

/** Views in tab order. */
export const VIEWS: readonly ViewInfo[] = [
  { id: 'tokens', label: 'Tokens', needs: 'dump', empty: 'Open a trace dump (.vtdump) to see every token with its probability.' },
  { id: 'attention', label: 'Attention', needs: 'dump', empty: 'Open a trace dump (.vtdump) to inspect attention heads and layers.' },
  { id: 'activations', label: 'Activations', needs: 'dump', empty: 'Open a trace dump (.vtdump) to see activation norms per layer.' },
  { id: 'spans', label: 'Spans', needs: 'dump', empty: 'Open a trace dump (.vtdump) to see the labelled spans of the sequence.' },
  { id: 'net', label: '3D Net', needs: 'dump', empty: 'Open a trace dump (.vtdump) to follow a prediction through the network in 3D.' },
  { id: 'replay', label: 'Replay', needs: 'demo', empty: 'Open a generation recording (.vtdemo) to replay it token by token.' },
];

export interface CameraPose {
  yaw: number;
  pitch: number;
  zoom: number;
}

/** Starting cameras of the two 3D scenes. */
export const NET_CAMERA: CameraPose = { yaw: 0.3, pitch: 0.14, zoom: 1 };
export const REPLAY_CAMERA: CameraPose = { yaw: 0.42, pitch: 0.16, zoom: 1 };

/** A file in the session: where it came from and how big it is. `id` is the server's number, null for a picked file. */
export interface FileInfo {
  name: string;
  size: number;
  id: number | null;
}

export interface LibraryState {
  status: 'loading' | 'ready' | 'unavailable';
  files: LibraryEntry[];
  error: string;
}

export interface AppState {
  screen: 'home' | 'app';
  view: ViewId;
  dump: TraceDump | null;
  demo: TraceDemo | null;
  dumpFile: FileInfo | null;
  demoFile: FileInfo | null;
  library: LibraryState;

  /** Unit shown by the inspector views, and the selected target token in it (-1: none). */
  unit: number;
  token: number;

  /** Attention view: family, layer and head (-1: the mean), and the all-layers-by-heads page. */
  fam: number;
  layer: number;
  head: number;
  grid: boolean;

  /** Activations view: latent map and its selected row. */
  latent: number;
  latentRow: number;

  /** Spans view selection, Tokens view colouring and cell width (0: fit the window). */
  span: number;
  mode: ColorMode;
  cellW: number;

  /** 3D net: camera, incoming edges per node, rollout-only drawing, the wave and its clock. */
  netYaw: number;
  netPitch: number;
  netZoom: number;
  netK: number;
  netPath: boolean;
  netPlay: boolean;
  netTime: number;

  /** Replay: position on the timeline in ticks, transport and camera. */
  pos: number;
  playing: boolean;
  speed: number;
  loop: boolean;
  ghost: boolean;
  yaw: number;
  pitch: number;
  zoom: number;

  helpOpen: boolean;
  filesOpen: boolean;
  railOpen: boolean;
}

export const INITIAL_STATE: AppState = {
  screen: 'home',
  view: 'tokens',
  dump: null,
  demo: null,
  dumpFile: null,
  demoFile: null,
  library: { status: 'loading', files: [], error: '' },

  unit: 0,
  token: -1,

  fam: 0,
  layer: -1,
  head: -1,
  grid: false,

  latent: 0,
  latentRow: -1,

  span: -1,
  mode: 'delta',
  cellW: 0,

  netYaw: NET_CAMERA.yaw,
  netPitch: NET_CAMERA.pitch,
  netZoom: NET_CAMERA.zoom,
  netK: 3,
  netPath: false,
  netPlay: true,
  netTime: 0,

  pos: 0,
  playing: true,
  speed: 1,
  loop: true,
  ghost: true,
  yaw: REPLAY_CAMERA.yaw,
  pitch: REPLAY_CAMERA.pitch,
  zoom: REPLAY_CAMERA.zoom,

  helpOpen: false,
  filesOpen: false,
  railOpen: true,
};
