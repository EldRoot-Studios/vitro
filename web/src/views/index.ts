/** The components of each view: the picture, the right panel and the bottom controls (null when a view has none). */

import type { ComponentType } from 'react';

import type { ViewId } from '../state/state.ts';
import { ActivationsInspector, ActivationsStage } from './ActivationsView.tsx';
import { AttentionInspector, AttentionStage } from './AttentionView.tsx';
import { NetDock, NetInspector, NetStage } from './NetView.tsx';
import { ReplayDock, ReplayInspector, ReplayStage } from './ReplayView.tsx';
import { SpansStage } from './SpansView.tsx';
import { TokensInspector, TokensStage } from './TokensView.tsx';

export interface ViewComponents {
  Stage: ComponentType;
  Inspector: ComponentType | null;
  Dock: ComponentType | null;
}

export const VIEW_COMPONENTS: Record<ViewId, ViewComponents> = {
  tokens: { Stage: TokensStage, Inspector: TokensInspector, Dock: null },
  attention: { Stage: AttentionStage, Inspector: AttentionInspector, Dock: null },
  activations: { Stage: ActivationsStage, Inspector: ActivationsInspector, Dock: null },
  spans: { Stage: SpansStage, Inspector: null, Dock: null },
  net: { Stage: NetStage, Inspector: NetInspector, Dock: NetDock },
  replay: { Stage: ReplayStage, Inspector: ReplayInspector, Dock: ReplayDock },
};
