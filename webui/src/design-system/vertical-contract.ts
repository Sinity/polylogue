import type { ComponentChildren } from 'preact';

import { DESIGN_SYSTEM_CONTRACT_VERSION } from '../generated/contracts';
import type { VerticalId, VerticalState } from './types';

export const WEBUI_VERTICAL_IDS = [
  'webui-02',
  'webui-03',
  'webui-04',
  'webui-05',
  'webui-06',
] as const satisfies ReadonlyArray<VerticalId>;

export const WEBUI_VERTICAL_STATES = [
  'ready',
  'loading',
  'empty',
  'unknown',
  'degraded',
  'error',
] as const satisfies ReadonlyArray<VerticalState>;

export const WEBUI_VERTICAL_CONTRACT = {
  version: String(DESIGN_SYSTEM_CONTRACT_VERSION),
  rootSelector: 'main#main-content[data-webui-contract][data-webui-vertical][data-state]',
  headingSuffix: '-heading',
} as const;

export interface VerticalFrameProps {
  id: VerticalId;
  state: VerticalState;
  title: string;
  description?: string;
  actions?: ComponentChildren;
  children: ComponentChildren;
}

export function verticalHeadingId(id: VerticalId): `${VerticalId}-heading` {
  return `${id}${WEBUI_VERTICAL_CONTRACT.headingSuffix}`;
}
