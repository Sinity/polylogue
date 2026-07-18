import { expect, type Page } from '@playwright/test';

import { DESIGN_SYSTEM_CONTRACT_VERSION } from '../../src/generated/contracts';
import type { VerticalId, VerticalState } from '../../src/design-system/types';

export async function expectVerticalContract(
  page: Page,
  id: VerticalId,
  state?: VerticalState,
): Promise<void> {
  const root = page.locator('main#main-content');
  const headingId = `${id}-heading`;

  await expect(root).toHaveCount(1);
  await expect(root).toHaveAttribute('data-webui-contract', String(DESIGN_SYSTEM_CONTRACT_VERSION));
  await expect(root).toHaveAttribute('data-webui-vertical', id);
  await expect(root).toHaveAttribute('aria-labelledby', headingId);
  await expect(root.locator(`h1#${headingId}`)).toBeVisible();
  if (state) await expect(root).toHaveAttribute('data-state', state);
}
