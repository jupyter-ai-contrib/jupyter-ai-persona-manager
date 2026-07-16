/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it.
const TEST_DIR = 'usage';
const PERSONAS = [FixturePersona.Usage, FixturePersona.Hello];

/**
 * Verifies the toolbar's usage chip renders token-based `Usage`
 * (`report_usage`): a context ring + percent, with the popover listing the
 * session-token breakdown and cost. Also verifies that a persona reporting no
 * usage renders no chip (absence reads as unknown, not empty).
 *
 * The Usage fixture reports context 1200/4000 (= 30%), session tokens 1500
 * (input 1000 / output 500), and a $0.42 cost, all fixed for deterministic UI
 * text (the client formats token counts compactly).
 */
test.describe('usage', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('token-based usage renders a context ring and a token/cost breakdown', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Usage);

    // Usage is advertised statically, so the chip is present on selection.
    await helpers.waitForUsage();

    // 1200 / 4000 = 30% context fill -> ring + "30%".
    expect(await helpers.hasUsageRing()).toBe(true);
    expect(await helpers.usageChipText()).toBe('30%');

    // The popover shows the context section, the session-token breakdown, and
    // the reported cost.
    const card = await helpers.openUsageCard();
    await expect(card).toContainText('Context');
    await expect(card).toContainText('1.2k of 4k');
    await expect(card).toContainText('Session tokens');
    await expect(card).toContainText('1.5k');
    await expect(card).toContainText('Input');
    await expect(card).toContainText('Output');
    await expect(card).toContainText('$0.42');
  });

  test('a persona reporting no usage renders no chip', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Hello);

    // The Hello fixture never calls report_usage. Give awareness a moment to
    // settle, then confirm the chip stays absent (absence reads as unknown).
    await page.waitForTimeout(2000);
    await expect(helpers.usageChip).toHaveCount(0);
  });
});
