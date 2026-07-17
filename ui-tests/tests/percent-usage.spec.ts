/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'percent-usage';
const PERSONAS = [FixturePersona.PercentUsage];

/**
 * Verifies the usage chip for a persona that reports context fill as a bare
 * percentage only (no token counts, no cost — the shape agents like kiro-cli
 * report). The chip shows a ring + percent; the popover shows the percent alone:
 * no "X of Y" token counts, no session-token breakdown, no cost.
 *
 * The Percent Usage fixture reports `context_percent: 42`, fixed for
 * deterministic UI text.
 */
test.describe('percent-usage', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('percent-only usage renders a ring and percent, no tokens', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.PercentUsage);

    await helpers.waitForUsage();

    // context_percent 42 -> ring + "42%".
    expect(await helpers.hasUsageRing()).toBe(true);
    expect(await helpers.usageChipText()).toBe('42%');

    // The popover shows the context percent alone: no "X of Y" token counts,
    // no cost, no session-token breakdown.
    const card = await helpers.openUsageCard();
    await expect(card).toContainText('Context');
    await expect(card).toContainText('42%');
    await expect(card).not.toContainText(' of ');
    await expect(card).not.toContainText('$');
    await expect(card).not.toContainText('Session tokens');
  });
});
