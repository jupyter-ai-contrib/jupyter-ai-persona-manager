/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'credits-usage';
const PERSONAS = [FixturePersona.CreditsUsage];

/**
 * Verifies the usage chip for a persona whose session cost is metered in a
 * vendor unit ("credits") rather than an ISO currency — the shape kiro-cli
 * reports (a context percentage plus a credit cost). `formatCost` renders an
 * ISO "USD" cost as "$0.42" but a unit-named cost with the unit after the
 * amount; the popover must show "0.09 credits", never a "$".
 *
 * The Credits Usage fixture reports `context_percent: 12` and a cost of 0.09
 * "credits", fixed for deterministic UI text.
 */
test.describe('credits-usage', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('a unit-named cost renders as "<amount> <unit>", not a currency', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.CreditsUsage);

    await helpers.waitForUsage();

    // context_percent 12 -> ring + "12%".
    expect(await helpers.hasUsageRing()).toBe(true);
    expect(await helpers.usageChipText()).toBe('12%');

    // The popover shows the session cost in the vendor's unit, with the unit
    // name after the amount and no currency symbol.
    const card = await helpers.openUsageCard();
    await expect(card).toContainText('Session cost');
    await expect(card).toContainText('0.09 credits');
    await expect(card).not.toContainText('$');
  });
});
