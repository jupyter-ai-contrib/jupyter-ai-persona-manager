/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'loading';
const PERSONAS = [FixturePersona.Loading];

/**
 * Verifies the chat toolbar shows its "loading personas" placeholder while the
 * persona list is still resolving (issue #77).
 *
 * The persona list the browser renders comes from the `PersonaManager`'s
 * awareness slot, which the manager only publishes once it has finished
 * constructing every persona. The Loading fixture never finishes initializing
 * (it blocks in `__init__`), so the manager for this chat never publishes its
 * slot, and the toolbar stays on the loading placeholder that
 * `PersonaManagerAwareness.from()` shows while it polls for the slot.
 *
 * A never-finishing persona is deliberate: it gives a *reproducible* loading
 * state with no timing to race (a merely-slow persona could publish before the
 * assertion runs). Because this chat's manager never finishes constructing, this
 * fixture gets its own suite directory — see the fixture's docstring.
 *
 * The eventual give-up/timeout behavior is intentionally out of scope here (issue
 * #77 defers it until lazy `async init_session()` exists).
 */
test.describe('loading', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('shows the loading placeholder while the persona list resolves', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // The placeholder appears and stays: the manager never publishes its slot,
    // so the toolbar keeps waiting rather than resolving to a picker.
    await expect(helpers.loadingPlaceholder).toBeVisible({ timeout: 30000 });
    await expect(helpers.loadingPlaceholder).toHaveAttribute(
      'title',
      'Loading personas'
    );

    // It never resolves into a picker: the persona button stays absent. Give the
    // frontend a beat, then confirm the placeholder still stands and no picker
    // has appeared.
    await page.waitForTimeout(3000);
    await expect(helpers.loadingPlaceholder).toBeVisible();
    await expect(helpers.personaPicker).toHaveCount(0);
  });
});
