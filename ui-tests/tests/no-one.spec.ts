/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, galata, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'no-one';
const PERSONAS = [FixturePersona.Refresher];

/**
 * Verifies an explicit "No one" selection survives a real persona-list
 * republish — deterministically.
 *
 * `reconcileSelection` only runs when the persona list actually changes (the
 * manager republishes it on init and when the persona set changes; awareness
 * clock renewals don't fire `change`). The dangerous path is the sole-persona
 * convenience: with exactly one persona installed, an empty selection is
 * auto-seeded with it — but an explicit user choice, including "No one", must
 * stick. So this suite installs exactly ONE persona and forces a genuine
 * republish (reloaded personas get new awareness client IDs, so the list
 * content changes) while "No one" is selected.
 *
 * Sequencing is signal-driven, not timed: the Refresher fixture arms on a
 * message, then waits for a `go-refresh` file which the test uploads only
 * after selecting "No one"; the refresh's own system message ("Refreshed all
 * AI personas…") marks when the republish has landed.
 */
test.describe('no-one', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('an explicit "No one" survives a persona-list republish', async ({
    page,
    request
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Make an explicit pick (userPicked), then explicitly select "No one".
    await helpers.selectPersona(FixturePersona.Refresher);
    const reply = await helpers.sendMessage('arm the refresher');
    expect(reply).toContain('armed');
    await helpers.selectNoOne();

    // Only now signal the refresher: selection strictly precedes the republish.
    const contents = galata.newContentsHelper(request);
    await contents.uploadContent('go', 'text', `${TEST_DIR}/go-refresh`);

    // The refresh posts a system message once the rebuilt list is republished.
    await expect
      .poll(async () => helpers.lastMessageText(), { timeout: 30000 })
      .toContain('Refreshed all AI personas');

    // The republished sole persona must not be reseeded over the explicit
    // "No one". The republish has provably happened (the system message is
    // posted after it); allow the client a moment to apply the awareness
    // update, then require the selection to still be "No one".
    await page.waitForTimeout(4000);
    await expect(helpers.personaPicker).toContainText('No one');
  });
});
