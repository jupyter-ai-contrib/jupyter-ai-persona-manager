/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it.
const TEST_DIR = 'picker';
const PERSONAS = [FixturePersona.Hello, FixturePersona.Hello2];

// Each fixture persona replies with a distinct word, so a reply unambiguously
// identifies which persona handled the message.
const REPLY: Record<string, string> = {
  [FixturePersona.Hello]: 'hello',
  [FixturePersona.Hello2]: 'bonjour'
};

/**
 * Verifies the persona picker: it lists the installed personas, routes each
 * message to the *selected* persona, and keeps an explicit "No one" selection.
 *
 * With more than one persona in a chat none is auto-selected, so the message's
 * `to_persona` metadata comes solely from the user's picker choice.
 */
test.describe('picker', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('lists installed personas and routes to the selected one', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Open the picker: both installed personas plus "No one" are offered.
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });
    await helpers.personaPicker.click();
    await expect(
      page.getByRole('option', { name: 'Hello Persona' })
    ).toBeVisible();
    await expect(
      page.getByRole('option', { name: 'Bonjour Persona' })
    ).toBeVisible();
    await expect(page.getByRole('option', { name: 'No one' })).toBeVisible();
    await page.keyboard.press('Escape');

    // Switch to each persona in turn (and back to the first) and confirm the
    // reply is that persona's distinctive word, never the other's.
    const order = [
      FixturePersona.Hello,
      FixturePersona.Hello2,
      FixturePersona.Hello
    ];
    for (const persona of order) {
      await helpers.selectPersona(persona);
      const reply = await helpers.sendMessage('who are you?');
      expect(reply).toContain(REPLY[persona]);
    }
  });

  test('keeps an explicit "No one" selection', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Hello);

    await helpers.selectNoOne();

    // Sending a message triggers awareness traffic (personas rebroadcast state
    // routinely); the explicit "No one" must survive it and reach no persona.
    await helpers.sendWithoutReply('anyone there?');
    await page.waitForTimeout(2500);
    await expect(helpers.personaPicker).toContainText('No one');
  });
});
