/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'system-messages';
const PERSONAS = [FixturePersona.SystemMessage];

// Kept in sync with SYSTEM_TEXT in fixtures/personas/system-message_persona.py.
const SYSTEM_TEXT = 'System check: all systems nominal.';

/**
 * Verifies system messages render in the chat (issue #124).
 *
 * `PersonaManager.send_system_message()` attributes a message to a single
 * "System" user that the manager registers once at initialization (with
 * `bot=True`, so Jupyter Chat hides it from the `@`-mention menu). The System
 * Message fixture calls it via `self.parent` on any input, so sending a message
 * makes a system message appear.
 */
test.describe('system-messages', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('a persona can post a system message that renders in the chat', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.SystemMessage);

    // Any message triggers the persona to post a system message.
    await helpers.send('trigger a system message');

    // The system message renders in the chat.
    const message = await helpers.waitForMessageContaining(SYSTEM_TEXT);
    expect(message).toContain(SYSTEM_TEXT);
  });
});
