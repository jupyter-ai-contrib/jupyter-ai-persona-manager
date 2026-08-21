/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'persona-status';
const PERSONAS = [FixturePersona.Status];

// The chat's writing indicator root (see jupyter-chat's WritingIndicator; the
// class is documented as "Used in E2E tests"). With a custom `typingIndicator`
// it renders "<name> <indicator>".
const WRITERS = '.jp-chat-writers';

/**
 * Verifies the persona status API (`set_status` / `clear_status`, issue #128)
 * end to end, against the real chat writers mechanism rather than a mock.
 *
 * The Status fixture, on any message, sets the default status ("is typing..."),
 * dwells, sets a caller-supplied status ("is thinking..."), dwells, then clears
 * it. We assert the chat's writing indicator reflects each step.
 */
test.describe('persona-status', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('set_status updates the writing indicator and clear_status removes it', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Status);

    const writers = helpers.chat.locator(WRITERS);

    // Trigger the persona to step through its status sequence.
    await helpers.send('go');

    // Default status.
    await expect(writers).toContainText('is typing...', { timeout: 30000 });
    // Caller-set status.
    await expect(writers).toContainText('is thinking...', { timeout: 30000 });
    // Cleared: the indicator no longer shows a status.
    await expect(writers).not.toContainText('is thinking...', {
      timeout: 30000
    });
    await expect(writers).not.toContainText('is typing...');
  });
});
