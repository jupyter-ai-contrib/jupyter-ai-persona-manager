/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'prepare-on-select';
const PERSONAS = [FixturePersona.PrepareConfig];

// The rendered chat messages; used to prove no message was sent while the
// controls appeared.
const MESSAGE = '.jp-chat-rendered-message';

/**
 * Verifies the model & settings controls appear as soon as a persona is
 * *selected*, before the user sends any message.
 */
test.describe('prepare-on-select', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('controls appear on selection, before any message is sent', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Select the persona but do NOT send a message. Selection alone must drive
    // the server to prepare it, which is when it publishes its configuration.
    await helpers.selectPersona(FixturePersona.PrepareConfig);

    // The controls appear once prepare() completes — without a message.
    await helpers.waitForControls();
    await expect(helpers.control('Model')).toBeVisible();
    await expect(helpers.control('Model')).toContainText('Prepared One');
    await expect(helpers.control('Thinking')).toBeVisible();
    await expect(helpers.control('Thinking')).toContainText('Medium');

    // Prove the controls appeared purely from selection: the chat has no
    // messages (no human message, no reply) at this point.
    await expect(helpers.chat.locator(MESSAGE)).toHaveCount(0);
  });

  test('the prepared model is usable once selected', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.PrepareConfig);
    await helpers.waitForControls();

    // Choose the non-default prepared model, then send: the persona is already
    // prepared, so it replies immediately.
    await helpers.setControl('Model', 'Prepared Two');
    const reply = await helpers.sendMessage('hi');
    expect(reply).toContain('prepared and ready');
  });
});
