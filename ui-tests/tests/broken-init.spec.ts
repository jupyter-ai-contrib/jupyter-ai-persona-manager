/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it: a
// persona that raises during init, alongside a working one.
const TEST_DIR = 'broken-init';
const PERSONAS = [FixturePersona.BrokenInit, FixturePersona.Hello];

/**
 * Verifies a persona that fails to initialize degrades gracefully (issue #77).
 *
 * The PersonaManager instantiates each persona inside a `try/except Exception`:
 * one that raises is skipped (with a system message naming it), and the rest of
 * the chat still loads. So a single broken persona must not take down the whole
 * picker — the toolbar resolves and its working personas stay usable.
 *
 * This suite installs the Broken Init fixture (raises in `__init__`) together
 * with the Hello persona, and asserts the toolbar still lists and routes to
 * Hello, and that a system message names the persona that failed.
 */
test.describe('broken-init', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('a persona failing to init still leaves the picker usable', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // The picker resolves despite the broken persona: the loading placeholder
    // gives way to the working persona, which is selectable and routes normally.
    await helpers.selectPersona(FixturePersona.Hello);
    const reply = await helpers.sendMessage('hi');
    expect(reply).toContain('hello');
  });

  test('a persona failing to init posts a system message naming it', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Opening the chat initializes the personas; the broken one triggers a
    // system message that names the failed persona's class and explains that
    // other personas are unaffected.
    const message = await helpers.waitForMessageContaining('failed to load');
    expect(message).toContain('BrokenInitPersona');
    expect(message).toContain('unavailable');

    // The traceback is tucked into a collapsible <details>, which survives the
    // renderer's HTML sanitizer — confirming the inline markup renders rather
    // than showing as escaped text.
    await expect(
      helpers.chat.locator('.jp-chat-rendered-message details summary').first()
    ).toContainText('BrokenInitPersona');
  });
});
