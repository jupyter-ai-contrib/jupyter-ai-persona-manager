/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';

import {
  FIXTURE_PERSONAS,
  FixturePersona,
  installPersonas,
  TestHelpers
} from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'persona-identity';
const PERSONAS = [FixturePersona.Hello];

/**
 * A persona must appear in the chat with its own name and avatar -- not as a
 * raw username with no avatar.
 *
 * A persona registers itself with `chat.set_user()` when it is initialized,
 * which happens *after* the web client has already connected. If that
 * registration is not pushed to connected clients, the client never learns the
 * persona's identity, and its replies render with the fallback the frontend
 * uses for an unknown sender: the raw persona id (e.g.
 * `jupyter-ai-personas::...::HelloPersona`) as the name, and no avatar.
 *
 * The other suites only assert reply *bodies* and the persona *picker* (which is
 * fed by a separate events channel), so none of them exercises this. This test
 * asserts the rendered sender identity directly.
 *
 * NOTE: this is expected to FAIL on the RTC-free (default) transport until the
 * jupyter-chat fix that broadcasts `set_user` updates is released
 * (jupyterlab/jupyter-chat#531). Under real-time collaboration it already passes,
 * because `set_user` writes to the shared document.
 */
test.describe('persona identity', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test("a persona reply shows the persona's name and avatar", async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    const chat = await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Hello);

    const reply = await helpers.sendMessage('hi there');
    expect(reply).toContain('hello');

    // The persona reply is the last message. Its header must show the persona's
    // display name and an avatar image -- the identity carried by
    // `persona.as_user()` -- rather than the unknown-sender fallback (raw id,
    // no avatar).
    const header = chat
      .locator('.jp-chat-message')
      .last()
      .locator('.jp-chat-message-header');

    const { name } = FIXTURE_PERSONAS[FixturePersona.Hello]; // 'Hello Persona'
    await expect(header).toContainText(name);
    await expect(header.locator('img')).toHaveCount(1);
  });
});
