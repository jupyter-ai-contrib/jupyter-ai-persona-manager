/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, galata, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it: a
// persona that gates prepare() on auth and is never authenticated, plus one
// that authenticates once a sentinel file appears (the resume-poll test).
const TEST_DIR = 'auth-gated';
const PERSONAS = [FixturePersona.AuthGated, FixturePersona.AuthResume];

// The rendered chat messages, used to prove nothing was posted on selection.
const MESSAGE = '.jp-chat-rendered-message';

/**
 * Verifies the auth lifecycle: an auth-gated persona prompts for sign-in only
 * when the user sends a message, never on mere selection.
 *
 * Selecting a persona eagerly runs its `prepare()`. For an auth-gated persona
 * that fails the auth check (NOT_AUTHED), but the manager must stay silent —
 * the sign-in prompt is reserved for `on_message`. This is the regression this
 * suite pins: previously the prompt (and a login terminal) fired the instant
 * Kiro was selected, before any message.
 */
test.describe('auth-gated', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('selecting an auth-gated persona does not prompt for sign-in', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Select the persona but send nothing. Selection eagerly prepares it, which
    // fails the auth check — and that must produce no chat message.
    await helpers.selectPersona(FixturePersona.AuthGated);

    // Give the eager prepare task time to run (and, under the bug, to post).
    await page.waitForTimeout(2000);

    await expect(helpers.chat.locator(MESSAGE)).toHaveCount(0);
  });

  test('sending a message while unauthenticated prompts for sign-in', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.AuthGated);

    // A message (unlike selection) triggers the sign-in prompt.
    const reply = await helpers.sendMessage('hello');
    expect(reply).toContain('please sign in');
  });

  test('resumes automatically once auth succeeds, without a further message', async ({
    page,
    request
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.AuthResume);

    // Unauthenticated message: the sign-in prompt appears and the resume poll
    // starts running its check_auth_fn.
    const reply = await helpers.sendMessage('hello');
    expect(reply).toContain('please sign in');

    // Create the sentinel file the fixture's check_auth_fn looks for. The
    // already-running poll observes it and the persona resumes on its own — no
    // further user message is sent.
    const contents = galata.newContentsHelper(request);
    const created = await contents.uploadContent(
      '1',
      'text',
      '.auth-resume-signal'
    );
    expect(created).toBe(true);

    await expect(
      helpers.chat.locator(MESSAGE, { hasText: 'signed in, resuming' })
    ).toBeVisible({ timeout: 15000 });
  });
});
