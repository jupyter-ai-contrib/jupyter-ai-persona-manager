/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'restart';
const PERSONAS = [FixturePersona.Lifecycle];

// Markers the Lifecycle fixture posts on start and stop (kept in sync with
// fixtures/personas/lifecycle_persona.py).
const START_MARKER = 'lifecycle:started';
const STOP_MARKER = 'lifecycle:stopped';

/**
 * Exercises `BasePersona.restart()` (the API behind `/restart`) directly, from
 * inside the persona: the Lifecycle fixture posts a start message when it is
 * constructed and a stop message when it shuts down, and calls `self.restart()`
 * when it receives a message whose body is exactly "restart".
 *
 * A restart tears one instance down and reconstructs a fresh one under the same
 * ID, so in the chat it must show up as: the original start, then a stop, then a
 * brand-new start — proving the persona went down and came back up. The sibling
 * suite in jupyter-ai-chat-commands asserts the same markers via the `/restart`
 * command path.
 */
test.describe('restart', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('restart() takes the persona down and brings it back up', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Lifecycle);

    // The persona posts its start marker when the chat first constructs it.
    await expect
      .poll(async () => helpers.countMessagesContaining(START_MARKER), {
        timeout: 30000
      })
      .toBe(1);

    // Trigger a restart from inside the persona (it calls self.restart()).
    await helpers.send('restart');

    // The stop message proves the old instance shut down.
    await expect
      .poll(async () => helpers.countMessagesContaining(STOP_MARKER), {
        timeout: 30000
      })
      .toBe(1);

    // A fresh start message proves a new instance came up under the same ID.
    await expect
      .poll(async () => helpers.countMessagesContaining(START_MARKER), {
        timeout: 30000
      })
      .toBe(2);
  });
});
