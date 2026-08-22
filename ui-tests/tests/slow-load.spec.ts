/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'slow-load';
const PERSONAS = [FixturePersona.SlowLoad];

/**
 * Verifies the chat toolbar shows its "loading personas" placeholder while the
 * persona list is still resolving, then resolves to a working picker once
 * loading finishes (issue #77).
 *
 * The persona list the browser renders comes from the PersonaManager's awareness
 * slot, which it publishes only after loading every persona. The Slow Load
 * fixture delays its own module import, so the manager takes a few seconds to
 * publish — and while it does, the toolbar stays on its loading placeholder
 * (what `PersonaManagerAwareness.from()` shows while polling for the slot).
 *
 * The delay is bounded, not forever: the manager is constructed synchronously on
 * the server's event loop, so a permanent stall would hang the shared server and
 * every other suite (see the fixture docstring). The eventual give-up/timeout
 * behavior is out of scope (issue #77 defers it until lazy `async init_session()`
 * exists).
 *
 * WS-only: this asserts on the transient loading placeholder, which only exists
 * when the chat id resolves *before* the persona list. That holds over the
 * WebSocket transport, where the id arrives in the connection frame (~2s) well
 * ahead of the slow persona load (~10s). Under RTC the id instead travels in the
 * collaborative document's initial sync, which is stalled behind the same
 * event-loop-blocking persona import; measured end to end, the id and the
 * persona list then reach the browser ~3ms apart (vs ~8s over WS), so the
 * placeholder window never opens even though the picker resolves correctly. The
 * eventual resolution under RTC is covered by picker.spec.ts. Skipped under RTC
 * (E2E_RTC=1, set by noxfile) until persona loading no longer blocks the event
 * loop (issue #77).
 */
const RTC = process.env.E2E_RTC === '1';

test.describe('slow-load', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('shows the loading placeholder, then resolves to the picker', async ({
    page
  }) => {
    test.skip(
      RTC,
      'RTC delivers the chat id via the document sync (stalled behind the ' +
        'blocking persona import), so the loading-placeholder window never ' +
        'opens; see the suite docstring and issue #77.'
    );
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // While the slow module is still importing, the toolbar shows its loading
    // placeholder rather than a picker or an empty toolbar.
    await expect(helpers.loadingPlaceholder).toBeVisible({ timeout: 30000 });
    await expect(helpers.loadingPlaceholder).toHaveAttribute(
      'title',
      'Loading personas'
    );

    // Once loading finishes, the placeholder gives way to the real picker and
    // the persona is usable — the loading state was transient, not terminal.
    await helpers.selectPersona(FixturePersona.SlowLoad);
    await expect(helpers.loadingPlaceholder).toHaveCount(0);
    const reply = await helpers.sendMessage('hi');
    expect(reply).toContain('loaded');
  });
});
