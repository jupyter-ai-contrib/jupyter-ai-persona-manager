/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import {
  expect,
  galata,
  IJupyterLabPageFixture,
  test
} from '@jupyterlab/galata';
import { User } from '@jupyterlab/services';
import { UUID } from '@lumino/coreutils';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'rtc-reload';
const PERSONAS = [FixturePersona.Hello];
const HELLO = 'Hello Persona';

const TIMEOUT = 30000;

/** Auto-dismiss JupyterLab dialogs on a page (e.g. RTC's "document is taking
 * some time to load"), so they don't steal focus from the chat tab. */
async function autoDismissDialogs(page: IJupyterLabPageFixture): Promise<void> {
  await page.evaluate(() => {
    window.galata.on('dialog', d => {
      d?.resolve();
    });
  });
}

/**
 * Reproduces jupyterlab/jupyter-ai#1664: with RTC enabled, reloading the page
 * (or a second browser joining the same chat) leaves the persona control widget
 * empty, so there is no way to address an LLM.
 *
 * Why: the toolbar's persona list is delivered over Jupyter Events, which are
 * fire-and-forget. A client that joins *after* the list was emitted catches up
 * only when the PersonaManager re-publishes on jupyterlab_chat's `room/v1`
 * `client_connected` action. That action is emitted solely by jupyterlab_chat's
 * own WebSocket handler (the RTC-free `WsChatModel` transport). Under RTC
 * (jupyter_collaboration / jupyter_server_documents) clients connect through the
 * collaborative document's YRoom websocket instead, so `client_connected` never
 * fires, the catch-up never runs, and the reloaded/second toolbar never receives
 * the persona list.
 *
 * The first client to open a chat still sees its personas, because the manager
 * publishes the list once on init and that client is already subscribed — which
 * is why the bug only shows on reload or a second connection.
 *
 * These tests are NOT gated on RTC on purpose: under the RTC-free `default`
 * transport each WebSocket (re)connection fires `client_connected` and the
 * catch-up works, so they PASS — proving the test is valid and the defect is
 * RTC-specific. They are EXPECTED TO FAIL on the `jcollab` / `jsd` legs until
 * the catch-up is made transport-independent; they document the bug.
 *
 * The issue's headline scenario is a page reload. A reload produces the same
 * defect by the same mechanism — a fresh client joins an already-live room and
 * gets no catch-up — but the second-client form below is the deterministic E2E:
 * galata's `page.reload()` readiness wait is itself flaky under RTC (the same
 * collaborative document-loading timing the noxfile pins `jupyter_collaboration`
 * `<5` to avoid), which would confound the assertion with harness noise.
 */
test.describe('rtc reload (#1664)', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('a second client joining a live RTC chat sees the personas', async ({
    baseURL,
    browser,
    page,
    tmpPath,
    waitForApplication
  }) => {
    const chatPath = `${TEST_DIR}/second-client-${UUID.uuid4()}.chat`;
    await page.filebrowser.contents.uploadContent('{}', 'text', chatPath);

    // Client A opens the chat first: this creates the collaborative room and the
    // manager publishes its persona list on init, which A receives.
    const clientA = new TestHelpers({ dir: TEST_DIR, page });
    await clientA.openChat(chatPath);
    await expect(clientA.personaPicker).toBeVisible({ timeout: TIMEOUT });
    await expect(clientA.personaPicker).toContainText(HELLO);

    // Client B, a second browser context, joins the already-live room.
    const guest: Partial<User.IUser> = {
      identity: {
        username: 'jovyan_2',
        name: 'jovyan_2',
        display_name: 'jovyan_2',
        initials: 'J2',
        color: 'var(--jp-collaborator-color2)'
      }
    };
    const { page: pageB } = await galata.newPage({
      baseURL: baseURL!,
      browser,
      mockUser: guest,
      tmpPath,
      waitForApplication
    });
    await autoDismissDialogs(pageB);
    let guestPage: IJupyterLabPageFixture | null = pageB;
    try {
      const clientB = new TestHelpers({ dir: TEST_DIR, page: pageB });
      await clientB.openChat(chatPath);

      // B must resolve the persona list too. Under #1664 no `client_connected`
      // fires for B under RTC, so B's toolbar never shows the persona picker.
      await expect(clientB.personaPicker).toBeVisible({ timeout: TIMEOUT });
      await expect(clientB.personaPicker).toContainText(HELLO);
    } finally {
      await guestPage?.close();
      guestPage = null;
    }
  });
});
