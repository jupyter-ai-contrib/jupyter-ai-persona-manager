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
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory, the fixture personas installed into it, and
// the single shared chat both clients join.
const TEST_DIR = 'multi-user';
const PERSONAS = [FixturePersona.EchoConfig, FixturePersona.Models];
const SHARED_CHAT = `${TEST_DIR}/shared.chat`;

/**
 * Verifies persona/model selection is per-client, and routing is per-message.
 *
 * Two browser clients join the *same* collaborative chat. Selection lives only
 * in the frontend — it's stamped onto each outgoing message's metadata and is
 * never broadcast over awareness — so one client picking a persona/model must
 * not change the other client's toolbar. And because routing is driven by that
 * per-message `to_persona` metadata, each client's message must reach exactly
 * the persona *that client* selected, even though both personas live in the one
 * shared chat.
 *
 * Client A picks the Echo Config persona and its "Claude Opus" model (id
 * `claude-opus`); client B picks the Models persona and its "Model Gamma" (id
 * `gamma`). `claude-opus` is only an Echo Config option and `gamma` only a
 * Models option, so each distinctive reply proves the right persona handled the
 * right client's message.
 *
 * This is the hardest suite (two contexts + collaborative sync). If it proves
 * flaky in CI, prefer marking it `test.fixme` with a note over letting it block
 * the rest of the suite — the single-client specs already cover the controls.
 */
test.describe('multi-user', () => {
  let guestPage: IJupyterLabPageFixture;

  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test.beforeEach(
    async ({ baseURL, browser, page, tmpPath, waitForApplication }) => {
      // The shared chat file both clients open.
      await page.filebrowser.contents.uploadContent('{}', 'text', SHARED_CHAT);

      // A second, distinct collaborator in its own browser context.
      const guest: Partial<User.IUser> = {
        identity: {
          username: 'jovyan_2',
          name: 'jovyan_2',
          display_name: 'jovyan_2',
          initials: 'J2',
          color: 'var(--jp-collaborator-color2)'
        }
      };
      const { page: newPage } = await galata.newPage({
        baseURL: baseURL!,
        browser,
        mockUser: guest,
        tmpPath,
        waitForApplication
      });
      await newPage.evaluate(() => {
        window.galata.on('dialog', d => {
          d?.resolve();
        });
      });
      guestPage = newPage;
    }
  );

  test.afterEach(async ({ page }) => {
    await guestPage.close();
    if (await page.filebrowser.contents.fileExists(SHARED_CHAT)) {
      await page.filebrowser.contents.deleteFile(SHARED_CHAT);
    }
  });

  test('per-client selection stays isolated and routes to the right persona', async ({
    page
  }) => {
    const clientA = new TestHelpers({ dir: TEST_DIR, page });
    const clientB = new TestHelpers({ dir: TEST_DIR, page: guestPage });

    // Both clients join the same chat.
    await clientA.openChat(SHARED_CHAT);
    await clientB.openChat(SHARED_CHAT);

    // Each client makes a different persona + model selection.
    await clientA.selectPersona(FixturePersona.EchoConfig);
    await clientA.waitForControls();
    await clientA.setControl('Model', 'Claude Opus');

    await clientB.selectPersona(FixturePersona.Models);
    await clientB.waitForControls();
    await clientB.setControl('Model', 'Model Gamma');

    // Selection isolation: each client's picker + model control reflect only
    // its own choice, and never the other's.
    await expect(clientA.personaPicker).toContainText('Echo Config Persona');
    await expect(clientA.control('Model')).toContainText('Claude Opus');
    await expect(clientB.personaPicker).toContainText('Models Persona');
    await expect(clientB.control('Model')).toContainText('Model Gamma');

    // A selecting Opus did not push a model onto B's control, and vice versa.
    await expect(clientA.control('Model')).not.toContainText('Model Gamma');
    await expect(clientB.control('Model')).not.toContainText('Claude Opus');

    // Routing: each client's message reaches the persona that client selected.
    const replyA = await clientA.sendMessage('which model?');
    expect(replyA).toContain('applied model: claude-opus');

    const replyB = await clientB.sendMessage('which model?');
    expect(replyB).toContain('applied model: gamma');
  });
});
