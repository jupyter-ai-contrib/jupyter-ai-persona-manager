/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { TestHelpers } from './test-helpers';

// No backend personas are installed: this suite exercises the frontend-only
// registration path (`registerFrontendPersona`), which is how JupyterLite
// extensions contribute personas when there is no server.
const TEST_DIR = 'frontend-persona';

test.describe('frontend-persona', () => {
  test('registers a frontend persona and shows it in the picker', async ({
    page,
    baseURL
  }) => {
    // In JupyterLite, autoGoto is disabled so galata doesn't apply its
    // JupyterLab URL matcher. Navigate manually and wait for the app.
    if (process.env.JAI_E2E_SUITE === 'jupyterlite') {
      // Galata's goto calls hookHelpersUp() which requires window.galata (galata's
      // own JupyterLab extension). That extension is not installed in JupyterLite,
      // so we bypass the galata wrapper and call the underlying Playwright page.
      await (page as any).page.goto(`${baseURL}/lab/index.html`);
      await page.waitForFunction(
        () => {
          const app = (window as any).jupyterapp;
          return app?.commands.hasCommand('jupyterlab-chat:open');
        },
        {
          timeout: 30000
        }
      );
    }

    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Inject a fake frontend persona directly via the session registry,
    // mimicking what a JupyterLite extension does via registerFrontendPersona.
    // Return chatId so the unregister step doesn't need to re-derive it from
    // currentWidget (which shifts after the Escape keypress closes the menu).
    const chatId = await page.evaluate(async () => {
      const app = (window as any).jupyterapp;
      const plugins: Map<string, any> = app.pluginRegistry._plugins;
      const sessionRegistry = plugins.get(
        '@jupyter-ai/persona-manager:session-registry'
      )?.service;
      // model.id is set asynchronously; await model.ready to get the stable id.
      const id = await (app.shell.currentWidget as any)?.model?.ready;
      if (!sessionRegistry || !id) {
        throw new Error(
          `Missing: sessionRegistry=${sessionRegistry}, chatId=${id}`
        );
      }
      sessionRegistry.registerFrontendPersona(id, {
        id: 'test-frontend-persona',
        name: 'Test Frontend Persona'
      });
      return id;
    });

    await expect(helpers.personaPicker).toBeVisible({ timeout: 10000 });
    await helpers.personaPicker.click();
    await expect(
      page.getByRole('menuitem', { name: 'Test Frontend Persona' })
    ).toBeVisible();
    await page.keyboard.press('Escape');

    // Unregister the persona and verify it disappears.
    await page.evaluate((id: string) => {
      const app = (window as any).jupyterapp;
      const plugins: Map<string, any> = app.pluginRegistry._plugins;
      const sessionRegistry = plugins.get(
        '@jupyter-ai/persona-manager:session-registry'
      )?.service;
      sessionRegistry.unregisterFrontendPersona(id, 'test-frontend-persona');
    }, chatId);

    if (await helpers.personaPicker.isVisible()) {
      // Other personas remain — the unregistered one must be gone from the menu.
      await helpers.personaPicker.click();
      await expect(
        page.getByRole('menuitem', { name: 'Test Frontend Persona' })
      ).not.toBeVisible();
      await page.keyboard.press('Escape');
    } else {
      // It was the only persona — the picker itself must have disappeared.
      await expect(helpers.personaPicker).not.toBeVisible();
    }
  });
});
