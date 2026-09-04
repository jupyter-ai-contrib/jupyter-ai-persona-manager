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
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Inject a fake frontend persona directly via the session registry,
    // mimicking what a JupyterLite extension does via registerFrontendPersona.
    await page.evaluate(() => {
      const app = window.jupyterapp as any;
      const plugins: Map<string, any> = app.pluginRegistry._plugins;
      const sessionRegistry = plugins.get(
        '@jupyter-ai/persona-manager:session-registry'
      )?.service;
      const chatId = (app.shell.currentWidget as any)?.model?.id;
      if (!sessionRegistry || !chatId) {
        throw new Error(
          `Missing: sessionRegistry=${sessionRegistry}, chatId=${chatId}`
        );
      }
      sessionRegistry.registerFrontendPersona(chatId, {
        id: 'test-frontend-persona',
        name: 'Test Frontend Persona'
      });
    });

    await expect(helpers.personaPicker).toBeVisible({ timeout: 10000 });
    await helpers.personaPicker.click();
    await expect(
      page.getByRole('menuitem', { name: 'Test Frontend Persona' })
    ).toBeVisible();
    await page.keyboard.press('Escape');
  });
});
