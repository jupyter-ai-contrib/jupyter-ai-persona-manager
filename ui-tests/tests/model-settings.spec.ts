/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'model-settings';
const PERSONAS = [FixturePersona.EchoConfig];

/**
 * Verifies model-config settings (advertised in `ModelConfiguration.settings`)
 * render as their own control next to the model picker, and that selecting one
 * applies it via `update_model_settings` — the echoed reply proves it.
 *
 * The Echo Config fixture advertises one model setting, "Thinking"
 * (low/medium/high, current medium), alongside its Model picker.
 */
test.describe('model-settings', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('renders the model setting next to the model picker', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();

    // Both the Model picker and the Thinking model setting render as controls.
    await expect(helpers.control('Model')).toBeVisible();
    await expect(helpers.control('Thinking')).toBeVisible();
    // On its default it shows the current value's name.
    await expect(helpers.control('Thinking')).toContainText('Medium');
  });

  test('selecting a model setting applies it via update_model_settings', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();
    await helpers.setControl('Thinking', 'High');

    const reply = await helpers.sendMessage('show config');
    expect(reply).toContain('applied thinking: high');
    // The general settings were left on their defaults.
    expect(reply).toContain('applied effort: (default)');
    expect(reply).toContain('applied model: (default)');
  });
});
