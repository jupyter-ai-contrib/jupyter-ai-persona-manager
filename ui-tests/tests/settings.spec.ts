/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'settings';
const PERSONAS = [FixturePersona.EchoConfig];

/**
 * Verifies general settings (advertised via `report_settings_configuration`)
 * render as their own controls, in advertised order, and that a selection
 * round-trips through `update_settings`.
 *
 * The Echo Config fixture advertises two general settings in this order:
 *   1. "Effort" — a multi-value select (low/medium/high, current medium)
 *   2. "Verbose" — a two-option (boolean-style) select (on/off, current off)
 * Both are separate from the model settings, which render nearer the model.
 */
test.describe('settings', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('renders general settings as separate controls in advertised order', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();

    // Both general settings render; "Effort" comes before "Verbose".
    await expect(helpers.control('Effort')).toBeVisible();
    await expect(helpers.control('Verbose')).toBeVisible();
    const labels = await helpers.chat
      .locator(
        '.jp-jai-personaControls-controls > .jp-jai-personaControls-control-btn'
      )
      .evaluateAll(els => els.map(el => el.getAttribute('title')));
    expect(labels.indexOf('Effort')).toBeLessThan(labels.indexOf('Verbose'));
  });

  test('a multi-value select round-trips via update_settings', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();
    await helpers.setControl('Effort', 'High');

    const reply = await helpers.sendMessage('show config');
    expect(reply).toContain('applied effort: high');
    expect(reply).toContain('applied verbose: (default)');
  });

  test('a boolean-style select round-trips via update_settings', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();
    await helpers.setControl('Verbose', 'On');

    const reply = await helpers.sendMessage('show config');
    expect(reply).toContain('applied verbose: on');
    expect(reply).toContain('applied effort: (default)');
  });
});
