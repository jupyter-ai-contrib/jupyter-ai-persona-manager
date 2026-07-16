/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it.
const TEST_DIR = 'model-selector';
const PERSONAS = [FixturePersona.Models, FixturePersona.EchoConfig];

/**
 * Verifies the model picker, driven by `report_model_configuration`:
 *
 *   - its options are exactly the advertised models, in advertised order;
 *   - the button shows the current model, and the "Default (…)" row names it;
 *   - selecting a model applies it — the reply, which echoes what `update_model`
 *     recorded, proves the apply path ran;
 *   - leaving the picker on Default sends a null model id, so `update_model`
 *     never fires and the reply reports "(default)".
 */
test.describe('model-selector', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('lists advertised models in order and marks the current default', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Models);
    await helpers.waitForControls();

    // The Models fixture advertises Alpha, Beta, Gamma with Beta current.
    // The button reflects the current model.
    await expect(helpers.control('Model')).toContainText('Model Beta');

    // The dropdown offers a "Default (…)" row naming the current model, then the
    // three models in advertised order.
    const options = await helpers.controlOptions('Model');
    expect(options).toEqual([
      'Default (Model Beta)',
      'Model Alpha',
      'Model Beta',
      'Model Gamma'
    ]);
  });

  test('selecting a model applies it (echoed reply proves update_model ran)', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Models);
    await helpers.waitForControls();
    await helpers.setControl('Model', 'Model Gamma');

    const reply = await helpers.sendMessage('which model?');
    expect(reply).toContain('applied model: gamma');
  });

  test('leaving the model on Default sends null (update_model does not fire)', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.EchoConfig);
    await helpers.waitForControls();

    // No model picked -> ModelSpec.id is null -> apply_model_spec skips
    // update_model, so the persona reports its default.
    const reply = await helpers.sendMessage('which model?');
    expect(reply).toContain('applied model: (default)');
  });
});
