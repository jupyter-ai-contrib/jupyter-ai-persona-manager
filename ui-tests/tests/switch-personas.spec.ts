/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it.
const TEST_DIR = 'switch-personas';
const PERSONAS = [FixturePersona.SwitchA, FixturePersona.SwitchB];

/**
 * Verifies a persona's model/settings selection is NOT carried over to another
 * persona when the user switches between them (issue #62).
 *
 * The controls (model, model settings, general settings) are per-persona: the
 * toolbar keys each persona's picks by persona ID and stamps only the *selected*
 * persona's selection onto an outgoing message's metadata. So a switch-then-send
 * must not stamp the previous persona's model/settings under the new
 * `to_persona` — doing so would apply the old persona's choices to the new one.
 *
 * The Switch A / Switch B fixtures are structurally identical echo-config
 * personas: each echoes what a selection *applied* (via the `update_*` hooks), so
 * a reply proves whether the apply path ran with an inherited value. The reply's
 * leading "persona: Switch A/B" line confirms which persona handled it.
 */
test.describe('switch-personas', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('switching personas does not carry over the previous selection', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Pick Switch A and give every control a non-default value.
    await helpers.selectPersona(FixturePersona.SwitchA);
    await helpers.waitForControls();
    await helpers.setControl('Model', 'A Two');
    await helpers.setControl('Thinking', 'High');
    await helpers.setControl('Effort', 'High');

    // Its toolbar reflects those selections.
    await expect(helpers.control('Model')).toContainText('A Two');
    await expect(helpers.control('Thinking')).toContainText('High');
    await expect(helpers.control('Effort')).toContainText('High');

    // Switch to Switch B *without* sending as A first. Its controls must render
    // fresh at B's own defaults — not A's picks (and A's "A Two" model isn't even
    // one of B's options, so a carried-over model id would be doubly wrong).
    await helpers.selectPersona(FixturePersona.SwitchB);
    await helpers.waitForControls();
    await expect(helpers.control('Model')).toContainText('B One');
    await expect(helpers.control('Thinking')).toContainText('Medium');
    await expect(helpers.control('Effort')).toContainText('Medium');

    // Send as B: the applied model/settings must all be defaults. A carried-over
    // selection would show up here as "a-two" / "high", proving the previous
    // persona's picks leaked into B's message metadata.
    const reply = await helpers.sendMessage('show config');
    expect(reply).toContain('persona: Switch B');
    expect(reply).toContain('applied model: (default)');
    expect(reply).toContain('applied thinking: (default)');
    expect(reply).toContain('applied effort: (default)');
    // And, defensively, none of A's selected values appear anywhere in it.
    expect(reply).not.toContain('a-two');
    expect(reply).not.toContain('applied thinking: high');
    expect(reply).not.toContain('applied effort: high');
  });

  test('each persona keeps its own selection across a switch back', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Give A a non-default model, then switch to B and give it a *different*
    // non-default model. Neither should see the other's pick.
    await helpers.selectPersona(FixturePersona.SwitchA);
    await helpers.waitForControls();
    await helpers.setControl('Model', 'A Two');

    await helpers.selectPersona(FixturePersona.SwitchB);
    await helpers.waitForControls();
    await expect(helpers.control('Model')).toContainText('B One');
    await helpers.setControl('Model', 'B Two');

    // Back to A: its own earlier pick is remembered, not B's.
    await helpers.selectPersona(FixturePersona.SwitchA);
    await helpers.waitForControls();
    await expect(helpers.control('Model')).toContainText('A Two');
    const replyA = await helpers.sendMessage('show config');
    expect(replyA).toContain('persona: Switch A');
    expect(replyA).toContain('applied model: a-two');

    // Back to B: its own pick is remembered, and A's never bled in.
    await helpers.selectPersona(FixturePersona.SwitchB);
    await helpers.waitForControls();
    await expect(helpers.control('Model')).toContainText('B Two');
    const replyB = await helpers.sendMessage('show config');
    expect(replyB).toContain('persona: Switch B');
    expect(replyB).toContain('applied model: b-two');
  });
});
