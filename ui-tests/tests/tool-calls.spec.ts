/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'tool-calls';
const PERSONAS = [FixturePersona.ToolCall];

/**
 * Verifies the general tool-call UI (BasePersona.report_tool_call): status
 * rendering, grouping, a file diff, and a permission attached to a tool-call
 * row. This is the generalized equivalent of ACP's tool-call rendering, which
 * had no E2E coverage of its own.
 */
test.describe('tool-calls', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('renders tool calls with status, diff, and an attached permission', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.ToolCall);

    await helpers.send('go');

    // Both tool calls render (read + edit), grouped in one message.
    await expect
      .poll(async () => helpers.toolCalls.count(), { timeout: 30000 })
      .toBe(2);
    await expect(
      helpers.chat.getByText('Reading example.py').first()
    ).toBeVisible();
    await expect(
      helpers.chat.getByText('Editing config.py').first()
    ).toBeVisible();

    // The edit's diff renders (path + the added line).
    await expect(helpers.diffBlocks.first()).toBeVisible();
    await expect(helpers.diffBlocks.first()).toContainText('config.py');
    await expect(helpers.diffBlocks.first()).toContainText('value = 2');

    // The permission is attached to the edit tool call — approve it.
    await helpers.waitForPermissionButtons();
    await helpers.clickPermission('Allow');

    await helpers.waitForMessageContaining('tool decision: allow');
    // Buttons are gone once resolved.
    await expect
      .poll(async () => helpers.permissionButtonCount(), { timeout: 30000 })
      .toBe(0);
  });

  test('denying the tool-call permission fails the tool call', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.ToolCall);

    await helpers.send('go');
    await helpers.waitForPermissionButtons();
    await helpers.clickPermission('Deny');

    await helpers.waitForMessageContaining('tool decision: denied');
  });
});
