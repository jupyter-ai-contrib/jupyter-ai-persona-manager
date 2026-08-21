/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'permissions';
const PERSONAS = [FixturePersona.Permission];

/**
 * Verifies the general permission API end-to-end.
 *
 * The Permission fixture calls `BasePersona.request_permission`, which reflects
 * the request in the chat as a `permission_request` metadata block. The
 * persona-manager frontend renders Allow/Deny buttons from it. Clicking a button
 * emits a `permission_response` Jupyter Event (client -> server, POST
 * /api/events); the manager routes it by (room_id, persona_id, request_id) to
 * the persona, resolving the suspended `request_permission`. The persona then
 * posts "decision: <option_id>", so the reply proves the round-trip.
 *
 * This is the generalized replacement for the ACP client's REST-based flow, and
 * behaves exactly like it from the user's point of view: buttons appear in chat,
 * the click resolves them, and the agent proceeds.
 */
test.describe('permissions', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('allowing a permission request resolves it and resumes the persona', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Permission);

    await helpers.send('do the thing');

    // The request is reflected in the chat with its detail + buttons.
    await helpers.waitForPermissionButtons();
    await expect(helpers.permissionRequest.first()).toContainText(
      'Approve action?'
    );
    await expect(helpers.permissionRequest.first()).toContainText(
      'permission-fixture-detail'
    );

    // Click Allow: the frontend emits the decision event, the persona resumes.
    await helpers.clickPermission('Allow');

    await helpers.waitForMessageContaining('decision: allow');
  });

  test('denying a permission request forwards the chosen option', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.Permission);

    await helpers.send('do the other thing');
    await helpers.waitForPermissionButtons();
    await helpers.clickPermission('Deny');

    await helpers.waitForMessageContaining('decision: deny');
  });
});
