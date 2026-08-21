/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it.
const TEST_DIR = 'permissions';
const PERSONAS = [
  FixturePersona.Permission,
  FixturePersona.QuietPermission,
  FixturePersona.VerbosePermission,
  FixturePersona.StopPermission
];

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

  test('a persona can raise two permission requests at once (no preamble)', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.QuietPermission);

    await helpers.send('go');

    // Both requests render (each in its own auto-created message).
    await helpers.waitForPermissionButtons();
    await expect
      .poll(async () => helpers.permissionRequestCount(), { timeout: 30000 })
      .toBe(2);

    // Approve both; the persona resumes once both resolve.
    await helpers.approveAll('Allow');

    await helpers.waitForMessageContaining('quiet decisions: allow, allow');
  });

  test('two permission requests can share one message without clobbering', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.VerbosePermission);

    await helpers.send('go');

    // The persona writes "Sure", then attaches both requests to that message.
    await helpers.waitForMessageContaining('Sure');
    await helpers.waitForPermissionButtons();
    await expect
      .poll(async () => helpers.permissionRequestCount(), { timeout: 30000 })
      .toBe(2);

    // Both requests coexist (not clobbered); approve both and the persona
    // resumes with both decisions. (The same-message, no-clobber guarantee is
    // pinned precisely by the unit test test_multiple_requests_same_message.)
    await helpers.approveAll('Allow');

    await helpers.waitForMessageContaining('verbose decisions: allow, allow');
  });

  test('clicking stop cancels a pending permission request', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.StopPermission);

    await helpers.send('go');

    // The request is pending and the persona is writing, so stop is enabled.
    await helpers.waitForPermissionButtons();
    await helpers.waitForWriting();

    // Stop: the persona-manager cancel flow cancels the pending request.
    await helpers.clickStop();

    await helpers.waitForMessageContaining('stop decision: cancelled');
    // The pending buttons are gone once the request resolves as cancelled.
    await expect
      .poll(async () => helpers.permissionButtonCount(), { timeout: 30000 })
      .toBe(0);
  });
});
