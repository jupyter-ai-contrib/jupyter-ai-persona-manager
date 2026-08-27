/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'metadata-preservation';
const PERSONAS = [FixturePersona.MetadataEcho];

// Stands in for metadata another extension contributes to the shared chat
// input — e.g. jupyterlab-commands-toolkit stamps a `web_client_id` there to
// route frontend commands back to the web client that triggered them. It must
// survive the persona controls stamping their own metadata.
const THIRD_PARTY_KEY = 'third_party_key';
const THIRD_PARTY_VALUE = 'preserve-me';

/**
 * Guards that the persona controls merge their metadata onto the chat input
 * rather than replacing the whole map. The controls stamp `to_persona` (plus
 * model/settings) onto each outgoing message; if they clear the input's
 * metadata first, they wipe keys other extensions contributed.
 *
 * The Metadata Echo fixture replies with the metadata it received, so the keys
 * that actually reached the message are observable.
 */
test.describe('metadata-preservation', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('persona controls preserve metadata contributed by other extensions', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.MetadataEcho);
    await helpers.waitForControls();

    // Another extension stamps a key onto the shared chat input.
    await helpers.stampInputMetadata({ [THIRD_PARTY_KEY]: THIRD_PARTY_VALUE });

    // Changing a control makes the persona controls re-stamp their metadata.
    // This is precisely where a clear-then-set would wipe the third-party key;
    // a merge keeps it.
    await helpers.setControl('Mode', 'B');

    const reply = await helpers.sendMessage('echo metadata');
    // The third-party key survived the re-stamp...
    expect(reply).toContain(`${THIRD_PARTY_KEY}: ${THIRD_PARTY_VALUE}`);
    // ...alongside the routing metadata the persona controls contribute.
    expect(reply).toContain('to_persona');
  });
});
