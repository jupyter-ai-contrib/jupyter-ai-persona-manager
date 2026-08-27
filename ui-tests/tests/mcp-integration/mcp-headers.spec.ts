/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from '../test-helpers';

// This suite's working directory and the fixture persona installed into it.
const TEST_DIR = 'mcp-integration';
const PERSONAS = [FixturePersona.McpProbe];

/**
 * End-to-end check that the identity headers persona-manager stamps onto MCP
 * servers actually reach a real MCP server. The suite runs a FastMCP server
 * (mcp_probe_ext.py) as the built-in Jupyter MCP server; the MCP Probe persona
 * connects to it with the config from `get_mcp_settings()` — the same config the
 * ACP client hands the agent — and replies with the headers the server received.
 * We assert the reply carries this chat's id and the persona's id under the
 * identity headers.
 *
 * Runs only in the `mcp` nox env (JAI_E2E_SUITE=mcp), where fastmcp and mcp are
 * installed and the probe server is configured as the built-in MCP server.
 */
test.describe('mcp integration', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('identity headers reach the built-in MCP server', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await helpers.selectPersona(FixturePersona.McpProbe);

    // The persona replies with the JSON of the headers the MCP server received.
    const reply = await helpers.sendMessage('hi');
    const received = JSON.parse(reply.trim());

    // get_http_headers() lowercases keys and preserves custom x- headers.
    expect(received['x-jupyter-chat-id']).toBeTruthy();
    expect(received['x-jupyterai-persona-id']).toContain('McpProbePersona');
  });
});
