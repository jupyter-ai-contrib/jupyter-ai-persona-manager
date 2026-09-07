/**
 * Configuration for Playwright using default from @jupyterlab/galata.
 *
 * A single test server serves every suite. Each spec declares and installs the
 * fixture personas it needs into its own working directory (see
 * tests/test-helpers.ts and AGENTS.md); the PersonaManager loads the nearest
 * `.jupyter/personas/` walking up from a chat's directory, so suites stay
 * isolated on one server.
 */
const path = require('path');
const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

// Random port so a run doesn't collide with a dev server (or another run) on a
// fixed port. Playwright re-`require`s this config in each worker, so compute it
// once and pin it into the environment — a fresh random value per reload would
// desync the server's port from the port the test workers connect to.
if (!process.env.JAI_TEST_PORT) {
  process.env.JAI_TEST_PORT = String(8989 + Math.floor(Math.random() * 900));
}
const PORT = Number(process.env.JAI_TEST_PORT);

// The `mcp-integration` suite (nox env `mcp`) runs a real FastMCP server as the
// built-in MCP server on this port and verifies the identity headers reach it.
if (!process.env.JAI_MCP_PROBE_PORT) {
  process.env.JAI_MCP_PROBE_PORT = String(PORT + 200);
}
const SUITE = process.env.JAI_E2E_SUITE;
const isLite = SUITE === 'jupyterlite';
const isMcp = SUITE === 'mcp';

module.exports = {
  ...baseConfig,
  // Route to the suite under test.
  ...(isMcp
    ? { testDir: 'tests/mcp-integration' }
    : isLite
      ? { testMatch: ['**/frontend-persona.spec.ts'] }
      : { testIgnore: ['**/mcp-integration/**'] }),
  use: {
    ...(baseConfig.use || {}),
    baseURL: `http://localhost:${PORT}`,
    // In JupyterLite, disable galata's auto-navigation (it expects the
    // JupyterLab URL pattern and fails for a static JupyterLite site).
    // The test navigates manually via page.goto() instead.
    ...(isLite ? { autoGoto: false } : {})
  },
  webServer: {
    // Serve Jupyterlite or Jupyterlab, depending on the test suite.
    // MCP port offset from the HTTP port so it doesn't collide with a
    // default (3001) or a dev server. CLI args win over galata's defaults.
    command: isLite
      ? `python -m http.server ${PORT} --directory ${path.resolve(__dirname, '..', '_output')}`
      : `jlpm start --ServerApp.port=${PORT} --MCPExtensionApp.mcp_port=${PORT + 100}`,
    url: `http://localhost:${PORT}/lab`,
    timeout: 120 * 1000,
    // Forward the suite + probe port to the server process so the config can
    // enable the FastMCP probe and point the built-in MCP server at it.
    env: {
      ...process.env,
      JAI_MCP_PROBE_PORT: process.env.JAI_MCP_PROBE_PORT,
      ...(SUITE ? { JAI_E2E_SUITE: SUITE } : {})
    },
    // Never reuse an already-running server: reusing an unrelated dev server
    // would leave the E2E persona-disabling config unapplied. Free the port
    // before running locally.
    reuseExistingServer: false
  }
};
