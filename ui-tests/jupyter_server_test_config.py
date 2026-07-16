"""Server configuration for integration tests.

!! Never use this configuration in production because it
opens the server to the world and provide access to JupyterLab
JavaScript objects through the global window variable.
"""
import os
from pathlib import Path

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)

# The HTTP port (--ServerApp.port) and MCP port (--MCPExtensionApp.mcp_port) are
# passed on the `jlpm start` command line (see playwright.config.js) — CLI args
# win over the defaults set above, so nothing to do here for ports.

# Uncomment to set server log level to debug level
# c.ServerApp.log_level = "DEBUG"

# No default persona: the persona list is exactly the fixtures each suite
# installs. Without this the manager advertises its built-in default ID
# (Jupyternaut) over PageConfig and the toolbar seeds that as the initial
# selection; since no such persona is installed here, an empty default keeps the
# picker starting at "No one" and leaves the suites in full control.
c.PersonaManager.default_persona_id = ""

# If jupyter-ai-acp-client happens to be installed alongside persona-manager
# (e.g. in the shared workbench env), its vendored ACP personas register as
# `jupyter_ai.personas` entry points and would load in *every* chat, polluting
# the deterministic fixture-only persona list. This env var makes that package's
# vendored personas raise `PersonaRequirementsUnmet` on import, which the
# PersonaManager treats as "skip" — so only the test fixtures load. It's a no-op
# when acp-client isn't installed (the case on persona-manager's own CI).
os.environ["JUPYTER_AI_ACP_CLIENT_E2E_TESTING_ONLY"] = "1"

# Each test suite installs the fixture personas it needs at runtime, into its
# own working directory (see tests/test-helpers.ts). The PersonaManager loads
# the nearest `.jupyter/personas/` walking up from a chat's directory, so a
# suite's chats see only its personas. The fixture personas locate their shared
# avatar asset via this env var, so they need no image file of their own.
os.environ["JAI_TEST_ASSETS_DIR"] = str(
    Path(__file__).parent.resolve() / "fixtures" / "assets"
)
