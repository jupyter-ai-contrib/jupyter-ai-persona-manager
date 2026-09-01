"""Fixture persona for the ``mcp-integration`` E2E suite.

On each message it connects to the built-in Jupyter MCP server using the config
from ``get_mcp_settings()`` — the same config the ACP client hands the agent —
calls the probe server's ``echo_headers`` tool, and replies with the headers
the server received as JSON. The suite asserts the reply carries the identity
headers (``X-Jupyter-Chat-Id``, ``X-JupyterAI-Persona-Id``) that persona-manager
stamps, proving they reach a real MCP server over the wire.

This fixture is only installed by the ``mcp-integration`` suite, whose env has
``mcp`` and ``fastmcp`` available; the other suites never load it.
"""

import httpx2
import json
import os

from jupyter_ai_persona_manager import BasePersona, McpServerHttp, PersonaDefaults
from jupyterlab_chat.models import Message
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


class McpProbePersona(BasePersona):
    """Test-only persona that reports the headers the built-in MCP server saw."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="MCP Probe Persona",
            description="Reports the headers reaching the built-in MCP server.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        settings = self.get_mcp_settings()
        server = next(
            s for s in settings.mcp_servers
            if isinstance(s, McpServerHttp) and s.name == "MCP Probe Server"
        )
        http_client = httpx2.AsyncClient(
            headers={header.name: header.value for header in server.headers}
        )
        async with (
            streamable_http_client(server.url, http_client=http_client) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("echo_headers", {})
        received = (result.structuredContent or {}).get("headers", {})
        self.send_message(json.dumps(received))
