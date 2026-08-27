"""Test-only Jupyter Server extension for the ``mcp-integration`` E2E suite.

Runs a real FastMCP server inside the Jupyter Server process, exposing a single
``echo_headers`` tool that returns the HTTP headers the request arrived with.
The suite configures this server as the PersonaManager's built-in MCP server
(see ``jupyter_server_test_config.py``), so a fixture persona connecting to it
with the config from ``get_mcp_settings()`` — the same config the ACP client
hands the agent — can read back the identity headers persona-manager stamped,
proving they reach a real MCP server over the wire.

Not for production use.
"""

import asyncio
import os

import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

MCP_PROBE_PORT = int(os.environ.get("JAI_MCP_PROBE_PORT", "3999"))

mcp = FastMCP("jai-mcp-probe")


@mcp.tool
async def echo_headers() -> dict:
    """Return the HTTP headers this MCP request arrived with."""
    return {"headers": dict(get_http_headers())}


class _EmbeddedServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # don't touch Jupyter's signals
        pass


def _jupyter_server_extension_points():
    return [{"module": "mcp_probe_ext"}]


def _load_jupyter_server_extension(server_app):
    async def _serve():
        app = mcp.http_app(transport="http")
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=MCP_PROBE_PORT,
            lifespan="on",
            log_level="warning",
        )
        await _EmbeddedServer(config).serve()

    server_app.io_loop.add_callback(lambda: asyncio.ensure_future(_serve()))
    server_app.log.info(
        "mcp_probe_ext: FastMCP on http://127.0.0.1:%s/mcp", MCP_PROBE_PORT
    )
