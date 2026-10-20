"""
Route a persona's frontend commands to the web client that triggered them.

`jupyter-server-mcp` adds this middleware to the built-in MCP server through
the `jupyter_server_mcp.middleware` entry point. On each tool call, it reads
the identity headers that `BasePersona.get_mcp_settings()` stamps on the MCP
config, finds the persona, and binds the `web_client_id` of the message the
persona is processing to the `target_client_id` context variable of
`jupyterlab-commands-toolkit`. Without the toolkit, the headers, or a message,
the command runs on all web clients.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware

from .mcp_server_models import CHAT_ID_HEADER, PERSONA_ID_HEADER

WEB_CLIENT_ID_METADATA_KEY = "web_client_id"


class ClientRoutingMiddleware(Middleware):
    """
    Binds the target web client of the calling persona for the duration of
    each tool call.
    """

    def __init__(self, get_settings: Callable[[], dict] | None = None) -> None:
        # `get_settings` returns the Jupyter Server `web_app.settings` dict.
        # Injectable for testing; defaults to the running `ServerApp`.
        self._get_settings = get_settings

    def _settings(self) -> dict:
        if self._get_settings is not None:
            return self._get_settings()
        from jupyter_server.serverapp import ServerApp

        return ServerApp.instance().web_app.settings

    def _web_client_id(self) -> str | None:
        """
        Returns the web client id for the current tool call, or `None` to
        target all web clients.
        """
        headers = get_http_headers()
        chat_id = headers.get(CHAT_ID_HEADER.lower())
        persona_id = headers.get(PERSONA_ID_HEADER.lower())
        if not chat_id or not persona_id:
            return None
        managers = self._settings().get("jupyter-ai", {}).get("persona-managers", {})
        manager = managers.get(chat_id)
        persona = manager.personas.get(persona_id) if manager else None
        message = persona.processing_message if persona else None
        if message is None:
            return None
        return (message.metadata or {}).get(WEB_CLIENT_ID_METADATA_KEY)

    async def on_call_tool(self, context: Any, call_next: Callable) -> Any:
        try:
            from jupyterlab_commands_toolkit.tools import target_client_id
        except ImportError:
            return await call_next(context)

        token = target_client_id.set(self._web_client_id())
        try:
            return await call_next(context)
        finally:
            target_client_id.reset(token)
