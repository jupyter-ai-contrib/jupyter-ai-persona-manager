import asyncio
import sys
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jupyterlab_commands_toolkit.tools import target_client_id

from jupyter_ai_persona_manager.client_routing import ClientRoutingMiddleware
from jupyter_ai_persona_manager.mcp_server_models import (
    CHAT_ID_HEADER,
    PERSONA_ID_HEADER,
)

GET_HTTP_HEADERS = "jupyter_ai_persona_manager.client_routing.get_http_headers"


def headers(persona_id="persona-1"):
    """The identity headers of a persona in chat `chat-1`."""
    return {CHAT_ID_HEADER.lower(): "chat-1", PERSONA_ID_HEADER.lower(): persona_id}


def persona(web_client_id=None, processing=True):
    """A persona processing a message sent by the given web client."""
    message = None
    if processing:
        metadata = {} if web_client_id is None else {"web_client_id": web_client_id}
        message = SimpleNamespace(metadata=metadata)
    return SimpleNamespace(processing_message=message)


def middleware(personas):
    """A middleware over a settings dict with one chat, `chat-1`."""
    manager = SimpleNamespace(personas=personas)
    settings = {"jupyter-ai": {"persona-managers": {"chat-1": manager}}}
    return ClientRoutingMiddleware(get_settings=lambda: settings)


async def bound_client_id(routing, request_headers):
    """Runs a tool call through the middleware and returns the bound client id."""

    async def call_next(context):
        return target_client_id.get()

    with patch(GET_HTTP_HEADERS, return_value=request_headers):
        return await routing.on_call_tool(None, call_next)


@pytest.mark.asyncio
async def test_routes_to_the_web_client_of_the_message():
    routing = middleware({"persona-1": persona("client-a")})
    assert await bound_client_id(routing, headers()) == "client-a"
    assert target_client_id.get() is None


@pytest.mark.asyncio
async def test_no_headers_targets_all_clients():
    routing = middleware({"persona-1": persona("client-a")})
    assert await bound_client_id(routing, {}) is None


@pytest.mark.asyncio
async def test_unknown_persona_targets_all_clients():
    routing = middleware({})
    assert await bound_client_id(routing, headers()) is None


@pytest.mark.asyncio
async def test_persona_not_processing_targets_all_clients():
    routing = middleware({"persona-1": persona(processing=False)})
    assert await bound_client_id(routing, headers()) is None


@pytest.mark.asyncio
async def test_concurrent_calls_are_isolated():
    routing = middleware(
        {"persona-1": persona("client-a"), "persona-2": persona("client-b")}
    )
    request_headers = ContextVar("request_headers")

    async def call(persona_id):
        request_headers.set(headers(persona_id))

        async def call_next(context):
            await asyncio.sleep(0.05)
            return target_client_id.get()

        return await routing.on_call_tool(None, call_next)

    with patch(GET_HTTP_HEADERS, side_effect=request_headers.get):
        results = await asyncio.gather(call("persona-1"), call("persona-2"))
    assert results == ["client-a", "client-b"]


@pytest.mark.asyncio
async def test_without_the_toolkit_the_call_passes_through():
    routing = middleware({"persona-1": persona("client-a")})

    async def call_next(context):
        return "result"

    with patch.dict(sys.modules, {"jupyterlab_commands_toolkit.tools": None}):
        assert await routing.on_call_tool(None, call_next) == "result"
