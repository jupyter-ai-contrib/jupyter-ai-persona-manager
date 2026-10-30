"""Regression tests: stopping a chat's PersonaManager must unsubscribe it from
Jupyter Events, so it no longer handles events and can be garbage-collected."""

import asyncio
import gc
import logging
import types
import weakref
from unittest.mock import MagicMock

import pytest
from jupyter_events import EventLogger
from jupyterlab_chat.events import CHAT_ROOM_EVENT_SCHEMA, CHAT_ROOM_EVENT_SCHEMA_ID

from jupyter_ai_persona_manager.base_persona import BasePersona
from jupyter_ai_persona_manager.extension import PersonaManagerExtension
from jupyter_ai_persona_manager.persona_events import (
    PERSONA_SELECTED_EVENT_SCHEMA_ID,
    register_persona_event_schemas,
)
from jupyter_ai_persona_manager.persona_manager import PersonaManager

CHAT_ID = "chat-1"


def _listeners(logger, schema_id):
    return len(logger._modified_listeners.get(schema_id, ())) + len(
        logger._unmodified_listeners.get(schema_id, ())
    )


@pytest.fixture
def logger():
    logger = EventLogger()
    logger.register_event_schema(CHAT_ROOM_EVENT_SCHEMA)
    register_persona_event_schemas(logger)
    return logger


@pytest.fixture
def extension():
    """The state `PersonaManagerExtension._stop_persona_manager` reads."""
    settings = {"jupyter-ai": {"persona-managers": {}}}
    return types.SimpleNamespace(
        _stop_lock=asyncio.Lock(),
        _stopping_chats={},
        log=logging.getLogger("test-stop"),
        serverapp=types.SimpleNamespace(web_app=types.SimpleNamespace(settings=settings)),
    )


@pytest.fixture
def open_chat(logger, extension, tmp_path, monkeypatch):
    """Opens the chat: creates its PersonaManager with one mock persona and
    registers it as the chat's live manager, as the extension does."""
    monkeypatch.setattr(PersonaManager, "_ep_persona_classes", [])
    managers = extension.serverapp.web_app.settings["jupyter-ai"]["persona-managers"]

    def _open_chat():
        chat = MagicMock()
        chat.get_id.return_value = CHAT_ID
        chat.get_path.return_value = "test.chat"
        chat.get_metadata.return_value = {}
        pm = PersonaManager(
            chat=chat,
            root_dir=str(tmp_path),
            event_loop=asyncio.get_running_loop(),
            event_logger=logger,
        )
        persona = MagicMock(spec=BasePersona)
        persona.id = "p1"
        pm._personas = {"p1": persona}
        managers[CHAT_ID] = pm
        return pm, persona

    return _open_chat


async def _stop_chat(extension):
    await PersonaManagerExtension._stop_persona_manager(extension, CHAT_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_error", [None, RuntimeError("shutdown failed")])
async def test_stopping_chat_removes_listeners(
    logger, extension, open_chat, shutdown_error
):
    _, persona = open_chat()
    persona.shutdown.side_effect = shutdown_error
    assert _listeners(logger, PERSONA_SELECTED_EVENT_SCHEMA_ID) == 1
    assert _listeners(logger, CHAT_ROOM_EVENT_SCHEMA_ID) == 1

    await _stop_chat(extension)

    assert _listeners(logger, PERSONA_SELECTED_EVENT_SCHEMA_ID) == 0
    assert _listeners(logger, CHAT_ROOM_EVENT_SCHEMA_ID) == 0
    persona.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_reopened_chat_prepares_only_the_live_manager(logger, extension, open_chat):
    _, stale_persona = open_chat()
    await _stop_chat(extension)
    _, live_persona = open_chat()

    logger.emit(
        schema_id=PERSONA_SELECTED_EVENT_SCHEMA_ID,
        data={"chat_id": CHAT_ID, "persona_id": "p1"},
    )
    await logger.gather_listeners()

    live_persona._ensure_prepared.assert_called_once()
    stale_persona._ensure_prepared.assert_not_called()


@pytest.mark.asyncio
async def test_no_stale_manager_left_after_chat_stops(logger, extension, open_chat):
    """
    Each chat websocket close stops that chat's manager through
    `PersonaManagerExtension._stop_persona_manager`, and the reconnect creates a
    new one. After N such cycles only the live manager may remain: the stopped
    ones must be unsubscribed from the EventLogger and garbage-collectable.
    """
    refs = [weakref.ref(open_chat()[0])]
    for _ in range(5):
        await _stop_chat(extension)
        refs.append(weakref.ref(open_chat()[0]))
    gc.collect()

    live = [r for r in refs if r() is not None]
    assert live == [refs[-1]], f"{len(live) - 1} stale PersonaManager(s) still alive"
    assert _listeners(logger, PERSONA_SELECTED_EVENT_SCHEMA_ID) == 1
    assert _listeners(logger, CHAT_ROOM_EVENT_SCHEMA_ID) == 1
