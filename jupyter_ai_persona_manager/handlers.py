import asyncio
import json
import mimetypes
import os
import time
import uuid
from urllib.parse import unquote

from jupyter_server.base.handlers import JupyterHandler
from jupyter_ydoc.ybasedoc import YBaseDoc
from jupyterlab_chat.models import Message, User
from jupyterlab_chat.ychat import YChat
import tornado


# Maximum avatar file size (5MB)
MAX_AVATAR_SIZE = 5 * 1024 * 1024
DEFAULT_SENDER = "user"
DEFAULT_SENDER_NAME = "User"
DEFAULT_RESPONSE_TIMEOUT = 120.0

# Module-level cache: {persona_id: avatar_path}
# This is populated when personas are initialized/refreshed
_avatar_cache: dict[str, str] = {}


def build_avatar_cache(persona_managers: dict) -> None:
    """
    Build the avatar cache from all persona managers.

    This should be called when personas are initialized or refreshed.
    """
    global _avatar_cache
    _avatar_cache = {}

    for room_id, persona_manager in persona_managers.items():
        for persona in persona_manager.personas.values():
            try:
                avatar_path = persona.defaults.avatar_path
                if avatar_path and os.path.exists(avatar_path):
                    _avatar_cache[persona.id] = avatar_path
            except Exception:
                # Skip personas with invalid avatar paths
                continue


def clear_avatar_cache() -> None:
    """Clear the avatar cache. Called during persona refresh."""
    global _avatar_cache
    _avatar_cache = {}


class MessageHandler(JupyterHandler):
    """
    Handler to receive a persona ID and a message, route it through a temporary PersonaManager,
    and return the persona's response.
    """

    @tornado.web.authenticated
    async def post(self, persona_name: str):
        try:
            data = json.loads(self.request.body)
            persona_name = unquote(persona_name)
            message_text = data.get("message")
            if not persona_name or not message_text:
                raise tornado.web.HTTPError(400, "Missing 'persona' or 'message' field")
            metadata = data.get("metadata")
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Invalid JSON body: {e}")

        serverapp = self.serverapp
        settings = serverapp.web_app.settings
        contents_manager = serverapp.contents_manager
        root_dir = getattr(contents_manager, "root_dir", "")
        base_url = settings.get("base_url", "/")

        # Mint a unique, throwaway chat for this one request. We do NOT depend on
        # a file_id_manager here: the room id is a plain uuid and the chat model
        # is obtained from the ChatManager (transport-neutral) when available,
        # falling back to an in-memory YChat otherwise.
        uid = uuid.uuid4()
        temp_path = f"{uid}.chat"
        temp_room_id = f"text:chat:{uid}"

        chat_manager = settings.get("chat_manager")
        ychat = None
        if chat_manager is not None:
            try:
                ychat = await chat_manager.create(temp_path)
            except Exception:
                self.log.warning(
                    "ChatManager.create failed for temporary chat; "
                    "falling back to an in-memory YChat.",
                    exc_info=True,
                )
        if ychat is None:
            ychat = YChat()

        from .persona_manager import PersonaManager

        loop = asyncio.get_running_loop()
        persona_manager = PersonaManager(
            room_id=temp_room_id,
            chat=ychat,
            fileid_manager=settings.get("file_id_manager"),
            root_dir=root_dir,
            event_loop=loop,
            base_url=base_url,
        )
        try:
            target_persona = next(
                (
                    p
                    for p in persona_manager.personas.values()
                    if getattr(p, "name", None) == persona_name
                ),
                None,
            )
            if not target_persona:
                raise tornado.web.HTTPError(404, f"Persona '{persona_name}' not found")

            msg = Message(
                id="msgid",
                body=message_text,
                time=time.time(),
                sender=User(username=DEFAULT_SENDER,
                            name=DEFAULT_SENDER_NAME,
                            display_name=DEFAULT_SENDER_NAME).username,
                raw_time=False,
                metadata=metadata
            )

            await target_persona.process_message(msg)
            # Streaming personas may still be working after process_message
            # returns. Wait until the persona is no longer processing, up to the
            # response timeout.
            deadline = loop.time() + DEFAULT_RESPONSE_TIMEOUT
            while target_persona.processing:
                if loop.time() > deadline:
                    self.log.warning("Timeout waiting for persona to finish")
                    break
                await asyncio.sleep(0.05)

            # Return the captured response
            response = "".join(
                m.body if getattr(m, "body", None) is not None else str(m)
                for m in ychat.get_messages()
            )
        finally:
            # Always tear down the throwaway manager, even on error/timeout.
            try:
                await persona_manager.shutdown_personas()
            except Exception:
                self.log.warning(
                    "Failed to shut down temporary PersonaManager", exc_info=True
                )

        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({"response": response}))


class CancelHandler(JupyterHandler):
    """
    Handler to cancel personas' in-progress responses in a chat.

    The frontend POSTs here (with the chat's path as a query parameter) when the
    user interrupts. Each persona in the chat is asked to stop via
    `BasePersona.cancel_response()`, which halts whatever its reply set in motion
    (a model stream, an agent turn, pending tool calls). Backend-agnostic: a
    persona with nothing cancellable inherits the base no-op.
    """

    @property
    def file_id_manager(self):
        manager = self.serverapp.web_app.settings.get("file_id_manager")
        if manager is None:
            raise tornado.web.HTTPError(500, "file_id_manager is not available")
        return manager

    @tornado.web.authenticated
    async def post(self):
        chat_path = self.get_argument("chat_path", None)
        if not chat_path:
            raise tornado.web.HTTPError(
                400, "chat_path is required as a URL query parameter"
            )

        persona_managers = self.serverapp.web_app.settings.get(
            "jupyter-ai", {}
        ).get("persona-managers", {})

        # The router registers each PersonaManager under the room_id it supplies,
        # which is the chat's path in RTC-free mode and `text:chat:{file_id}`
        # under RTC. Resolve the path first (RTC-free), then fall back to the RTC
        # room_id, so cancellation works regardless of transport.
        persona_manager = persona_managers.get(chat_path)
        if persona_manager is None:
            file_id = self.file_id_manager.get_id(chat_path)
            if file_id:
                persona_manager = persona_managers.get(f"text:chat:{file_id}")

        if not persona_manager:
            raise tornado.web.HTTPError(404, f"Chat not initialized: {chat_path}")

        cancelled = []
        for persona in persona_manager.personas.values():
            # Only interrupt personas that are actually processing a response;
            # cancelling an idle persona may be out of spec for some backends
            # (e.g. ACP's session/cancel is defined only for an ongoing turn).
            if not persona.processing:
                continue
            try:
                await persona.cancel_response()
                cancelled.append(persona.id)
            except Exception:
                self.log.warning(
                    f"Failed to cancel response for persona '{persona.id}'",
                    exc_info=True,
                )
        self.finish(json.dumps({"status": "cancelled", "cancelled": cancelled}))


class AvatarHandler(JupyterHandler):
    """
    Handler for serving persona avatar files.

    Looks up avatar files by persona ID and serves the image file
    with appropriate content-type headers.
    """

    @tornado.web.authenticated
    async def get(self, persona_id: str):
        """Serve an avatar file by persona ID."""
        # URL-decode the persona ID
        persona_id = unquote(persona_id)

        # Get the avatar file path
        avatar_path = self._find_avatar_file(persona_id)

        if avatar_path is None:
            raise tornado.web.HTTPError(404, f"Avatar not found for persona")

        # Check file size
        try:
            file_size = os.path.getsize(avatar_path)
            if file_size > MAX_AVATAR_SIZE:
                self.log.error(f"Avatar file too large: {file_size} bytes (max: {MAX_AVATAR_SIZE})")
                raise tornado.web.HTTPError(413, "Avatar file too large")
        except OSError as e:
            self.log.error(f"Error checking avatar file size: {e}")
            raise tornado.web.HTTPError(500, "Error accessing avatar file")

        # Serve the file
        try:
            # Set content type based on file extension
            content_type, _ = mimetypes.guess_type(avatar_path)
            if content_type:
                self.set_header("Content-Type", content_type)

            # Read and serve the file
            with open(avatar_path, 'rb') as f:
                content = f.read()
                self.write(content)

            await self.finish()
        except Exception as e:
            self.log.error(f"Error serving avatar file: {e}")
            raise tornado.web.HTTPError(500, f"Error serving avatar file: {str(e)}")

    def _find_avatar_file(self, persona_id: str) -> str | None:
        """
        Find the avatar file path by persona ID using the module-level cache.

        The cache is built when personas are initialized or refreshed,
        so this is an O(1) lookup instead of iterating all personas.
        """
        return _avatar_cache.get(persona_id)
