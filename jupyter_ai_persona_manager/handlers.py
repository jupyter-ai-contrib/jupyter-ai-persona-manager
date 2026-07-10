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
from pycrdt import Awareness
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
    """Handler to receive a persona ID and a message, route it through a
    temporary PersonaManager, and return the persona's response.

    metadata can contain a room argument which persists a room
    argument into a thread which remembers chats

    example
 ```
    %%ai @Jupyternaut -m '{"model_id": "openai/openclaw", "model_args": {"api_base": "http://openclaw:18789/v1"}, "room": "test1"}' 
 ```
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
            room = metadata.get("room")
        except Exception as e:
            raise tornado.web.HTTPError(400, f"Invalid JSON body: {e}")

        serverapp = self.serverapp
        fileid_manager = serverapp.web_app.settings.get("file_id_manager")
        contents_manager = serverapp.contents_manager
        root_dir = getattr(contents_manager, "root_dir", "")
        temp_room = False
        if room is None:
            room =  uuid.uuid4()
            temp_room = True
        elif metadata.get("model_args", {}).get("persistence") is None:
            metadata.setdefault("model_args", {}).setdefault("persistence", "true")

        room_uid = fileid_manager.index(os.path.join(root_dir, f"{room}"))
        room_id = f"text:chat:{room_uid}"
        router = self.serverapp.web_app.settings.get(
            "jupyter-ai", {}
        ).get("router")
        if router and room_id in router.active_chats:
            self.log.info(f"Found room_id {room_id} for {room}")
            ychat = router.active_chats[room_id]
        else:
            self.log.info(f"Create new room_id {room_id} for {room}")
            ychat = YChat()
            ychat.awareness = Awareness(ydoc=ychat._ydoc)
            ychat.set_id(room_id)
            router.connect_chat(room_id, ychat)

        persona_managers = self.serverapp.web_app.settings[
            "jupyter-ai"
        ]["persona-managers"]
        persona_manager = persona_managers.get(room_id)

        target_persona = next(
            (
                p
                for p in persona_manager.personas.values()
                if getattr(p, "name", None) == persona_name
            ),
            None,
        )
        if not target_persona:
            raise tornado.web.HTTPError(
                404, f"Persona '{persona_name}' not found"
            )
        msg_time = time.time()
        msg = Message(
            id="msgid",
            body=message_text,
            time=msg_time,
            sender=User(username=DEFAULT_SENDER,
                        name=DEFAULT_SENDER_NAME,
                        display_name=DEFAULT_SENDER_NAME).username,
            raw_time=False,
            metadata=metadata
        )

        done_event = asyncio.Event()
        await target_persona.process_message(msg)
        def on_awareness_change(event, *args, **kwargs):
            local_state = target_persona.awareness.get_local_state()
            if not local_state.get('isWriting', False):
                done_event.set()

        ychat.awareness.observe(on_awareness_change)

        try:
            # If currently writing, wait for the event that indicates it's done
            if target_persona.awareness.get_local_state().get(
                    'isWriting', False
            ):
                await asyncio.wait_for(
                    done_event.wait(), timeout=DEFAULT_RESPONSE_TIMEOUT
                )
        except asyncio.TimeoutError:
            self.log.warning("Timeout waiting for persona to finish writing")

        # Return the captured response
        response = "".join(
            msg.body if getattr(msg, "body", None) is not None else str(msg)
            for msg in ychat.get_messages()
            if getattr(msg, "time", None) and msg.time > msg_time
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

        file_id = self.file_id_manager.get_id(chat_path)
        if not file_id:
            raise tornado.web.HTTPError(404, f"Chat not found: {chat_path}")
        room_id = f"text:chat:{file_id}"

        persona_manager = (
            self.serverapp.web_app.settings.get("jupyter-ai", {})
            .get("persona-managers", {})
            .get(room_id)
        )
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
