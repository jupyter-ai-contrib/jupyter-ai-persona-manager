import json
import mimetypes
import os
from typing import Dict, Optional
from urllib.parse import unquote

from jupyter_server.base.handlers import JupyterHandler
import tornado


# Maximum avatar file size (5MB)
MAX_AVATAR_SIZE = 5 * 1024 * 1024

# Module-level cache: {persona_id: avatar_path}
# This is populated when personas are initialized/refreshed
_avatar_cache: Dict[str, str] = {}


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

    def _find_avatar_file(self, persona_id: str) -> Optional[str]:
        """
        Find the avatar file path by persona ID using the module-level cache.

        The cache is built when personas are initialized or refreshed,
        so this is an O(1) lookup instead of iterating all personas.
        """
        return _avatar_cache.get(persona_id)
