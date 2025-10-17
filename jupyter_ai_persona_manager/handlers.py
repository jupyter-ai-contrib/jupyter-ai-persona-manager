import json
import mimetypes
import os
from functools import lru_cache
from typing import Optional

from jupyter_server.base.handlers import JupyterHandler
import tornado


class RouteHandler(JupyterHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({
            "data": "This is /jupyter-ai-persona-manager/get-example endpoint!"
        }))


class AvatarHandler(JupyterHandler):
    """
    Handler for serving persona avatar files.

    Looks up avatar files through the PersonaManager to find the correct file path,
    then serves the file with appropriate content-type headers.
    """

    @tornado.web.authenticated
    async def get(self, filename: str):
        """Serve an avatar file by filename."""
        self.log.info(f"Avatar request for filename: {filename}")

        # Get the avatar file path from persona managers
        avatar_path = self._find_avatar_file(filename)

        if avatar_path is None:
            self.log.warning(f"Avatar file not found for filename: {filename}")
            raise tornado.web.HTTPError(404, f"Avatar file not found: {filename}")

        self.log.info(f"Found avatar at: {avatar_path}")

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
            self.log.error(f"Error serving avatar file {filename}: {e}")
            raise tornado.web.HTTPError(500, f"Error serving avatar file: {str(e)}")

    @lru_cache(maxsize=128)
    def _find_avatar_file(self, filename: str) -> Optional[str]:
        """
        Find the avatar file path by searching through all persona managers.

        Uses LRU cache to avoid repeated lookups for the same filename.
        """
        self.log.info(f"Looking up avatar file: {filename}")

        # Get all persona managers from settings
        persona_managers = self.settings.get('jupyter-ai', {}).get('persona-managers', {})
        self.log.info(f"Found {len(persona_managers)} persona managers")

        for room_id, persona_manager in persona_managers.items():
            self.log.info(f"Checking room {room_id} with {len(persona_manager.personas)} personas")
            # Check each persona's avatar path
            for persona in persona_manager.personas.values():
                try:
                    avatar_path = persona.defaults.avatar_path
                    avatar_filename = os.path.basename(avatar_path)
                    self.log.info(f"Persona {persona.name} avatar: {avatar_filename}")

                    if avatar_path and avatar_filename == filename:
                        # Found a match, return the absolute path
                        if os.path.exists(avatar_path):
                            self.log.info(f"Match found: {avatar_path}")
                            return avatar_path
                        else:
                            self.log.warning(f"Avatar file not found at path: {avatar_path}")
                except Exception as e:
                    self.log.warning(f"Error checking avatar for persona {persona.name}: {e}")
                    continue

        self.log.warning(f"No avatar found for filename: {filename}")
        return None
