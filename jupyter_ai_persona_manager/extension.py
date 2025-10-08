from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from jupyter_server.extension.application import ExtensionApp
from jupyter_server.serverapp import ServerApp
from jupyter_server_fileid.manager import BaseFileIdManager
from traitlets import Type
from traitlets.config import Config

from jupyter_ai_persona_manager.handlers import RouteHandler

from .persona_manager import PersonaManager

if TYPE_CHECKING:
    from typing import Any
    from asyncio import AbstractEventLoop
    from jupyterlab_chat.ychat import YChat


class PersonaManagerExtension(ExtensionApp):
    """
    Jupyter AI Persona Manager Extension
    
    This extension handles persona management for Jupyter AI chat interactions.
    It depends on jupyter-ai-router for message routing and coordination.
    """
    
    name = "jupyter_ai_persona_manager"
    handlers = [
        (r"jupyter-ai-persona-manager/health/?", RouteHandler)
    ]  # No direct HTTP handlers, works through router integration
    
    persona_manager_class = Type(
        klass=PersonaManager,
        default_value=PersonaManager,
        config=True,
        help="The `PersonaManager` class.",
    )
    
    def initialize(self, argv: Any = None) -> None:
        super().initialize()
        
    @property
    def event_loop(self) -> AbstractEventLoop:
        """
        Returns a reference to the asyncio event loop.
        """
        from asyncio import get_event_loop_policy
        return get_event_loop_policy().get_event_loop()
    
    def initialize_settings(self):
        """Initialize persona manager settings and router integration."""
        start = time.time()
        
        # Ensure 'jupyter-ai.persona-manager' is in `self.settings`, which gets
        # copied to `self.serverapp.web_app.settings` after this method returns
        if 'jupyter-ai' not in self.settings:
            self.settings['jupyter-ai'] = {}
        if 'persona-manager' not in self.settings['jupyter-ai']:
            self.settings['jupyter-ai']['persona-managers'] = {}
        
        # Set up router integration task
        self.event_loop.create_task(self._setup_router_integration())

        # Log server extension startup time
        self.log.info(f"Registered {self.name} server extension")
        startup_time = round((time.time() - start) * 1000)
        self.log.info(f"Initialized Persona Manager server extension in {startup_time} ms.")
    
    async def _setup_router_integration(self) -> None:
        """
        Set up integration with jupyter-ai-router.
        This allows persona manager to work through the centralized MessageRouter.
        """
        self.log.info("Waiting for the router to be ready")
        
        # Wait until the router field is available
        while True:
            router = self.serverapp.web_app.settings.get("jupyter-ai", {}).get("router")
            if router is not None:
                self.log.info("Router is ready, continuing with persona manager integration")
                break
            await asyncio.sleep(0.1)  # Check every 100ms
        
        # Wait for the 'jupyter-ai.persona-managers' dictionary to be available
        # in `self.serverapp.web_app.settings`. This will occur after
        # `initialize_settings()` returns.
        while self.serverapp.web_app.settings.get("jupyter-ai", {}).get("persona-managers") is None:
            self.log.warning("PersonaManagers dictionary not found, retrying in 100ms")
            await asyncio.sleep(0.1)

        try:
            self.log.info("Found jupyter-ai-router, registering persona manager callbacks")
            
            # Register callback for new chat initialization
            router.observe_chat_init(self._on_router_chat_init)
            
            # Store reference to router for later use
            self.router = router
            
        except Exception as e:
            self.log.error(f"Error setting up router integration: {e}")
    
    def _on_router_chat_init(self, room_id: str, ychat: "YChat") -> None:
        """
        Callback for when router detects a new chat initialization.
        This initializes persona manager for the new chat room.
        """
        self.log.info(f"Router detected new chat room, initializing persona manager: {room_id}")
        
        # Initialize persona manager for this chat
        persona_manager = self._init_persona_manager(room_id, ychat)
        if not persona_manager:
            self.log.error(
                "Jupyter AI was unable to initialize its AI personas. They are not available for use in chat until this error is resolved. "
                + "Please verify your configuration and open a new issue on GitHub if this error persists."
            )
            return
        
        # Cache the persona manager in server settings dictionary.
        #
        # NOTE: This must be added to `self.serverapp.web_app.settings`, not
        # `self.settings`. `self.settings` is a local dictionary that is only
        # copied to `self.serverapp.web_app.settings` immediately after
        # `self.initialize_settings` returns.
        persona_managers_by_room = self.serverapp.web_app.settings['jupyter-ai']['persona-managers']
        persona_managers_by_room[room_id] = persona_manager
        
        # Register persona manager callbacks with router
        self.router.observe_chat_msg(room_id, persona_manager.on_chat_message)
    
    def _init_persona_manager(
        self, room_id: str, ychat: "YChat"
    ) -> PersonaManager | None:
        """
        Initializes a `PersonaManager` instance scoped to a `YChat`.
        
        This method should not raise an exception. Upon encountering an
        exception, this method will catch it, log it, and return `None`.
        """
        persona_manager: PersonaManager | None = None
        
        try:
            assert self.serverapp
            assert self.serverapp.web_app
            assert self.serverapp.web_app.settings
            fileid_manager = self.serverapp.web_app.settings.get(
                "file_id_manager", None
            )
            assert isinstance(fileid_manager, BaseFileIdManager)
            
            contents_manager = self.serverapp.contents_manager
            root_dir = getattr(contents_manager, "root_dir", None)
            assert isinstance(root_dir, str)
            
            PersonaManagerClass = self.persona_manager_class
            persona_manager = PersonaManagerClass(
                parent=self,
                room_id=room_id,
                ychat=ychat,
                fileid_manager=fileid_manager,
                root_dir=root_dir,
                event_loop=self.event_loop,
            )
        except Exception as e:
            self.log.error(
                f"Unable to initialize PersonaManager in YChat with ID '{ychat.get_id()}' due to an exception printed below."
            )
            self.log.exception(e)
        finally:
            return persona_manager
    
    async def stop_extension(self):
        """
        Public method called by Jupyter Server when the server is stopping.
        """
        try:
            await self._stop_extension()
        except Exception as e:
            self.log.error("Persona Manager extension raised an exception while stopping:")
            self.log.exception(e)
    
    async def _stop_extension(self):
        """
        Private method that defines the cleanup code to run when the server is
        stopping.
        """
        # Clean up persona managers
        for room_id, persona_manager in self.persona_managers_by_room.items():
            try:
                await persona_manager.shutdown_personas()
            except Exception as e:
                self.log.error(f"Error cleaning up persona manager for room {room_id}: {e}")
        
        self.persona_managers_by_room.clear()
    
    def _link_jupyter_server_extension(self, server_app: ServerApp):
        """Setup custom config needed by this extension."""
        c = Config()
        c.ContentsManager.allow_hidden = True
        c.ContentsManager.hide_globs = [
            "__pycache__",  # Python bytecode cache directories
            "*.pyc",  # Compiled Python files
            "*.pyo",  # Optimized Python files
            ".DS_Store",  # macOS system files
            "*~",  # Editor backup files
            ".ipynb_checkpoints",  # Jupyter notebook checkpoint files
            ".git",  # Git version control directory
            ".venv",  # Python virtual environment directory
            "venv",  # Python virtual environment directory
            "node_modules",  # Node.js dependencies directory
            ".pytest_cache",  # PyTest cache directory
            ".mypy_cache",  # MyPy type checker cache directory
            "*.egg-info",  # Python package metadata directories
        ]
        server_app.update_config(c)
        super()._link_jupyter_server_extension(server_app)