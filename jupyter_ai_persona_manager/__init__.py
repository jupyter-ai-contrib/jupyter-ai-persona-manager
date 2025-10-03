try:
    from ._version import __version__
except ImportError:
    # Fallback when using the package in dev mode without installing
    # in editable mode with pip. It is highly recommended to install
    # the package from a stable release or in editable mode: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs
    import warnings
    warnings.warn("Importing 'jupyter_ai_persona_manager' outside a proper installation.")
    __version__ = "dev"

from .base_persona import BasePersona, PersonaDefaults
from .persona_manager import PersonaManager
from .persona_awareness import PersonaAwareness
from .extension import PersonaManagerExtension


def _jupyter_labextension_paths():
    return [{
        "src": "labextension",
        "dest": "@jupyter-ai/persona-manager"
    }]


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_persona_manager", "app": PersonaManagerExtension}]
