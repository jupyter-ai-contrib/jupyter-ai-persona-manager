import json
import os
import sys
from pathlib import Path

from jupyterlite_core.addons.base import BaseAddon

# The labextension is installed to share/jupyter/labextensions/ (shared-data),
# not into the Python package directory. Check sys.prefix first (covers venvs
# and nox sessions), then fall back to the source tree for editable installs.
_SHARE_PACKAGE_JSON = (
    Path(sys.prefix)
    / "share"
    / "jupyter"
    / "labextensions"
    / "@jupyter-ai"
    / "persona-manager"
    / "package.json"
)
_LOCAL_PACKAGE_JSON = Path(__file__).parent / "labextension" / "package.json"


def _disabled_extensions_from_package_json() -> list[str]:
    """ Read jupyterlab.disabledExtensions from the installed labextension package.json. """
    for candidate in (_SHARE_PACKAGE_JSON, _LOCAL_PACKAGE_JSON):
        if candidate.exists():
            data = json.loads(candidate.read_text())
            return data.get("jupyterlab", {}).get("disabledExtensions", [])
    return []


class DisableConflictingExtensionAddon(BaseAddon):
    """ Disable extensions in jupyterlite build """
    __all__ = ["post_build"]

    def post_build(self, manager):
        yield dict(
            name="disable-conflicting-extension",
            actions=[lambda: self._patch_config(manager)],
        )

    def _patch_config(self, manager):
        to_disable = _disabled_extensions_from_package_json()

        config_path = manager.output_dir / "jupyter-lite.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        jupyter_config = config.setdefault("jupyter-config-data", {})
        disabled = jupyter_config.setdefault("disabledExtensions", [])

        for extension_id in to_disable:
            if extension_id not in disabled:
                disabled.append(extension_id)

        # Expose window.jupyterapp for Galata/Playwright tests. Opt-in only:
        # enabled when JAI_E2E_SUITE is set (i.e. inside the nox e2e session).
        if os.environ.get("JAI_E2E_SUITE"):
            jupyter_config["exposeAppInBrowser"] = True

        config_path.write_text(json.dumps(config, indent=2) + "\n")
