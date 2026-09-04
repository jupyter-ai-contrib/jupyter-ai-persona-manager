import json
from pathlib import Path

from jupyterlite_core.addons.base import BaseAddon

_PACKAGE_JSON = Path(__file__).parent / "labextension" / "package.json"


def _disabled_extensions_from_package_json() -> list[str]:
    """ Read jupyterlab.disabledExtensions from labextension package.json. """
    if not _PACKAGE_JSON.exists():
        return []
    data = json.loads(_PACKAGE_JSON.read_text())
    return data.get("jupyterlab", {}).get("disabledExtensions", [])


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
        if not to_disable:
            return

        config_path = manager.output_dir / "jupyter-lite.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        jupyter_config = config.setdefault("jupyter-config-data", {})
        disabled = jupyter_config.setdefault("disabledExtensions", [])

        for extension_id in to_disable:
            if extension_id not in disabled:
                disabled.append(extension_id)

        config_path.write_text(json.dumps(config, indent=2) + "\n")
