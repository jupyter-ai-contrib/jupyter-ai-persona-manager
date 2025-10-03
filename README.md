# jupyter_ai_persona_manager

[![Github Actions Status](https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/workflows/Build/badge.svg)](https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/actions/workflows/build.yml)

The core manager & registry for AI personas in Jupyter AI.

This package provides the foundational infrastructure for managing AI personas in Jupyter AI chat environments. It includes:

- **BasePersona**: Abstract base class for creating custom AI personas
- **PersonaManager**: Registry and lifecycle management for personas
- **PersonaAwareness**: Awareness integration for multi-user chat environments
- **Entry Point Support**: Automatic discovery of personas via Python entry points

AI personas are analogous to "bots" in other chat applications, allowing different AI assistants to coexist in the same chat environment. Each persona can have unique behavior, models, and capabilities.

## Adding a New Persona via Entry Points

To create and register a custom AI persona:

### 1. Create Your Persona Class

```python
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

class MyCustomPersona(BasePersona):
    @property
    def defaults(self):
        return PersonaDefaults(
            name="MyPersona",
            description="A helpful custom assistant",
            avatar_path="/api/ai/static/custom-avatar.svg",
            system_prompt="You are a helpful assistant specialized in...",
        )

    async def process_message(self, message: Message):
        # Your custom logic here
        response = f"Hello! You said: {message.body}"
        self.send_message(response)
```

### 2. Register via Entry Points

Add to your package's `pyproject.toml`:

```toml
[project.entry-points."jupyter_ai.personas"]
my-custom-persona = "my_package.personas:MyCustomPersona"
```

### 3. Install and Restart

```bash
pip install your-package
# Restart JupyterLab to load the new persona
```

Your persona will automatically appear in Jupyter AI chats and can be @-mentioned by name.

## Loading Personas from .jupyter Directory

For development and local customization, personas can be loaded from the `.jupyter/personas/` directory:

### Directory Structure

```
.jupyter/
└── personas/
    ├── my_custom_persona.py
    ├── research_assistant.py
    └── debug_helper.py
```

### File Requirements

- Place Python files in `.jupyter/personas/` (not directly in `.jupyter/`)
- Filename must contain "persona" (case-insensitive)
- Cannot start with `_` or `.` (treated as private/hidden)
- Must contain a class inheriting from `BasePersona`

### Example Local Persona

**File: `.jupyter/personas/my_persona.py`**

```python
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

class MyLocalPersona(BasePersona):
    @property
    def defaults(self):
        return PersonaDefaults(
            name="Local Dev Assistant",
            description="A persona for local development",
            avatar_path="/api/ai/static/jupyternaut.svg",
            system_prompt="You help with local development tasks.",
        )

    async def process_message(self, message: Message):
        self.send_message(f"Local persona received: {message.body}")
```

### Refreshing Personas

Use the `/refresh-personas` slash command in any chat to reload personas without restarting JupyterLab:

```
/refresh-personas
```

This allows for iterative development - modify your local persona files and refresh to see changes immediately.

Development install:

```
micromamba install uv jupyterlab nodejs=22
jlpm
jlpm dev:install
```

## Requirements

- JupyterLab >= 4.0.0

## Install

To install the extension, execute:

```bash
pip install jupyter_ai_persona_manager
```

## Uninstall

To remove the extension, execute:

```bash
pip uninstall jupyter_ai_persona_manager
```

## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```

## Contributing

### Development install

Note: You will need NodeJS to build the extension package.

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Change directory to the jupyter_ai_persona_manager directory
# Install package in development mode
pip install -e ".[test]"
# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Server extension must be manually installed in develop mode
jupyter server extension enable jupyter_ai_persona_manager
# Rebuild extension Typescript source after making changes
jlpm build
```

You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
jlpm watch
# Run JupyterLab in another terminal
jupyter lab
```

With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

By default, the `jlpm build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:

```bash
jupyter lab build --minimize=False
```

### Development uninstall

```bash
# Server extension must be manually disabled in develop mode
jupyter server extension disable jupyter_ai_persona_manager
pip uninstall jupyter_ai_persona_manager
```

In development mode, you will also need to remove the symlink created by `jupyter labextension develop`
command. To find its location, you can run `jupyter labextension list` to figure out where the `labextensions`
folder is located. Then you can remove the symlink named `@jupyter-ai/persona-manager` within that folder.

### Testing the extension

#### Server tests

This extension is using [Pytest](https://docs.pytest.org/) for Python code testing.

Install test dependencies (needed only once):

```sh
pip install -e ".[test]"
# Each time you install the Python package, you need to restore the front-end extension link
jupyter labextension develop . --overwrite
```

To execute them, run:

```sh
pytest -vv -r ap --cov jupyter_ai_persona_manager
```

#### Frontend tests

This extension is using [Jest](https://jestjs.io/) for JavaScript code testing.

To execute them, execute:

```sh
jlpm
jlpm test
```

#### Integration tests

This extension uses [Playwright](https://playwright.dev/docs/intro) for the integration tests (aka user level tests).
More precisely, the JupyterLab helper [Galata](https://github.com/jupyterlab/jupyterlab/tree/master/galata) is used to handle testing the extension in JupyterLab.

More information are provided within the [ui-tests](./ui-tests/README.md) README.

### Packaging the extension

See [RELEASE](RELEASE.md)
