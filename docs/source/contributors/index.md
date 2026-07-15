# jupyter-ai-persona-manager

The **persona manager** is the core registry and lifecycle manager for AI
personas in Jupyter AI. A _persona_ is an AI participant in a chat — analogous to
a "bot" in other chat apps — with its own name, avatar, model, and behavior.
Multiple personas can coexist in the same chat and even mention one another.

This package provides:

- **`BasePersona`** — the abstract base class you subclass to build a persona. It
  handles chat integration, awareness (typing indicators, model/usage broadcast),
  message streaming, and file-attachment access, so a subclass only implements
  how to turn a message into a reply.
- **`PersonaManager`** — discovers personas (from Python entry points and from a
  chat's local `.jupyter/personas/` directory), instantiates one per chat, and
  routes each incoming message to the mentioned persona(s).
- **A small REST API** — server endpoints for one-shot messaging, interrupting an
  in-progress reply, and serving persona avatars.

Most contributors interact with this package by **writing a persona**: subclass
`BasePersona`, implement two things, and register it via an entry point. The
pages below document that contract and the HTTP surface.

```{toctree}
:maxdepth: 2

base-persona-api
rest-api
```

## Building a persona at a glance

```python
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
import os

AVATAR_PATH = os.path.join(os.path.dirname(__file__), "assets", "avatar.svg")


class MyPersona(BasePersona):
    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="MyPersona",
            description="A helpful custom assistant",
            avatar_path=AVATAR_PATH,
            system_prompt="You are a helpful assistant specialized in…",
        )

    async def process_message(self, message: Message) -> None:
        # Turn the message into a reply. Stream a model response, or just:
        self.send_message(f"You said: {message.body}")
```

Register it via an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."jupyter_ai.personas"]
my-persona = "my_package.my_persona:MyPersona"
```

See the [`BasePersona` API](./base-persona-api.md) for the full method contract —
what you must implement, what you may override, and what the base class provides
for you — and the [REST API](./rest-api.md) for the HTTP endpoints this package
adds to Jupyter Server.
