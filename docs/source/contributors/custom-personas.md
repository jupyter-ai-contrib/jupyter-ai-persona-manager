# Developing a custom persona

A persona is a subclass of `BasePersona`. At minimum you implement **two things**
— a `defaults` property and a `process_message` method — and register the class
via an entry point. Everything else has a working default you can override as
needed. The full contract, grouped by what you _must_ vs. _may_ implement, is in
the [Python API](./python-api.md).

## 1. Subclass `BasePersona`

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

- **`defaults`** ({py:attr}`~jupyter_ai_persona_manager.BasePersona.defaults`) —
  Required. Returns a
  {py:class}`~jupyter_ai_persona_manager.PersonaDefaults` with the persona's
  name, description, avatar path, and system prompt.
- **`process_message`**
  ({py:meth}`~jupyter_ai_persona_manager.BasePersona.process_message`) —
  Required. The main entry point: produce a reply, typically via
  {py:meth}`~jupyter_ai_persona_manager.BasePersona.stream_message` (streaming)
  or {py:meth}`~jupyter_ai_persona_manager.BasePersona.send_message` (one shot).

## 2. Register it via an entry point

In your package's `pyproject.toml`:

```toml
[project.entry-points."jupyter_ai.personas"]
my-persona = "my_package.my_persona:MyPersona"
```

The manager discovers personas from this entry point group at startup. (For
quick local iteration you can also drop a `*persona*.py` file into a chat's
`.jupyter/personas/` directory and run `/refresh-personas` — no reinstall.)

## 3. Implement more of the contract as needed

Beyond the two required members, the pieces you're most likely to add:

- **Streaming a model reply.** Pass an async iterator of chunks to
  {py:meth}`~jupyter_ai_persona_manager.BasePersona.stream_message`; it creates
  and progressively updates a single chat message and manages your typing
  indicator automatically.
- **Interrupting** ({py:meth}`~jupyter_ai_persona_manager.BasePersona.cancel_response`,
  _Recommended_ for streaming personas). Override it to stop your backend when
  the user hits the interrupt button.
- **Being configurable.** If users can pick your model or settings, override the
  _Optional_ `update_model` / `update_model_settings` / `update_settings`
  methods to switch your backend, and publish your options with the
  `report_model_configuration` / `report_settings_configuration` methods. The
  base class handles broadcasting the _current_ selection for you — see
  [Architecture](./architecture.md) for the full flow.
- **Cleanup** ({py:meth}`~jupyter_ai_persona_manager.BasePersona.shutdown`,
  _Recommended_). Stop background tasks; call `await super().shutdown()` first.
- **Reading attachments.** Use
  {py:meth}`~jupyter_ai_persona_manager.BasePersona.process_attachments` to pull
  attached file contents into your prompt.

See the [Python API](./python-api.md) for the exhaustive list — each member is
labeled by its contract level.

## Marking your own API for the docs

The contract levels on the [Python API](./python-api.md) page are not hand-
maintained — they come from **documentation markers** on each method. If you
build an API that others subclass or call (a new base class, a mixin), mark its
public members the same way so its generated docs group them and a CI check
keeps every member marked.

The markers live in `jupyter_ai_persona_manager/doc_markers.py`. They are
**documentation-only** — each `mark_*` decorator just stamps a metadata attribute
and returns the function unchanged, so there is no runtime effect. Apply one per
public member:

```python
from jupyter_ai_persona_manager.doc_markers import (
    mark_required, mark_recommended, mark_optional,
    mark_subclass_api, mark_consumer_api,
)

class MyBase(ABC):
    @mark_required
    @abstractmethod
    def must_implement(self): ...

    @mark_recommended
    def should_override(self): ...

    @mark_subclass_api
    def call_me_from_a_subclass(self): ...

    @mark_consumer_api
    def call_me_from_the_framework(self): ...
```

`doc_markers.py` is self-contained and dependency-free. To add this to **another
package**, copy that one file over and mark its members — there is no shared
package for it yet. Then render the grouped reference with the `contract-api`
directive (see the main Jupyter AI contributor guide on subpackage
documentation).
