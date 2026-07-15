# Python API

The complete `BasePersona` API, generated from source and **separated by
contract level**:

- **Required** — you MUST implement these (the class is abstract without them).
- **Recommended** — you SHOULD implement these; a default exists, but most
  personas override them.
- **Optional** — you MAY implement these; a safe default is used otherwise.
- **Available to subclasses** — provided by `BasePersona` for your persona to
  _call_; you may override them.
- **Available to consumers** — provided by `BasePersona` for _consumers_ (the
  `PersonaManager` and other extensions) to call on a persona; you should
  generally not override them.

Each member's level is declared in the code (via the `@mark_required` /
`@mark_recommended` / `@mark_optional` / `@mark_subclass_api` /
`@mark_consumer_api` documentation markers in
`jupyter_ai_persona_manager.doc_markers`) and a CI check keeps every member
marked, so this classification can't drift from the code. Every entry shows its
signature, type hints, a contract badge, and a `[source]` link to the exact
lines on GitHub.

```{eval-rst}
.. contract-api:: jupyter_ai_persona_manager.BasePersona
```

## Data models

The Pydantic models a persona works with, rendered from source with their
fields, types, defaults, and constraints:

```{eval-rst}
.. autopydantic_model:: jupyter_ai_persona_manager.PersonaDefaults
.. autopydantic_model:: jupyter_ai_persona_manager.ModelConfiguration
.. autopydantic_model:: jupyter_ai_persona_manager.ModelOption
.. autopydantic_model:: jupyter_ai_persona_manager.SettingConfiguration
.. autopydantic_model:: jupyter_ai_persona_manager.SettingOption
.. autopydantic_model:: jupyter_ai_persona_manager.Usage
.. autopydantic_model:: jupyter_ai_persona_manager.CommandOption
.. autopydantic_model:: jupyter_ai_persona_manager.ModelSpec
```
