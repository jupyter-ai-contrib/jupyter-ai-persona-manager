# `BasePersona` API

[`BasePersona`][BasePersona] is the abstract base class for every AI persona. It
is a [`LoggingConfigurable`][LoggingConfigurable] (so it has a `.log` and
participates in traitlets configuration) and an
[`abc.ABC`][abc], combined via a small metaclass so it can be both.

A persona instance is **scoped to a single chat** ([`YChat`][ychat]). The
[`PersonaManager`][PersonaManager] creates one instance of your persona per chat,
sets the instance attributes below, registers the persona as a chat user, then
routes matching messages to it.

## Contract summary

The interface splits into three groups by **who is responsible** for each member:

| Group | Members | You… |
|-------|---------|------|
| **Must implement** | [`defaults`](#defaults), [`process_message`](#process-message) | **must** define these — the class is abstract without them |
| **May override** | [`cancel_response`](#cancel-response), [`shutdown`](#shutdown), [`handle_uncaught_exception`](#handle-uncaught-exception), [`update_model`](#update-model) / [`update_model_settings`](#update-model-settings) / [`update_settings`](#update-settings) | override only if your persona needs the behavior; each has a safe default |
| **Provided — call, don't override** | messaging, session readers/reporters, spec application, attachments, paths, identity | call these from your persona; the base class implements them |

---

## Must implement

These are `@abstractmethod`; a subclass that omits either cannot be instantiated.

(defaults)=
### `defaults`

```python
@property
@abstractmethod
def defaults(self) -> PersonaDefaults: ...
```

Return a [`PersonaDefaults`](#personadefaults) describing the persona's default
identity and behavior — `name`, `description`, `avatar_path`, `system_prompt`,
and optionally `slash_commands` and `model_uid`. Several base-class properties
(`name`, `system_prompt`) read from this, so it must be cheap and side-effect
free. — [source][defaults-src]

(process-message)=
### `process_message`

```python
@abstractmethod
async def process_message(self, message: Message) -> None: ...
```

The **main entry point** of a persona: given an incoming
[`Message`][chat-models], produce a reply. This is where you call your model or
agent and write the response back to the chat — typically via
[`stream_message`](#stream-message) (streaming) or [`send_message`](#send-message)
(one shot). Read chat history through `self.ychat`.

`PersonaManager` calls this for every message routed to the persona, after
applying any per-message model/settings spec (see
[`apply_specs_in_message`](#apply-specs-in-message)) and while holding
[`track_processing`](#track-processing) so [`processing`](#processing) is true for
its duration. — [source][process-message-src]

---

## May override

Each of these has a working default. Override it when your persona needs the
behavior; otherwise inherit it.

(cancel-response)=
### `cancel_response`

```python
async def cancel_response(self) -> None: ...   # default: no-op
```

Stop the persona's in-progress reply when the user interrupts it. The counterpart
to [`process_message`](#process-message): halt whatever the reply set in motion (a
model stream, an agent turn, pending tool calls). The default is a **no-op**, fine
for a persona that replies synchronously or has nothing to cancel. A streaming or
long-running persona (e.g. an ACP agent) overrides this to interrupt its backend.

Only invoked when the persona is [`processing`](#processing) — the cancel handler
gates on that — so an override may assume a reply is in flight. — [source][cancel-src]

(shutdown)=
### `shutdown`

```python
async def shutdown(self) -> None: ...
```

Tear the persona down: stop background tasks and remove it from chat awareness.
Called on `/refresh-personas`, and may be called on server shutdown or when a
chat closes. The default removes the persona from awareness; an override that
adds custom cleanup should call `await super().shutdown()` **first**. — [source][shutdown-src]

(handle-uncaught-exception)=
### `handle_uncaught_exception`

```python
async def handle_uncaught_exception(self, exc: Exception) -> None: ...
```

Called by `PersonaManager` when [`process_message`](#process-message) raises. The
default posts a chat message showing the error type/message with the full
traceback tucked under a collapsible `<details>`. Override to customize error
reporting. — [source][handle-exc-src]

(update-model)=
### `update_model`

```python
async def update_model(self, model_id: str) -> None: ...   # default: no-op
```

(update-model-settings)=
### `update_model_settings`

```python
async def update_model_settings(self, settings: dict[str, str | None]) -> None: ...   # default: no-op
```

(update-settings)=
### `update_settings`

```python
async def update_settings(self, settings: dict[str, str | None]) -> None: ...   # default: no-op
```

These apply a user's selection to your **backend** — switch the model, apply model
settings (e.g. context size), or apply general settings (e.g. mode/effort). They
are no-ops by default; override them **only if your persona is configurable**.

Important division of labor: an `update_*` method should *only* tell the backend
to switch — it must **not** touch awareness. The base class records the new
current value and rebroadcasts it for you (see
[`apply_model_spec`](#apply-model-spec) / [`apply_settings_spec`](#apply-settings-spec)).
A configurable persona overrides the relevant `update_*` **and** publishes its
options via [`report_model_configuration`](#report-model-configuration) /
[`report_settings_configuration`](#report-settings-configuration) so the UI has
something to select. There is deliberately no `update_*` for usage or slash
commands — a user can't set those. — [source][update-src]

---

## Provided by the base class

Call these; don't override them. Grouped by purpose.

### Messaging

(stream-message)=
#### `stream_message`

```python
async def stream_message(self, reply_stream: "AsyncIterator") -> None
```

Stream an async iterator of chunks into a single, progressively-updated chat
message. Chunks may be `str` or [`litellm.ModelResponseStream`][litellm-stream]
objects (the `.choices[0].delta.content` is extracted). It creates the message on
the first non-empty chunk, manages the `is_writing` awareness flag automatically
(clearing it even on error), and runs mention detection once the stream completes
so personas can @-mention each other. This is the usual way to emit a model
reply. — [source][stream-src]

(send-message)=
#### `send_message`

```python
def send_message(self, body: str) -> None
```

Post a complete message to the chat from this persona in one shot. Use it for
non-streaming replies or status notes. — [source][send-src]

### Identity & properties

| Member | Kind | Returns | Notes |
|--------|------|---------|-------|
| [`id`](#id) | property | `str` | Stable unique ID `jupyter-ai-personas::<package>::<class>`; sets `username` |
| `name` | property | `str` | Display name; reads `defaults.name` |
| `avatar_path` | property | `str` | Avatar **URL** route `…/api/ai/avatars/<id>` (not the filesystem path) |
| `system_prompt` | property | `str` | Reads `defaults.system_prompt` |
| `event_loop` | property | `AbstractEventLoop` | The process event loop |
| [`processing`](#processing) | property | `bool` | Whether a `process_message` call is in flight |
| `as_user()` | method | [`User`][chat-models] | The chat-user model for this persona |
| `as_user_dict()` | method | `dict` | `as_user()` as a plain dict |

(id)=
`id` is guaranteed to be `jupyter-ai-personas::<package-name>::<persona-class-name>`.
The prefix lets consumers tell AI personas apart from human users; if one package
ships several personas, their class names must differ. — [source][id-src]

(processing)=
`processing` is backed by an internal counter (a persona may process several
messages at once) maintained by [`track_processing`](#track-processing). Use it —
as the cancel endpoint does — to avoid interrupting a persona with nothing to
cancel. — [source][processing-src]

(track-processing)=
#### `track_processing`

```python
@contextlib.contextmanager
def track_processing(self)
```

Context manager that marks the persona as [`processing`](#processing) for its
duration (restoring the count even on error). `PersonaManager` already wraps each
`process_message` call in it, so you rarely call this yourself. — [source][track-src]

### Reading session information

These `get_*` readers are thin, always-current views over the persona's
[awareness](#instance-attributes) slot (no separate in-memory copy).

| Method | Returns |
|--------|---------|
| `get_model_configuration()` | [`ModelConfiguration`](#modelconfiguration) — current model, options, and model settings |
| `get_setting_configurations()` | `list[`[`SettingConfiguration`](#settingconfiguration)`]` — general settings |
| `get_model()` | `str \| None` — current model ID (None = persona default) |
| `get_model_settings()` | `dict[str, str \| None]` — model settings by ID |
| `get_settings()` | `dict[str, str \| None]` — general settings by ID |
| `get_usage()` | [`Usage`](#usage) — token/cost usage for the session |
| `get_slash_commands()` | `list[`[`CommandOption`](#commandoption)`]` — advertised slash commands |

— [source][get-src]

### Reporting session information

A persona calls these `report_*` setters to publish its own state over awareness
(so the UI can render the model picker, settings, usage meter, and slash-command
menu). Assigning an awareness property rebroadcasts it, so most just forward.

| Method | Publishes |
|--------|-----------|
| [`report_model_configuration(model)`](#report-model-configuration) | model, options, and model settings |
| [`report_settings_configuration(settings)`](#report-settings-configuration) | general (non-model) settings |
| `report_usage(usage, *, append=False)` | merges [`Usage`](#usage) fields and rebroadcasts |
| `report_slash_commands(commands)` | advertised slash commands |

`report_usage` is the one with real logic: only the fields set on `usage` are
merged. `append=False` (default) replaces each provided field (for sources that
report cumulative totals, e.g. ACP); `append=True` adds to the stored value (for
per-turn deltas). Snapshot fields (`context_*`) should never be sent with
`append=True`. — [source][report-src]

(report-model-configuration)=
(report-settings-configuration)=

### Applying a message's model & settings spec

You normally don't call these — `PersonaManager` does — but they explain how
per-message model/settings selections reach your `update_*` overrides.

| Method | Role |
|--------|------|
| [`apply_specs_in_message(message)`](#apply-specs-in-message) | Reads `message.metadata` and dispatches to the two below; called before `process_message` for every routed message |
| [`apply_model_spec(spec)`](#apply-model-spec) | Applies a [`ModelSpec`](#modelspec): calls `update_model` / `update_model_settings` only for values that actually change, then records the new current values |
| [`apply_settings_spec(spec)`](#apply-settings-spec) | Same, for general settings |

(apply-specs-in-message)=
(apply-model-spec)=
(apply-settings-spec)=
A `None` model ID or setting value means "keep the persona's current value" and
is skipped; the backend is asked to switch (and the change broadcast) only when a
specified value differs from the current one. — [source][apply-src]

### File attachments

| Method | Returns | Purpose |
|--------|---------|---------|
| `process_attachments(message)` | `str \| None` | Read every attachment on the message and concatenate their contents into a single string (each wrapped with a `File: <path>` header), for feeding into a prompt |
| `resolve_attachment_to_path(attachment_id)` | `str \| None` | Resolve one attachment ID to a filesystem path (tries the workspace dir, then an absolute path) |

— [source][attach-src]

### Chat file & workspace paths

Thin delegates to the [`PersonaManager`][PersonaManager] for filesystem context
of the current chat:

| Method | Returns |
|--------|---------|
| `get_chat_path(relative=False)` | Absolute path of the chat file (or relative to the Contents root) |
| `get_chat_dir()` | Directory containing the chat file |
| `get_dotjupyter_dir()` | The chat's `.jupyter` directory, or `None` |
| `get_workspace_dir()` | The chat's workspace directory |
| `get_mcp_settings()` | The chat's [`McpSettings`](#mcp-models), or `None` |

— [source][paths-src]

---

(instance-attributes)=
## Instance attributes

Set for you by `BasePersona` / its parent before `process_message` is called:

| Attribute | Type | Set by | Meaning |
|-----------|------|--------|---------|
| `ychat` | [`YChat`][ychat] | `BasePersona` | The collaborative chat this persona is scoped to; read history and write messages through it |
| `parent` | [`PersonaManager`][PersonaManager] | `LoggingConfigurable` | The manager for this chat |
| `log` | [`logging.Logger`][logging] | `LoggingConfigurable` | Logger for this persona |
| `awareness` | [`PersonaAwareness`][persona-awareness] | `BasePersona` | This persona's awareness slot; typed properties (`model`, `settings`, `usage`, `slash_commands`, `is_writing`) publish over the Yjs awareness channel when assigned. The `get_*`/`report_*` methods are views over this. |

---

## Types & models

Everything below is a [Pydantic][pydantic] `BaseModel` unless noted. All are
importable from the package root, e.g.
`from jupyter_ai_persona_manager import PersonaDefaults`.

(personadefaults)=
### `PersonaDefaults`

A persona's default identity/behavior, returned by [`defaults`](#defaults). Fields
may be overridden through the settings UI. — [source][personadefaults-src]

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | ✓ | Display name, e.g. `"Jupyternaut"` |
| `description` | `str` | ✓ | Short description |
| `avatar_path` | `str` | ✓ | **Absolute filesystem path** to an avatar image (SVG/PNG/JPG) |
| `system_prompt` | `str` | ✓ | System prompt |
| `slash_commands` | `set[str]` | | Enabled slash commands; defaults to `{"*"}` (all) |
| `model_uid` | `str \| None` | | Default model to use on a fresh start, e.g. `"ollama:deepseek-coder-v2"` |

(modelconfiguration)=
### `ModelConfiguration`

The persona's current model plus its selectable options and model settings.
Returned by `get_model_configuration()`; published via
`report_model_configuration()`. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `current` | `str \| None` | Current model ID (None = persona default) |
| `options` | `list[`[`ModelOption`](#modeloption)`]` | Selectable models |
| `settings` | `list[`[`SettingConfiguration`](#settingconfiguration)`]` | Settings rendered near the model picker |

(modeloption)=
### `ModelOption`

One selectable model. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Model ID |
| `name` | `str \| None` | Display name |
| `description` | `str \| None` | Description |

(settingoption)=
### `SettingOption`

One selectable value for a setting. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Option ID |
| `name` | `str \| None` | Display name |
| `description` | `str \| None` | Description |

(settingconfiguration)=
### `SettingConfiguration`

A single setting: its current value plus all options. Used for both model
settings and general settings; list order controls UI order. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Setting ID, e.g. `"agent_mode"` |
| `current` | `str \| None` | Current value (None = persona default) |
| `name` | `str \| None` | Display name |
| `description` | `str \| None` | Description |
| `options` | `list[`[`SettingOption`](#settingoption)`]` | Available values |

(usage)=
### `Usage`

Token and cost usage for the current session, reported via `report_usage()`.
Every field is optional. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `context_tokens` | `int \| None` | Tokens currently in the context window (snapshot — can decrease) |
| `context_size` | `int \| None` | Total context-window size (snapshot) |
| `input_tokens` | `int \| None` | Cumulative input tokens |
| `output_tokens` | `int \| None` | Cumulative output tokens |
| `cached_read_tokens` | `int \| None` | Cumulative cached-read tokens |
| `cached_write_tokens` | `int \| None` | Cumulative cached-write tokens |
| `thought_tokens` | `int \| None` | Cumulative reasoning/thought tokens |
| `total_tokens` | `int \| None` | Cumulative total tokens |
| `cost_amount` | `float \| None` | Cumulative session cost |
| `cost_currency` | `str \| None` | ISO 4217 currency, e.g. `"USD"` |

(commandoption)=
### `CommandOption`

One slash command advertised by a persona. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Command including leading `/`, e.g. `"/compact"` |
| `description` | `str \| None` | Description |

(modelspec)=
### `ModelSpec`

A user's model selection, carried on **outgoing message metadata** (not stored in
awareness — each user's choices are private and applied per message). Consumed by
[`apply_model_spec`](#apply-model-spec). — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Selected model ID (None = keep current) |
| `settings` | `dict[str, str \| None]` | Model-setting ID → selected option ID (None = keep current) |

(persona-option)=
### `PersonaOption`

One persona as advertised in the chat's persona selector; published by the
manager's awareness slot, not by an individual persona. — [source][awareness-src]

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Persona ID |
| `name` | `str` | Display name |
| `avatar_url` | `str \| None` | Avatar URL |
| `yjs_client_id` | `int` | Yjs client ID of this persona's awareness slot (for O(1) lookup) |

(mcp-models)=
### MCP settings models

`McpSettings`, `McpServerHttp`, and `McpServerStdio` describe the MCP server
configuration returned by `get_mcp_settings()`. — [source][mcp-src]

[BasePersona]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L68
[defaults-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L142
[process-message-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L152
[cancel-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L164
[shutdown-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L716
[handle-exc-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L385
[update-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L546
[stream-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L314
[send-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L379
[id-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L214
[processing-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L187
[track-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L196
[get-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L450
[report-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L489
[apply-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L561
[attach-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L647
[paths-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L409
[personadefaults-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/base_persona.py#L36
[awareness-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/awareness_models.py
[mcp-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/mcp_server_models.py
[PersonaManager]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/persona_manager.py
[persona-awareness]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/persona_awareness.py
[ychat]: https://github.com/jupyterlab/jupyter-chat/blob/main/python/jupyterlab-chat/jupyterlab_chat/ychat.py
[chat-models]: https://github.com/jupyterlab/jupyter-chat/blob/main/python/jupyterlab-chat/jupyterlab_chat/models.py
[LoggingConfigurable]: https://traitlets.readthedocs.io/en/stable/config-api.html#traitlets.config.LoggingConfigurable
[abc]: https://docs.python.org/3/library/abc.html
[logging]: https://docs.python.org/3/library/logging.html#logging.Logger
[pydantic]: https://docs.pydantic.dev/latest/
[litellm-stream]: https://docs.litellm.ai/docs/completion/stream
