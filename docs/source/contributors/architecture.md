# Architecture

This page explains how the pieces fit together: how a persona instance is scoped
to a chat, how it is selected and receives a message, how its state reaches the
browser, and where the UI controls come from.

## One `PersonaManager` per chat

The unit of the system is the chat. When a chat opens, jupyter-ai-router notifies
the persona manager, which creates **one `PersonaManager` instance bound to that
chat's room** (`text:chat:<file-id>`). That manager:

- loads the available persona **classes** — from the `jupyter_ai.personas`
  entry-point group of installed packages, and from `*persona*.py` files in the
  chat's local `.jupyter/personas/` directory — and
- **instantiates one persona object per class, scoped to that chat's
  `YChat`**. Each persona instance therefore has its own `self.ychat` and its own
  awareness slot, and lives as long as the chat session does.

`/refresh-personas` tears the instances down and rebuilds them (picking up code
changes and new local personas) without restarting the server.

## Selecting a persona: message metadata, not `@`-mentions

As of Jupyter AI v3.1, personas are **not** selected by `@`-mentioning them in the
message text (see the [Overview](./overview.md)). Instead, the chat input's
persona picker stamps the chosen persona's ID onto the **message metadata** under
`metadata["to_persona"]`. When a message arrives, the manager reads that key and
delivers the message to exactly that persona by calling its
{py:meth}`~jupyter_ai_persona_manager.BasePersona.process_message`. A message that
names no target — or an uninstalled one — is routed nowhere.

The user's **model and settings selections travel the same way**. The picker adds
a `model` spec (a {py:class}`~jupyter_ai_persona_manager.ModelSpec`) and
`settings` to the message metadata. Before your `process_message` runs, the
manager calls
{py:meth}`~jupyter_ai_persona_manager.BasePersona.apply_specs_in_message`, which
applies those selections to your persona (calling your `update_*` overrides only
for values that actually changed) and records the new current values. So each
user's choices are private to their message, and your `process_message` sees an
already-configured persona.

## Broadcasting state: the awareness channel

Session state flows the _other_ way — from persona to browser — over the chat's
**Yjs awareness channel**, not through REST polling. Awareness is a per-client
key/value map on the shared document; this package reserves two kinds of slots:

- **`PersonaManagerAwareness`** (a fixed, well-known client ID) publishes the
  **list of personas** in the chat (each a
  {py:class}`~jupyter_ai_persona_manager.PersonaOption`), so the picker can render
  them and map each to its awareness slot.
- **`PersonaAwareness`** (one per persona, on that persona's own client ID)
  publishes that persona's **model configuration, general settings, usage, slash
  commands, and writing status**.

Inside a persona, `self.awareness` exposes these as typed properties — assigning
one rebroadcasts it. The `get_*` readers and `report_*` setters on `BasePersona`
are thin views over this slot: `report_usage(...)` publishes usage the UI renders
in a meter; the streaming helper toggles the `is_writing` flag that drives the
typing indicator. There's no separate in-memory copy to keep in sync — the
awareness slot _is_ the state.

## Where the UI comes from

This package is not backend-only: its frontend extension ships the chat input
**toolbar** — the persona selector, model picker, settings, usage chip, and stop
button — as well as the slash-command completions. It registers these as
JupyterLab plugins (see `src/index.ts`):

- an **input-toolbar factory** (`persona-controls.tsx`, `stop-button.tsx`) that
  extends Jupyter Chat's default toolbar (Send, Attach, Cancel) with the persona
  controls;
- a **slash-command provider** (`slash-commands.ts`) that offers each selected
  persona's advertised commands; and
- the core plugin.

The chat surface itself — the message list and input box — still comes from the
**Jupyter Chat** extension (`jupyterlab-chat`); this package plugs its toolbar
into Jupyter Chat's input-toolbar registry. (These controls previously lived in
`jupyter-ai-acp-client` and were moved here so persona UI and persona logic ship
together.)

The two halves meet at awareness and message metadata: the controls render from
what personas broadcast over awareness (the persona list, each persona's models,
settings, and usage), and writing to the chat stamps the user's selection onto
the message metadata (`metadata.ts`) that the backend reads. The stop button
POSTs to the [cancel endpoint](./rest-api.md), which asks each processing persona
to {py:meth}`~jupyter_ai_persona_manager.BasePersona.cancel_response`.

## Putting it together: one message, end to end

1. The user picks a persona (and optionally a model/settings) and sends a
   message. The picker stamps `to_persona`, `model`, and `settings` onto the
   message metadata.
2. The router delivers the message to that chat's `PersonaManager`, which reads
   `to_persona` and finds the persona instance.
3. The manager applies the message's model/settings spec to the persona
   (`apply_specs_in_message`), then calls `process_message` inside
   `track_processing` (so `processing` is true for its duration).
4. The persona streams its reply via `stream_message`, which creates/updates one
   chat message and toggles the `is_writing` awareness flag; usage is published
   via `report_usage`.
5. The browser renders the streamed message, the typing indicator, and the usage
   meter — all from awareness. If the user interrupts, the cancel endpoint calls
   the persona's `cancel_response`.

See the [Python API](./python-api.md) for the members named above.
