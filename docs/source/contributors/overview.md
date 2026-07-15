# Overview

The persona manager provides the foundational infrastructure for **AI personas**
in Jupyter AI. A persona is an AI participant in a chat — analogous to a "bot" in
other chat applications — with its own identity (name, avatar), model, and
behavior. Several personas can share a chat, respond to users, and even mention
one another.

This package supplies three things:

- **`BasePersona`** — the abstract base class you subclass to build a persona. It
  handles chat integration, awareness (typing indicators, model/usage broadcast),
  message streaming, and file-attachment access, so your subclass only decides
  how to turn a message into a reply.
- **`PersonaManager`** — discovers personas (from Python entry points and from a
  chat's local `.jupyter/personas/` directory), instantiates them per chat, and
  routes messages to the right persona(s).
- **A small REST API** — server endpoints for one-shot messaging, interrupting a
  reply, and serving avatars.

## What changed in Jupyter AI v3.1

Everything after `jupyter_ai_persona_manager` 0.0.x (Jupyter AI **v3.1**) uses a
new interaction model:

- **`@`-mentioning was dropped.** In earlier versions you selected which persona
  responded by `@`-mentioning it in the message text. That mechanism is gone.
- **A new UI selects the persona and its options instead.** Jupyter AI now
  provides dedicated chat UI controls — a persona selector, model picker, and
  settings — contributed by the Jupyter Chat plugin. The user's selection rides
  on **message metadata** rather than being parsed out of the message body, and a
  persona picks it up automatically before it processes the message.

The practical upshot for persona authors: you no longer parse mentions or
commands out of the text to decide whether to respond. The manager routes a
message to your persona, applies the user's model/settings selection for you, and
calls your [`process_message`](./python-api.md). How the selection travels from
the UI to your persona is covered in [Architecture](./architecture.md).

## Where to go next

- [Developing a custom persona](./custom-personas.md) — the practical guide.
- [Architecture](./architecture.md) — how routing, awareness, and the UI fit
  together.
- [Python API](./python-api.md) — the full `BasePersona` reference, grouped by
  what you must vs. may implement.
- [REST API](./rest-api.md) — the HTTP endpoints.
