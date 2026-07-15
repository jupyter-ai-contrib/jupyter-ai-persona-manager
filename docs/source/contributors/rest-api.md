# REST API

The persona manager registers three HTTP endpoints on Jupyter Server (see
[`extension.py`][ext-src]). All are Jupyter Server handlers, so they require the
usual authentication — include the XSRF token / auth cookie that JupyterLab uses
for its own API calls. Paths below are relative to the server's `base_url`.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/ai/message/<persona_name>` | Send a one-shot message to a persona and get its reply |
| `POST` | `/api/ai/personas/cancel?chat_path=<path>` | Interrupt in-progress replies in a chat |
| `GET`  | `/api/ai/avatars/<persona_id>` | Fetch a persona's avatar image |

```{note}
These endpoints are backend infrastructure. In normal use, the chat UI talks to
personas over the collaborative `YChat` document and the Yjs **awareness**
channel, not over REST — awareness is how a persona's model, settings, usage, and
typing status reach the browser. The message and cancel endpoints exist for
one-shot / programmatic messaging and for the interrupt button.
```

## `POST /api/ai/message/<persona_name>`

Route a single message to the persona whose [`name`](#defaults)
matches `<persona_name>` (URL-encoded), in a fresh temporary chat room, and return
the concatenated reply. Implemented by [`MessageHandler`][msg-src].

**Path parameter**

- `persona_name` — the persona's display name (its `defaults.name`), URL-encoded.
  Note this matches on **name**, not on the persona [`id`](#id).

**Request body** (JSON)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | `str` | ✓ | The message text to send |
| `metadata` | `object` | | Optional message metadata, e.g. a [`ModelSpec`](#modelspec) under a `model` key and general settings under `settings` (see [`apply_specs_in_message`](#apply-specs-in-message)) |

```bash
curl -X POST "$BASE_URL/api/ai/message/MyPersona" \
  -H "Content-Type: application/json" \
  -H "Authorization: token $JUPYTER_TOKEN" \
  -d '{"message": "Hello!"}'
```

**Response** `200` (JSON)

```json
{ "response": "…the persona's reply text…" }
```

The handler creates a temporary room, sends the message mentioning the target
persona, waits for the persona to finish writing (its `isWriting` awareness flag
clears), then returns all message bodies in the room joined together. If the
persona is still writing after **120 seconds**, the wait times out and whatever
has been written so far is returned.

**Errors**

| Status | When |
|--------|------|
| `400` | Body is not valid JSON, or `message` is missing/empty |
| `404` | No persona with that name is found |

## `POST /api/ai/personas/cancel`

Interrupt in-progress replies in a specific chat. Implemented by
[`CancelHandler`][cancel-handler-src]. The chat UI's interrupt button POSTs here.

**Query parameter**

- `chat_path` (required) — the path of the chat file, resolved to a room via the
  file-ID manager.

**Request:** no body needed.

```bash
curl -X POST "$BASE_URL/api/ai/personas/cancel?chat_path=my-chat.chat" \
  -H "Authorization: token $JUPYTER_TOKEN"
```

For each persona in that chat that is currently
[`processing`](#processing), the handler calls
[`cancel_response()`](#cancel-response). Idle personas are
skipped (a cancel with no active reply is out of spec for some backends, e.g.
ACP's `session/cancel`). A persona whose `cancel_response` raises is logged and
skipped, not surfaced as an error.

**Response** `200` (JSON) — the IDs of the personas that were asked to cancel:

```json
{ "status": "cancelled", "cancelled": ["jupyter-ai-personas::my_package::MyPersona"] }
```

**Errors**

| Status | When |
|--------|------|
| `400` | `chat_path` query parameter is missing |
| `404` | No chat/room found for `chat_path`, or the chat has no initialized persona manager |
| `500` | The server's `file_id_manager` is unavailable |

## `GET /api/ai/avatars/<persona_id>`

Serve a persona's avatar image. Implemented by [`AvatarHandler`][avatar-src]. This
is the URL that [`BasePersona.avatar_path`](#instance-attributes)
returns and that appears as `avatar_url` on the persona's chat user — the browser
requests it directly; you rarely call it by hand.

**Path parameter**

- `persona_id` — the persona [`id`](#id) (URL-encoded).

**Response** `200` — the raw image bytes, with `Content-Type` guessed from the
file extension (SVG, PNG, or JPG). The file is resolved from a module-level cache
built when personas are initialized/refreshed, keyed by persona ID (an O(1)
lookup), from each persona's `defaults.avatar_path`.

**Errors**

| Status | When |
|--------|------|
| `404` | No avatar is cached for that persona ID |
| `413` | The avatar file exceeds the 5 MB limit |
| `500` | The file can't be read |

[ext-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/extension.py#L38
[msg-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/handlers.py#L54
[cancel-handler-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/handlers.py#L146
[avatar-src]: https://github.com/jupyter-ai-contrib/jupyter-ai-persona-manager/blob/a38d5a280714e7b3c1cc7640e3e3c0975c245e23/jupyter_ai_persona_manager/handlers.py#L203
