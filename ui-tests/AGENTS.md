# E2E tests for jupyter-ai-persona-manager

Notes for an agent writing or fixing E2E (Galata) tests here. For generic
Playwright/Galata setup and run commands, see [README.md](./README.md); this file
covers what's specific to persona-manager.

## The core idea: a fixture persona is just a `BasePersona` subclass

persona-manager owns the generic persona control surface — the persona picker,
model selector, model settings, general settings, usage chip, slash-command
completions, and stop button (see `src/persona-controls.tsx`, `src/awareness.ts`,
`src/slash-commands.ts`, `src/stop-button.tsx`). All of it is driven by the
awareness channel: a persona publishes its state with the `report_*` methods on
`BasePersona`, and the browser renders it. That is the entire contract these
tests exercise.

So to test the UI against a deterministic persona we don't need a model, a
network, or a subprocess — we write a small `BasePersona` subclass that, in
`__init__`, statically advertises its configuration by calling the API:

- `report_model_configuration(ModelConfiguration(...))` — the model picker's
  options + current, and any model settings rendered next to it.
- `report_settings_configuration([SettingConfiguration(...)])` — general
  settings, rendered as separate controls in list order.
- `report_usage(Usage(...))` — the usage chip.
- `report_slash_commands([CommandOption(...)])` — slash-command completions.

It implements the abstract `update_model` / `update_model_settings` /
`update_settings` to _record_ what a selection applied (so a test can prove the
apply path ran), and `process_message` echoes the current/applied config as YAML
so a control change shows up in the next reply. `BasePersona.apply_specs_in_message`
runs _before_ `process_message` (the PersonaManager applies the message's model &
settings metadata first), so by the time `process_message` runs, `get_model()` /
`get_model_settings()` / `get_settings()` already reflect the user's selection.

> **This is NOT ACP.** jupyter-ai-acp-client's fixtures are fake ACP agent
> _subprocesses_ wrapped by `BaseAcpPersona`; those tests exercise the ACP
> adapter. Here the fixtures subclass `BasePersona` directly and drive the
> generic API in-process. ACP-specific behavior (session modes, config-option
> dedup, subprocess/`session/*` RPCs) is deliberately out of scope — it belongs
> in acp-client.

Fixtures live in [`fixtures/personas/`](./fixtures/personas/), named
`<name>_persona.py`. The shared avatar is `fixtures/assets/persona.svg`, located
via the `JAI_TEST_ASSETS_DIR` env var the server config exports, so a fixture
needs no image asset of its own.

## Persona loading, isolation, and how a suite picks its personas

The PersonaManager loads persona classes from the nearest `.jupyter/personas/`
found by walking **up** from a chat's own directory. We use that for isolation:
one shared test server, but each suite works in its own directory with its own
personas.

A suite **declares its personas in the spec itself** and installs them in
`beforeAll`; a `TestHelpers` instance (from `tests/test-helpers.ts`) drives the
chat under the suite's directory:

```ts
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

const TEST_DIR = 'replies'; // this suite's working directory
const PERSONAS = [FixturePersona.Hello]; // its personas

test.describe('…', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('…', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat(); // chat lives under TEST_DIR
    await helpers.selectPersona(FixturePersona.Hello);
    const reply = await helpers.sendMessage('hi');
    // → routed to only this suite's persona
  });
});
```

The available fixtures are the `FixturePersona` enum in `test-helpers.ts`, whose
`FIXTURE_PERSONAS` table is the single source of truth for each persona's display
name — so specs never hardcode persona names. `installPersonas` reads each
fixture's source from `fixtures/personas/<value>_persona.py` and uploads it to
`<TEST_DIR>/.jupyter/personas/` via Galata's contents API.

Notes:

- **No entry points.** These fixtures are local `.jupyter/personas/` files, not
  `jupyter_ai.personas` entry points, so they load only in a suite's own
  directory. `jupyter_server_test_config.py` also sets
  `PersonaManager.default_persona_id = ""` (so the picker starts at "No one"
  rather than seeding a default that isn't installed) and
  `JUPYTER_AI_ACP_CLIENT_E2E_TESTING_ONLY=1` (so, if acp-client is installed in
  the same env, its vendored ACP entry-point personas are skipped) — keeping the
  persona list exactly the fixtures a suite installs.
- **A fixture persona's class must be defined in its file** (the loader keeps
  only classes whose `__module__` matches the file stem); it may import
  `BasePersona` etc. The filename must contain `persona`.
- **`beforeAll`, not per-test setup**, and it uses the worker-scoped `request`
  fixture (not `page`) — installing files needs no browser, and this keeps UI
  mode working.

## One shared server (ports)

`playwright.config.js` runs a single `webServer` on a random-but-reload-stable
port (pinned into `JAI_TEST_PORT` so every worker's config reload agrees). Its
MCP port is offset from the HTTP port so it doesn't collide with the default
(3001) or a dev server. `reuseExistingServer` is `false` so a run never silently
reuses a dev server that lacks the E2E config — free the port before running
locally.

Every `*.spec.ts` under `tests/` runs against that one server; the config defines
no `projects`, so adding a spec file needs no config change.

## Load-bearing selectors

- **Persona picker:** `.jp-jai-personaControls-persona-btn`. It appears once the
  PersonaManager publishes its persona list, so wait for it with a generous
  timeout. Click it and pick by name via `getByRole('menuitem', { name })`.
- **Session controls** (model / model setting / general setting): the controls
  row renders each control **twice** — a real visible copy and an `aria-hidden`,
  `inert` measurement copy used to size the row. Target the visible ones with the
  direct-child combinator `.…-controls > .…-control-btn` (a control's `title`
  attribute is its label). `TestHelpers#setControl(title, value)` handles this.
- **Usage chip:** `.jp-jai-usage-chip`, with a `.jp-jai-usage-ring` for context
  fill and `.jp-jai-usage-pct` for the percent/token label; the click-through
  popover card is `.jp-jai-usage-card` (a page-scoped MUI portal).
- **Slash completions:** typing `/` opens a page-scoped popup; each option's name
  is a `.jp-chat-command-name` span. `TestHelpers#slashCompletions` polls it.
- **Stop button:** `.jp-jai-stopButton`, enabled only while an AI persona writes.
- **Send:** type into `.jp-chat-input-container` `getByRole('combobox')`, click
  `.jp-chat-send-button`. Rendered messages are `.jp-chat-rendered-message`.

## Gotchas

- **Backend changes need a server restart, frontend changes a rebuild.** A
  `.py`-only change to a fixture persona is picked up when a chat next
  initializes; changes to the extension's `src/` need `jlpm build` first (see the
  workbench `rebuild-frontend` skill).
- **Slash commands arrive over awareness and the provider re-queries only on a
  keystroke.** Always pass the persona's commands to
  `slashCompletions(prefix, waitFor)`, or the read can race the awareness update.

## Linting (CI gate)

The repo's lint runs in CI and covers `ui-tests/`. Before pushing, run the root
`jlpm lint` (prettier + eslint) from the repo root so an unformatted spec or
fixture doesn't fail the build.
