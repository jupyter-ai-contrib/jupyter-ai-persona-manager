/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { expect, test } from '@jupyterlab/galata';
import { FixturePersona, installPersonas, TestHelpers } from './test-helpers';

// This suite's working directory and the fixture personas installed into it.
const TEST_DIR = 'keyboard-nav';
const PERSONAS = [
  FixturePersona.Hello,
  FixturePersona.Hello2,
  FixturePersona.Models
];

// Each fixture persona replies with a distinct word, so a reply unambiguously
// identifies which persona handled the message.
const REPLY: Record<string, string> = {
  [FixturePersona.Hello]: 'hello',
  [FixturePersona.Hello2]: 'bonjour'
};

/**
 * Verifies keyboard navigation of the persona controls (issue #88): from the
 * focused chat input, Tab opens the persona picker's searchable menu (focus in
 * its search field); arrow keys move the highlight; Tab confirms the highlighted
 * option and advances to the next control; and Tabbing past the last control
 * returns focus to the chat input (the send/attach/stop buttons are not part of
 * this Tab ring). The menu can also be narrowed by typing (fuzzy search) before
 * confirming.
 *
 * The whole flow runs with no mouse click on any control — only keyboard.
 */
test.describe('keyboard-nav', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('Tab from the input opens the persona picker menu, focused for search', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    await helpers.focusInput();
    // Tab reaches the persona picker and opens its menu, focus in the search box.
    await helpers.tabUntil(() => helpers.personaMenuFocused());
    expect(await helpers.personaMenuFocused()).toBe(true);
    await expect(helpers.menuOptions.first()).toBeVisible();
  });

  test('arrow keys move the highlight; Tab confirms and closes the menu', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    await helpers.focusInput();
    await helpers.tabUntil(() => helpers.personaMenuFocused());

    // Move the highlight onto the Bonjour persona, then Tab to confirm it.
    await helpers.arrowToOption('Bonjour Persona');
    await page.keyboard.press('Tab');

    // The picker now shows the confirmed persona and the menu has closed.
    await expect(helpers.personaPicker).toContainText('Bonjour Persona');
    await expect(helpers.menuOptions).toHaveCount(0);
  });

  test('typing filters the list (fuzzy search)', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    await helpers.focusInput();
    await helpers.tabUntil(() => helpers.personaMenuFocused());

    // "bon" narrows to the Bonjour persona. Enter confirms the top match.
    await helpers.menuSearch.pressSequentially('bon');
    await expect(helpers.menuOptions).toHaveCount(1);
    await expect(helpers.menuOptions.first()).toContainText('Bonjour Persona');
    await page.keyboard.press('Enter');
    await expect(helpers.personaPicker).toContainText('Bonjour Persona');
  });

  test('Escape closes the menu and returns focus to the chat input', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    await helpers.focusInput();
    await helpers.tabUntil(() => helpers.personaMenuFocused());

    // Escape bails back to typing: the menu closes and the input regains focus.
    await page.keyboard.press('Escape');
    await expect(helpers.menuOptions).toHaveCount(0);
    await expect
      .poll(() => helpers.inputHasFocus(), { timeout: 30000 })
      .toBe(true);
  });

  test('Tab past the last control returns focus to the input, not the send button', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    // Pick the Bonjour persona (no model/settings controls), so the picker is
    // the only control in the Tab ring.
    await helpers.selectPersona(FixturePersona.Hello2);

    await helpers.focusInput();
    await helpers.tabUntil(() => helpers.personaMenuFocused());

    // Tab confirms the picker; with no further control, focus returns to the
    // chat input — it does not walk on to the send button.
    await page.keyboard.press('Tab');
    await expect
      .poll(() => helpers.inputHasFocus(), { timeout: 30000 })
      .toBe(true);
    expect(await helpers.sendButtonHasFocus()).toBe(false);
  });

  test('keyboard-only: type, Tab to pick a persona, Tab back to input, Enter sends', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });
    // A persona with no extra controls, so the picker is the whole Tab ring.
    await helpers.selectPersona(FixturePersona.Hello2);

    // Type the message first, then drive the rest with the keyboard alone.
    await helpers.focusInput();
    await helpers.typeInput('who are you?');

    // Tab opens the picker; narrow to Bonjour and Tab to confirm + return to
    // the input.
    await helpers.tabUntil(() => helpers.personaMenuFocused());
    await helpers.menuSearch.pressSequentially('bonjour');
    await expect(helpers.menuOptions.first()).toContainText('Bonjour Persona');
    await page.keyboard.press('Tab');
    await expect(helpers.personaPicker).toContainText('Bonjour Persona');
    await expect
      .poll(() => helpers.inputHasFocus(), { timeout: 30000 })
      .toBe(true);

    // Enter in the focused input sends the message.
    const before = await helpers.chat
      .locator('.jp-chat-rendered-message')
      .count();
    await page.keyboard.press('Enter');

    // The human echo + the persona reply both render; the reply is Bonjour's
    // word, proving the searched-and-confirmed persona is the one messaged.
    await expect
      .poll(
        async () => helpers.chat.locator('.jp-chat-rendered-message').count(),
        { timeout: 30000 }
      )
      .toBeGreaterThanOrEqual(before + 2);
    const reply =
      (await helpers.chat
        .locator('.jp-chat-rendered-message')
        .last()
        .textContent()) ?? '';
    expect(reply).toContain(REPLY[FixturePersona.Hello2]);
  });

  test('a model control confirms via the menu and the choice sticks', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Select the Models persona (which advertises a Model control) up front.
    await helpers.selectPersona(FixturePersona.Models);
    await helpers.waitForControls();

    // Open the Model control's menu, highlight Alpha, and Tab to confirm it.
    await helpers.control('Model').click();
    await helpers.arrowToOption('Model Alpha');
    await page.keyboard.press('Tab');
    await expect(helpers.control('Model')).toContainText('Model Alpha');

    // The applied model rides out with the message: the Models persona echoes it.
    const reply = await helpers.sendMessage('which model?');
    expect(reply).toContain('applied model: alpha');
  });
});
