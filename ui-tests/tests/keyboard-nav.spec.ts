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
 * focused chat input, Tab reaches the persona picker; arrow keys change the
 * selection live (like a native <select>) without opening a menu; Tab moves
 * through the controls to the send button; and Enter on the send button sends.
 *
 * The whole flow runs with no mouse click on any control — only keyboard.
 */
test.describe('keyboard-nav', () => {
  test.beforeAll(async ({ request }) => {
    await installPersonas(request, TEST_DIR, PERSONAS);
  });

  test('Tab from the input focuses the persona picker', async ({ page }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    await helpers.focusInput();
    // Tab lands on the persona picker (the leftmost, first-in-DOM toolbar item).
    await helpers.tabUntil(() => helpers.pickerHasFocus());
    expect(await helpers.pickerHasFocus()).toBe(true);
  });

  test('arrow keys change the persona selection live, without opening a menu', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    await helpers.focusInput();
    await helpers.tabUntil(() => helpers.pickerHasFocus());

    // With several personas none is auto-selected; the picker starts on "No one".
    await expect(helpers.personaPicker).toContainText('No one');

    // ArrowDown from "No one" (the last value) is a no-op at the end; ArrowUp
    // walks back up the list. Step up to the previous persona and confirm the
    // button label changed with the popup still closed (aria-expanded false).
    await page.keyboard.press('ArrowUp');
    await expect(helpers.personaPicker).not.toContainText('No one');
    await expect(helpers.personaPicker).toHaveAttribute(
      'aria-expanded',
      'false'
    );
  });

  test('keyboard-only: Tab to picker, arrow-select a persona, Tab to Send, Enter sends', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();
    await expect(helpers.personaPicker).toBeVisible({ timeout: 30000 });

    // Type the message first, then drive the rest with the keyboard alone.
    await helpers.focusInput();
    await helpers.typeInput('who are you?');

    // Tab to the persona picker, then arrow-select the Bonjour persona (the
    // helper steps one at a time regardless of the persona-list order).
    await helpers.tabUntil(() => helpers.pickerHasFocus());
    await helpers.arrowToPersona(FixturePersona.Hello2);
    await expect(helpers.personaPicker).toContainText('Bonjour Persona');

    // Tab onward until the send button holds focus, then send with Enter.
    await helpers.tabUntil(() => helpers.sendButtonHasFocus());
    const before = await helpers.chat
      .locator('.jp-chat-rendered-message')
      .count();
    await page.keyboard.press('Enter');

    // The human echo + the persona reply both render; the reply is Bonjour's
    // word, proving the arrow-selected persona is the one that was messaged.
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

  test('arrow keys cycle a model control live and the choice sticks', async ({
    page
  }) => {
    const helpers = new TestHelpers({ dir: TEST_DIR, page });
    await helpers.openChat();

    // Select the Models persona (which advertises a Model control) up front; the
    // model-control keyboard nav is what this test exercises.
    await helpers.selectPersona(FixturePersona.Models);
    await helpers.waitForControls();

    // From the focused input, Tab past the picker to the Model control. It
    // starts on the persona's current model (Beta) — its selection is null
    // ("Default"), so ArrowDown steps Default -> Alpha (menu order is
    // Default(Beta), Alpha, Beta, Gamma).
    await helpers.focusInput();
    await helpers.tabUntil(() =>
      page.evaluate(
        () =>
          document.activeElement
            ?.getAttribute('aria-label')
            ?.startsWith('Model:') ?? false
      )
    );
    await page.keyboard.press('ArrowDown'); // Default -> Alpha
    await expect(helpers.control('Model')).toContainText('Model Alpha');
    // No popup opened while cycling (the control's menu stays closed).
    await expect(helpers.control('Model')).toHaveAttribute(
      'aria-expanded',
      'false'
    );

    // The applied model rides out with the message: the Models persona echoes it.
    const reply = await helpers.sendMessage('which model?');
    expect(reply).toContain('applied model: alpha');
  });
});
