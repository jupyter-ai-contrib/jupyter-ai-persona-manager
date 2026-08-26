import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  IChatCommandRegistry,
  IInputToolbarRegistryFactory,
  InputToolbarRegistry
} from '@jupyter/chat';

import { IEventListener } from 'jupyterlab-eventlistener';

import { PersonaControls } from './persona-controls';
import {
  IPersonaControlRegistry,
  PersonaControlRegistry
} from './persona-control-registry';
import {
  IPersonaSessionRegistry,
  PersonaSessionRegistry
} from './persona-events';
import {
  SLASH_COMMAND_PROVIDER_ID,
  SlashCommandProvider
} from './slash-commands';
import { StopButton } from './stop-button';

// Public persona session data types (event payloads).
export * from './awareness';

// Public persona session-state models + registry (fed by Jupyter Events).
export * from './persona-events';

// Public API for contributing controls to the persona controls toolbar.
export * from './persona-control-registry';

/**
 * Initialization data for the @jupyter-ai/persona-manager extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/persona-manager:plugin',
  description: 'The core manager & registry for AI personas in Jupyter AI',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log(
      'JupyterLab extension @jupyter-ai/persona-manager is activated!'
    );
  }
};

/**
 * Plugin that provides the shared per-chat persona session-state registry,
 * fed by Jupyter Events via `jupyterlab-eventlistener`.
 */
const sessionRegistryPlugin: JupyterFrontEndPlugin<PersonaSessionRegistry> = {
  id: '@jupyter-ai/persona-manager:session-registry',
  description:
    'Provides the per-chat persona session-state registry (fed by Jupyter Events).',
  autoStart: true,
  provides: IPersonaSessionRegistry,
  requires: [IEventListener],
  activate: (
    app: JupyterFrontEnd,
    eventListener: IEventListener
  ): PersonaSessionRegistry => {
    return new PersonaSessionRegistry(eventListener);
  }
};

/**
 * Plugin registering slash-command completions read from the selected
 * persona's session state (fed by Jupyter Events).
 */
const slashCommandPlugin: JupyterFrontEndPlugin<void> = {
  id: SLASH_COMMAND_PROVIDER_ID,
  description: 'Adds support for slash commands in Jupyter AI.',
  autoStart: true,
  requires: [IChatCommandRegistry, IPersonaSessionRegistry],
  activate: (
    app: JupyterFrontEnd,
    registry: IChatCommandRegistry,
    sessionRegistry: PersonaSessionRegistry
  ) => {
    registry.addProvider(new SlashCommandProvider(sessionRegistry));
  }
};

/**
 * Plugin that provides the persona control registry, the extension point other
 * extensions use to contribute controls (e.g. a persona's settings button) to
 * the persona controls in the chat input toolbar.
 */
const controlRegistryPlugin: JupyterFrontEndPlugin<IPersonaControlRegistry> = {
  id: '@jupyter-ai/persona-manager:control-registry',
  description:
    'Provides the registry for contributing controls to the persona controls toolbar.',
  autoStart: true,
  provides: IPersonaControlRegistry,
  activate: (): IPersonaControlRegistry => {
    return new PersonaControlRegistry();
  }
};

/**
 * Plugin that provides the chat input toolbar factory: the default toolbar
 * plus the persona controls (picker, model, settings, usage) and a stop
 * button. The chat panel picks this up and uses it to build the toolbar for
 * each chat.
 */
const toolbarPlugin: JupyterFrontEndPlugin<IInputToolbarRegistryFactory> = {
  id: '@jupyter-ai/persona-manager:input-toolbar',
  description: 'Provides the chat input toolbar with persona controls.',
  autoStart: true,
  provides: IInputToolbarRegistryFactory,
  requires: [IPersonaControlRegistry, IPersonaSessionRegistry],
  activate: (
    app: JupyterFrontEnd,
    controlRegistry: IPersonaControlRegistry,
    sessionRegistry: PersonaSessionRegistry
  ): IInputToolbarRegistryFactory => {
    // The event bus front-end; `PersonaControls` uses it to emit
    // `persona_selected` on selection.
    const events = app.serviceManager.events;
    // Wrap the persona controls to inject the control + session registries and
    // the event manager, which the generic toolbar-item props don't carry.
    const PersonaControlsItem = (
      itemProps: InputToolbarRegistry.IToolbarItemProps
    ) =>
      PersonaControls({
        ...itemProps,
        controlRegistry,
        sessionRegistry,
        events
      });
    return {
      create: () => {
        // Start with the default toolbar (Send, Attach, Cancel, SaveEdit)
        const registry = InputToolbarRegistry.defaultToolbarRegistry();
        // Add the active-persona controls (persona + model), leftmost.
        registry.addItem('persona', {
          element: PersonaControlsItem,
          position: 5
        });
        // Stop button, between the persona controls (5) and the default
        // toolbar's cancel button (10); a tie would leave the order to
        // insertion rather than position.
        registry.addItem('stop', {
          element: StopButton,
          position: 7
        });
        return registry;
      }
    };
  }
};

export default [
  plugin,
  sessionRegistryPlugin,
  slashCommandPlugin,
  controlRegistryPlugin,
  toolbarPlugin
];
