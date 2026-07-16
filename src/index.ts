import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  IChatCommandRegistry,
  IInputToolbarRegistryFactory,
  InputToolbarRegistry
} from '@jupyter/chat';

import { PersonaControls } from './persona-controls';
import {
  IPersonaControlRegistry,
  PersonaControlRegistry
} from './persona-control-registry';
import {
  SLASH_COMMAND_PROVIDER_ID,
  SlashCommandProvider
} from './slash-commands';
import { StopButton } from './stop-button';

// Public awareness API: typed, read-only views of the persona-manager and
// persona awareness slots, for consumers building on the awareness channel.
export * from './awareness';

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
 * Plugin registering slash-command completions read from the selected
 * persona's awareness slot.
 */
const slashCommandPlugin: JupyterFrontEndPlugin<void> = {
  id: SLASH_COMMAND_PROVIDER_ID,
  description: 'Adds support for slash commands in Jupyter AI.',
  autoStart: true,
  requires: [IChatCommandRegistry],
  activate: (app: JupyterFrontEnd, registry: IChatCommandRegistry) => {
    registry.addProvider(new SlashCommandProvider());
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
  requires: [IPersonaControlRegistry],
  activate: (
    app: JupyterFrontEnd,
    controlRegistry: IPersonaControlRegistry
  ): IInputToolbarRegistryFactory => {
    // Wrap the persona controls to inject the control registry, which the
    // generic toolbar-item props don't carry. Contributed controls (e.g. a
    // persona's settings button) render for the selected persona.
    const PersonaControlsItem = (
      itemProps: InputToolbarRegistry.IToolbarItemProps
    ) => PersonaControls({ ...itemProps, controlRegistry });
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
  slashCommandPlugin,
  controlRegistryPlugin,
  toolbarPlugin
];
