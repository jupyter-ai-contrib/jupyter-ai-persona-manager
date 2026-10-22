import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { Toolbar } from '@jupyterlab/ui-components';
import {
  IChatCommandRegistry,
  IInputToolbarRegistryFactory,
  InputToolbarRegistry
} from '@jupyter/chat';

import {
  PersonaPicker,
  PersonaUsageItem,
  PersonaControlItem,
  buildControls
} from './persona-controls';
import {
  IPersonaControlRegistry,
  PersonaControlRegistry,
  buildRegisteredControl
} from './persona-control-registry';
import {
  IPersonaSessionRegistry,
  PersonaSessionRegistry
} from './persona-events';
import { emptyPersonaSettings } from './metadata';
import { InputToolbarState } from './input-toolbar-state';
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

// Toolbar-side observable for the current persona selection.
export * from './input-toolbar-state';

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
 * fed by Jupyter Events via the `ServiceManager` event bus.
 */
const sessionRegistryPlugin: JupyterFrontEndPlugin<PersonaSessionRegistry> = {
  id: '@jupyter-ai/persona-manager:session-registry',
  description:
    'Provides the per-chat persona session-state registry (fed by Jupyter Events).',
  autoStart: true,
  provides: IPersonaSessionRegistry,
  activate: (app: JupyterFrontEnd): PersonaSessionRegistry => {
    return new PersonaSessionRegistry(app.serviceManager.events);
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
    const events = app.serviceManager.events;
    return {
      create: () => {
        // Start with the default toolbar (Send, Attach, Cancel, SaveEdit)
        const registry = InputToolbarRegistry.defaultToolbarRegistry();
        // Per-chat persona state.
        const inputToolbarState = new InputToolbarState(sessionRegistry);

        // Persona picker item.
        registry.addItem('persona', {
          position: 3,
          createWidget: InputToolbarRegistry.createReactToolbarItem(itemProps =>
            PersonaPicker({
              ...itemProps,
              inputToolbarState,
              sessionRegistry,
              events,
              commands: app.commands
            })
          )
        });

        // Tokens usage item.
        registry.addItem('persona-usage', {
          position: 4,
          createWidget: InputToolbarRegistry.createReactToolbarItem(itemProps =>
            PersonaUsageItem({
              ...itemProps,
              inputToolbarState: inputToolbarState
            })
          )
        });

        // Persona control items (model picker, settings)
        const personaControlNames = new Set<string>();
        const nameOfControl = (id: string) => `persona-control:${id}`;
        const reconcilePersonaControls = () => {
          const personaState = inputToolbarState.personaState;
          const settings = inputToolbarState.personaId
            ? (inputToolbarState.settings[inputToolbarState.personaId] ??
              emptyPersonaSettings())
            : emptyPersonaSettings();
          const controls = personaState
            ? buildControls(personaState, settings)
            : [];
          const nextNames = new Set(controls.map(c => nameOfControl(c.id)));

          // Clean previous items
          for (const name of Array.from(personaControlNames)) {
            if (!nextNames.has(name)) {
              registry.removeItem(name);
              personaControlNames.delete(name);
            }
          }

          // Add new items
          controls.forEach((control, idx) => {
            const name = nameOfControl(control.id);
            if (personaControlNames.has(name)) {
              return;
            }
            const controlId = control.id;
            registry.addItem(name, {
              position: 4.1 + idx * 0.0001,
              createWidget: InputToolbarRegistry.createReactToolbarItem(
                itemProps =>
                  PersonaControlItem({
                    ...itemProps,
                    inputToolbarState,
                    controlId
                  })
              )
            });
            personaControlNames.add(name);
          });
        };
        inputToolbarState.personaStateChanged.connect(reconcilePersonaControls);
        reconcilePersonaControls();

        // Add a spacer element.
        registry.addItem('persona-control-spacer', {
          createWidget: Toolbar.createSpacerItem,
          position: 5
        });

        // Registered controls items.
        const registeredControlNames = new Set<string>();
        const nameOf = (id: string) => `persona-registered-control:${id}`;
        const reconcileRegisteredControls = (forceRebuild = false) => {
          const id = inputToolbarState.personaId;
          const nextControls = id ? controlRegistry.getControls(id) : [];
          const nextControlNames = new Set(nextControls.map(c => nameOf(c.id)));

          // Clean previous items
          for (const name of Array.from(registeredControlNames)) {
            if (forceRebuild || !nextControlNames.has(name)) {
              registry.removeItem(name);
              registeredControlNames.delete(name);
            }
          }

          // Add new items
          nextControls.forEach((control, idx) => {
            const name = nameOf(control.id);
            if (registeredControlNames.has(name)) {
              return;
            }
            const position = 6 + idx * 0.0001;
            registry.addItem(
              name,
              buildRegisteredControl(control, id!, position)
            );
            registeredControlNames.add(name);
          });
        };

        inputToolbarState.changed.connect(() =>
          reconcileRegisteredControls(true)
        );
        controlRegistry.changed.connect(() => reconcileRegisteredControls());
        reconcileRegisteredControls();

        // Stop button item.
        registry.addItem('stop', {
          position: 7,
          createWidget: InputToolbarRegistry.createReactToolbarItem(itemProps =>
            StopButton({ ...itemProps, sessionRegistry })
          )
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
