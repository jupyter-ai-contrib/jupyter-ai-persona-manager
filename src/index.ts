import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

// Public awareness API: typed, read-only views of the persona-manager and
// persona awareness slots, for consumers building on the awareness channel.
export * from './awareness';

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

export default plugin;
