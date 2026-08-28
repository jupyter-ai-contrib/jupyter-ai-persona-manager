import React, { useEffect, useState } from 'react';
import StopIcon from '@mui/icons-material/Stop';
import { InputToolbarRegistry, TooltippedIconButton } from '@jupyter/chat';
import { cancelResponse } from './request';
import { PersonaSessionRegistry } from './persona-events';

const STOP_BUTTON_CLASS = 'jp-jai-stopButton';

/**
 * A stop button for the chat input toolbar. Enables itself while a persona in
 * the chat is processing a message — tracked via the persona-state events fed
 * into the `PersonaSessionRegistry` — and calls the persona cancel endpoint on
 * click.
 *
 * This tracks *processing* rather than the chat's writers list: a persona can
 * be processing (thinking, running tools, awaiting an agent turn) without
 * actively writing, and should be interruptible the whole time.
 */
export function StopButton(
  props: InputToolbarRegistry.IToolbarItemProps & {
    /**
     * The per-chat persona session-state registry (fed by Jupyter Events).
     * Optional so the button still renders (permanently disabled) without it.
     */
    sessionRegistry?: PersonaSessionRegistry;
  }
): JSX.Element {
  const { chatModel, sessionRegistry } = props;
  const [disabled, setDisabled] = useState(true);
  const [inFlight, setInFlight] = useState(false);
  const tooltip = 'Stop generating';

  // The chat's stable id is assigned asynchronously (once the model is
  // `ready`), so track it in state and update it when the model becomes ready.
  const [chatId, setChatId] = useState<string | null>(chatModel?.id ?? null);
  useEffect(() => {
    if (!chatModel) {
      setChatId(null);
      return;
    }
    let cancelled = false;
    void chatModel.ready.then(id => {
      if (!cancelled) {
        setChatId(id ?? null);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [chatModel]);

  // Enable the button while any persona in this chat is processing. The
  // manager session state fires `changed` on every persona-state update, which
  // includes the processing transitions emitted by `track_processing`.
  useEffect(() => {
    if (!sessionRegistry || !chatId) {
      setDisabled(true);
      return;
    }
    const managerState = sessionRegistry.get(chatId);
    const sync = () => setDisabled(!managerState.processing);
    sync();
    managerState.changed.connect(sync);
    return () => {
      managerState.changed.disconnect(sync);
    };
  }, [sessionRegistry, chatId]);

  async function handleStop() {
    if (!chatId) {
      return;
    }

    setInFlight(true);
    try {
      await cancelResponse(chatId);
    } finally {
      setInFlight(false);
    }
  }

  return (
    <TooltippedIconButton
      onClick={handleStop}
      tooltip={tooltip}
      disabled={disabled || inFlight}
      buttonProps={{
        title: tooltip,
        className: STOP_BUTTON_CLASS
      }}
      aria-label={tooltip}
    >
      <StopIcon />
    </TooltippedIconButton>
  );
}
