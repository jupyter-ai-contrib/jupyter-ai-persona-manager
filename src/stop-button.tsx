import React, { useEffect, useState } from 'react';
import StopIcon from '@mui/icons-material/Stop';
import { InputToolbarRegistry, TooltippedIconButton } from '@jupyter/chat';
import { cancelResponse } from './request';

const STOP_BUTTON_CLASS = 'jp-jai-stopButton';

/**
 * A stop button for the chat input toolbar. Observes the chat model's
 * writers list to enable itself when an AI persona is actively writing,
 * and calls the persona cancel endpoint on click.
 */
export function StopButton(
  props: InputToolbarRegistry.IToolbarItemProps
): JSX.Element {
  const { chatModel } = props;
  const [disabled, setDisabled] = useState(true);
  const [inFlight, setInFlight] = useState(false);
  const tooltip = 'Stop generating';

  useEffect(() => {
    if (!chatModel) {
      setDisabled(true);
      return;
    }

    const checkWriters = () => {
      const hasAIWriter = chatModel.writers.some(w => w.user.bot);
      setDisabled(!hasAIWriter);
    };

    checkWriters();
    chatModel.writersChanged?.connect(checkWriters);

    return () => {
      chatModel.writersChanged?.disconnect(checkWriters);
    };
  }, [chatModel]);

  async function handleStop() {
    if (!chatModel) {
      return;
    }

    setInFlight(true);
    try {
      await cancelResponse(chatModel.name);
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
