"""
Shared test fixtures for jupyter_ai_persona_manager tests.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from jupyter_server.serverapp import ServerApp
from jupyter_server_fileid.manager import BaseFileIdManager

if TYPE_CHECKING:
    from jupyterlab_chat.ychat import YChat


@pytest.fixture
def tmp_dir():
    """Create a temporary directory with guaranteed cleanup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_ychat():
    """Create a mock YChat instance."""
    # Import at runtime to avoid circular import
    from jupyterlab_chat.ychat import YChat
    mock = Mock(spec=YChat)
    mock.get_id.return_value = "test-chat-id"
    mock.set_user = Mock()
    mock.add_message = Mock(return_value="msg-123")
    mock.update_message = Mock()
    mock.awareness = Mock()
    mock._ydoc = Mock()
    mock._yusers = {}
    mock._background_tasks = set()
    return mock


@pytest.fixture
def mock_fileid_manager():
    """Create a mock BaseFileIdManager."""
    mock = Mock(spec=BaseFileIdManager)
    mock.get_path.return_value = "test/path/chat.ipynb"
    return mock


@pytest.fixture
def mock_server_app(mock_fileid_manager):
    """Create a mock ServerApp with basic settings."""
    mock_app = Mock(spec=ServerApp)
    mock_app.web_app = Mock()
    mock_app.web_app.settings = {
        "file_id_manager": mock_fileid_manager,
        "jupyter-ai": {}
    }
    mock_app.contents_manager = Mock()
    mock_app.contents_manager.root_dir = "/test/root"
    mock_app.log = Mock()
    return mock_app