# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build & Watch

- `jlpm dev:install` - Full development install (install dependencies, build, and link extension)
- `jlpm build` - Build TypeScript and create labextension for development
- `jlpm build:prod` - Production build (clean first, build optimized)
- `jlpm watch` - Watch TypeScript source and auto-rebuild
- `jlpm watch:src` - Watch TypeScript source only
- `jlpm watch:labextension` - Watch and rebuild labextension

### Testing

- `pytest -vv -r ap --cov jupyter_ai_persona_manager` - Run Python tests with coverage
- `jlpm test` - Run TypeScript tests with Jest
- Playwright integration tests are in `ui-tests/` directory

### Code Quality

- `jlpm lint` - Run all linters (stylelint + prettier + eslint) with fixes
- `jlpm lint:check` - Check without fixing
- `jlpm eslint` - ESLint with fixes
- `jlpm prettier` - Prettier with fixes
- `jlpm stylelint` - Stylelint with fixes

### Development Lifecycle

- `jlpm dev:uninstall` - Uninstall development extension
- `jlpm clean:all` - Clean all build artifacts

## Architecture Overview

This is a JupyterLab extension that provides AI persona management for Jupyter AI. It's a hybrid TypeScript/Python package with both frontend extension and server extension components.

### Core Components

**BasePersona** (`jupyter_ai_persona_manager/base_persona.py`):

- Abstract base class for all AI personas
- Handles chat integration, awareness management, and message processing
- Provides utilities for file attachment processing and workspace access
- Key methods: `process_message()`, `stream_message()`, `send_message()`

**PersonaManager** (`jupyter_ai_persona_manager/persona_manager.py`):

- Central registry and lifecycle manager for personas
- Loads personas from both Python entry points and local `.jupyter/personas/` directory
- Routes messages to appropriate personas based on @-mentions and chat context
- Handles multi-user vs single-user chat scenarios differently

**Persona Discovery**:

- Entry points: Uses `jupyter_ai.personas` entry point group for installed packages
- Local loading: Scans `.jupyter/personas/` for Python files containing "persona" in filename
- Supports `/refresh-personas` command for development iteration

### Extension Structure

**Frontend** (`src/`):

- TypeScript JupyterLab extension
- Minimal frontend - most logic is server-side
- Registers server extension and handles activation

**Server Extension** (`jupyter_ai_persona_manager/`):

- Python server extension that integrates with JupyterLab Chat
- Manages persona instances per chat room
- Handles message routing and persona lifecycle

### Key Integration Points

**Chat Integration**:

- Integrates with `jupyterlab_chat` for message handling
- Uses `YChat` (Yjs-based collaborative chat) for real-time messaging
- Persona awareness uses pycrdt for collaborative state management

**File System Access**:

- Personas can access workspace directory and .jupyter directory
- File attachment processing with multiple resolution strategies
- Integration with `jupyter_server_fileid` for file ID management

**Entry Point System**:

- Personas registered via `[project.entry-points."jupyter_ai.personas"]` in pyproject.toml
- Automatic discovery and loading of persona classes from installed packages

### Development Patterns

**Persona Development**:

- Inherit from `BasePersona` and implement `defaults` property and `process_message()` method
- Use `self.send_message()` or `self.stream_message()` for responses
- Access file attachments via `self.process_attachments(message)`
- Use `self.awareness` for collaborative state (typing indicators, etc.)

**Local Development**:

- Place persona files in `.jupyter/personas/` directory
- Use `/refresh-personas` command to reload without server restart
- Files must contain "persona" in name and not start with `_` or `.`

### Configuration

**Traitlets Configuration**:

- `PersonaManager.default_persona_id` - Sets which persona responds in single-user chats
- Inherits JupyterLab's configurable system via `LoggingConfigurable`

**Code Style**:

- TypeScript: ESLint + Prettier with interface naming convention (must start with 'I')
- Python: Standard conventions with pytest for testing
- Single quotes preferred, no trailing commas in TypeScript
