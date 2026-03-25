# Session Design Outline For `opai`

This note outlines how session state works in `opai` today.
It is written from the notebook user's point of view, but it also captures the current implementation boundaries so future work stays consistent with the repo architecture.

## 1. Design Intent

`opai` is designed for short, notebook-friendly calls at the package root.
The session model supports that by giving each notebook workflow a current session directory and a single active in-memory context.

The main goals are:

- keep notebook calls short, for example `opai.init(...)` and `opai.calibrate(...)`
- keep artifact writes inside one session-owned directory
- make session state explicit inside internal layers through `Context`
- allow sessions to be resumed by name

## 2. Core Concepts

### `Context`

The active runtime state is held in a `Context` object.
The current fields are:

- `name`
- `session_directory`
- `manifest_path`
- `gopro_socket_address`

`Context` is runtime state for the current Python process. It is not the full source of truth for all artifacts. Persisted outputs still live on disk under the session directory.

### `SessionManifest`

Structured session metadata is persisted in `session.json`.
The current manifest fields are:

- `session_name`
- `demos`
- `mapping`

Each demo entry stores:

- `demo_id`
- `source_path`
- `stored_path`
- `original_filename`

The mapping entry stores:

- `source_path`
- `stored_path`
- `original_filename`

## 3. Session Lifecycle

### Start Or Resume A Session

Notebook workflows start with:

```python
import opai

ctx = opai.init("session-001")
```

Current behavior:

- validates the session name at the public facade
- creates or resumes `sessions/<name>/`
- ensures the standard directory structure exists
- loads `session.json` if present, otherwise creates a default manifest
- sets that context as the active in-memory context

Repeated initialization is currently allowed.
Calling `opai.init(...)` again replaces the active in-memory context with the newly requested session and resumes that session directory if it already exists.

### Get The Active Session

Notebook-facing code can inspect the active context with:

```python
ctx = opai.get_context()
```

If no active context exists, facade operations raise an `OPAIContextError` that tells the user to call `opai.init(...)` first.

## 4. Session Directory Model

The session root is anchored to the current working directory:

```text
sessions/
```

Each session lives under:

```text
sessions/<name>/
```

The standard directory structure created during initialization is:

```text
sessions/<name>/
├── session.json
├── captures/
│   ├── demos/
│   └── mapping/
└── gopro_thumbnails/
```

Additional workflow artifacts are written into the same session directory.
Current examples include:

- `charuco_board.png`
- `charuco_config.json`
- `calibration.json`
- `calibration_verification.json`

### Capture Storage Rules

Demo videos are copied into per-demo folders:

```text
captures/demos/demo-0001/<original-file>
```

Mappings are treated as a single active asset and stored at:

```text
captures/mapping/current/<original-file>
```

Adding a new mapping replaces the previous `current` mapping directory.

## 5. Public Session API

The current session-related notebook-facing API is:

```python
ctx = opai.init("session-001")
ctx = opai.get_context()
assets = opai.add_demos([...])
mapping = opai.add_mapping("mapping.mp4")
names = opai.list_sessions()
files = opai.browse_session("session-001")
```

### `opai.list_sessions()`

This call:

- discovers session directories under `sessions/`
- sorts them by name
- marks the active session as `current`
- prints a Rich tree summary for notebook/terminal use
- returns the session names as `list[str]`

### `opai.browse_session(name)`

This call:

- reads the requested session without changing the active context
- prints a Rich tree view of that session's files
- returns relative file paths as `list[str]`

Both browsing functions currently depend on `rich`.

## 6. Internal Layer Responsibilities

The session design follows the repo's layered architecture.

### Presentation

The package root and facade:

- validate notebook-facing inputs
- resolve the active context
- expose short APIs such as `opai.init(...)`

### Application

Application code:

- orchestrates session use cases
- accepts `Context` explicitly
- computes session summaries and browse views

### Infrastructure

Infrastructure code:

- creates the session directory structure
- loads and writes `session.json`
- copies demo and mapping assets
- lists session files and builds file trees

### Domain

Domain code defines the session-related data objects such as:

- `Context`
- `DemoAsset`
- `MappingAsset`
- `SessionManifest`

## 7. Behavioral Invariants

The current implementation implies these rules:

- one active context exists at most per Python process
- standard notebook workflows should use the active context rather than passing deep paths around
- artifact writes should stay under the session directory
- session names should be stable and filesystem-safe
- hidden directories under `sessions/` are ignored when listing sessions
- the manifest records structured demo and mapping metadata, while the filesystem stores the actual artifacts

## 8. Current Constraints And Follow-Ups

This is the current design, not a final long-term contract.
The main constraints and open design points are:

- session root is tied to `Path.cwd()`, so changing the working directory changes the session root
- active-context state is process-local and in-memory
- switching sessions is currently done by calling `opai.init(...)` again
- manifest data and filesystem contents both matter, so the exact source-of-truth boundary should stay explicit as the package grows
- future workflows should continue to store artifacts inside the session directory instead of introducing ad hoc external paths

## 9. Recommended Reader Outcomes

After reading this outline, an engineer should be able to answer:

- what a session is in `opai`
- how a notebook activates or resumes a session
- where artifacts are stored
- which session behaviors are public API versus internal implementation
- which invariants new features should preserve
