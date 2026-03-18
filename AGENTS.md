# Introduction
Repository name: `open-physical-ai`
Package name: `opai`

# Architectural Design
- Follow Domain Driven Design.
- Design for Jupyter-native usage first.
- Optimize the public API for short, notebook-friendly calls.

## Principle
Users interact with `opai` directly from notebook cells. The package root is the public facade. Internal layers remain explicit in code organization, but notebook users should not need to call deep module paths for standard workflows.

## Layers

### Domain
- Holds core business concepts and rules.
- Contains entities, value objects, domain services, and domain invariants.
- Must not depend on notebook state, global context, filesystem access, or external tools.

### Application
- Implements use cases and workflow orchestration.
- Coordinates domain logic and infrastructure adapters.
- Accepts `Context` explicitly.
- Example internal APIs:
  - `opai.application.calibrate(ctx, frames)`
  - `opai.application.run_slam(ctx)`
  - `opai.application.run_pipeline(ctx, cfgs=None)`

### Infrastructure
- Implements filesystem access, serialization, artifact persistence, external process execution, and integrations with third-party libraries or tools.
- Owns reading and writing artifacts under the session directory.

### Presentation
- Exposes the notebook-facing API through `opai`.
- Hides architectural depth from the user for common tasks.
- Resolves the active global context and forwards calls into the application layer.

## Public API Design
The public API should be short and stable. Prefer this style (examples):

```python
ctx = opai.init(name: str) -> Context
opai.get_context() -> Context
opai.calibrate(frames)
opai.run_slam()
opai.run_pipeline()
```

Do not require users to call `opai.application.*` from notebooks for normal usage. `opai.application` is an internal architectural layer, even if it remains importable.

## Jupyter-Native Design

### Initialization
```python
ctx = opai.init(name: str) -> Context
```

Behavior:
- Creates and registers the active global context for the current notebook session.
- Creates a session directory for artifacts.
- Returns a `Context` object.

### Context
A `Context` is the notebook session state holder.

Responsibilities:
- Holds runtime variables for the current session.
- Stores `session_directory`, where all artifacts for the session live.
- Exposes paths and handles required by application use cases.
- Tracks session-level state needed across notebook cells.

Non-responsibilities:
- Should not contain core domain logic.
- Should not replace domain objects with unstructured state.

### Global Context Rules
- `opai.init(name)` sets the active context.
- Public facade calls such as `opai.calibrate(...)` operate on the active context.
- `opai.get_context()` returns the active context.
- If no active context exists, public operations should raise a clear error instructing the user to call `opai.init(...)` first.

The package should define the behavior for repeated initialization explicitly. Recommended default:
- Calling `opai.init(name)` again replaces the active in-memory context with a new one for that session name.
- If the session directory already exists, the implementation must define whether it resumes, overwrites, or errors. This behavior must be explicit and documented in code and user docs.

## Session Directory
Each context owns one session directory. All generated artifacts for that session are stored there.

Expected contents may include:
- `pipeline.yaml`
- `camera_trajectory.csv`
- `demos/`
- `mappings/`
- calibration outputs

Rules:
- Infrastructure code owns artifact creation and persistence.
- Artifact filenames and formats should be stable and documented.
- Application use cases should not hardcode ad hoc paths outside the context-owned session directory.

## Primary Use Cases

### 1. Start a session
```python
ctx = opai.init("session-001")
```

Outcome:
- A new active context exists.
- A session directory is available for artifact storage.

### 2. Run calibration
```python
opai.calibrate(frames)
```

Outcome:
- Calibration is executed against the active context.
- Calibration artifacts are persisted into the session directory.

### 3. Run SLAM
```python
opai.run_slam()
```

Outcome:
- SLAM consumes the required session inputs.
- Outputs such as trajectory artifacts are persisted.

### 4. Run pipeline
```python
opai.run_pipeline(cfgs=None)
```

Outcome:
- A higher-level workflow is executed using the active context.
- Pipeline configuration and outputs are stored under the session directory.

## Design Rules
- Keep notebook-facing calls short.
- Keep the application layer explicit in the codebase, but not as the primary UX surface.
- Pass `Context` explicitly inside internal layers.
- Avoid hidden filesystem writes outside the session directory.
- Make artifact creation deterministic where possible.
- Make errors notebook-friendly and actionable.
- Fail fast on required dependencies. If a feature depends on a package that is required by this repo, import it normally and let missing dependencies fail at import time rather than adding deferred runtime guards. Reserve lazy imports or fallback guards for truly optional integrations only.
- In `src/opai/presentation/facade.py`, prefer comprehensive public-function implementations over underscore-prefixed helper members. Keep the facade logic visible in the public functions rather than hiding it behind private abstractions unless the file would otherwise become unworkable.
- Do not import types from the `typing` module. Python 3.10+ native annotations are the repo standard.
- Prefer built-in generics such as `list[str]`, `dict[str, object]`, `tuple[int, ...]`, and unions like `Path | None`.
- When a protocol-style annotation is needed, import it from `collections.abc` instead of `typing`.

## Open Design Points To Resolve In Implementation
These decisions must be made consistently across the package:
- Whether existing session directories are resumed or rejected.
- Which fields live in `Context` versus dedicated domain models.
- Whether artifact files are the source of truth or persisted outputs derived from in-memory state.
- How users switch between multiple contexts in one notebook session, if supported.
