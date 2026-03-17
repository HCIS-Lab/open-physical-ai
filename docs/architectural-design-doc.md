# Architecture Extension Draft

## Purpose

This note records the recommended package placement for two planned capabilities:

- simulator support
- PyTorch dataset and training support

The recommendation follows the repository's Domain Driven Design split and the notebook-first API in `AGENTS.md`.

## Design Rule

Keep the notebook-facing API short at the `opai` package root. Put orchestration in `application`, engine and framework integrations in `infrastructure`, and only simulator-independent or training-independent business concepts in `domain`.

## Simulator Placement

### Recommended layers

- `src/opai/infrastructure/simulator/`
- `src/opai/application/simulation.py`
- `src/opai/presentation/facade.py`
- `src/opai/__init__.py`

### Rationale

A simulator is primarily an external runtime integration. It will likely own process lifecycle, adapter code, scene loading, configuration translation, filesystem artifacts, and communication with third-party tools. That is infrastructure work.

The application layer should define notebook-relevant use cases that accept `Context` explicitly, for example:

- `start_simulation(ctx, ...)`
- `run_rollout(ctx, ...)`
- `generate_demo(ctx, ...)`

The presentation layer should expose short facade calls such as:

- `opai.simulate(...)`
- `opai.run_simulation(...)`

### Domain boundary

Only add simulator-related code to `domain` if the concept is independent of the simulator engine, for example:

- `SimulationConfig`
- `RobotState`
- `Trajectory`
- task-level invariants

Do not put engine-specific APIs, process control, or scene adapters in `domain`.

### Suggested layout

```text
src/opai/domain/simulation.py
src/opai/application/simulation.py
src/opai/infrastructure/simulator/
src/opai/presentation/facade.py
```

## PyTorch Dataset And Training Placement

### Recommended layers

- `src/opai/domain/training.py`
- `src/opai/application/training.py`
- `src/opai/infrastructure/training/`
- `src/opai/presentation/facade.py`
- `src/opai/__init__.py`

### Rationale

PyTorch datasets, dataloaders, model wrappers, checkpoint handling, and training loops are framework-specific integrations. In this architecture they belong in `infrastructure`, not `domain`.

The application layer should expose training use cases that accept `Context` explicitly, for example:

- `prepare_dataset(ctx, ...)`
- `train_model(ctx, ...)`
- `evaluate_model(ctx, ...)`

The presentation layer should keep notebook usage short, for example:

- `opai.prepare_dataset(...)`
- `opai.train(...)`
- `opai.evaluate(...)`

### Domain boundary

Add training code to `domain` only for framework-agnostic concepts such as:

- `TrainingConfig`
- `DatasetSplit`
- `CheckpointRef`
- metric value objects
- domain invariants around dataset composition or experiment rules

Do not place `torch.utils.data.Dataset`, `DataLoader`, device logic, or trainer implementations in `domain`.

### Suggested layout

```text
src/opai/domain/training.py
src/opai/application/training.py
src/opai/infrastructure/training/
    datasets.py
    dataloaders.py
    trainer.py
    checkpoints.py
    serialization.py
```

## Public API Direction

The package root should remain the primary notebook UX surface. Users should not need to call deep module paths for standard workflows.

Recommended facade style:

```python
ctx = opai.init("session-001")
opai.simulate(...)
opai.prepare_dataset(...)
opai.train(...)
opai.evaluate(...)
```

Internal calls should continue to pass `Context` explicitly inside the application layer.

## Session Directory Guidance

Both simulator outputs and training artifacts should live under the active context's session directory. Infrastructure code should own these paths and keep filenames stable.

Likely artifact areas:

- simulation rollouts
- generated demos
- dataset manifests or split manifests
- checkpoints
- evaluation reports

Application code should avoid ad hoc writes outside the session directory.

## Recommended Next Step

If these capabilities are added soon, create the following modules first even if they start small:

- `src/opai/application/simulation.py`
- `src/opai/application/training.py`
- `src/opai/infrastructure/simulator/__init__.py`
- `src/opai/infrastructure/training/__init__.py`

That preserves architectural boundaries before the implementation grows.

