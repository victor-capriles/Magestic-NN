---
name: developer-experience
description: Use when setting up, standardizing, or repairing the local developer workflow for a Python repository. Consult this whenever the user wants Poetry dependency groups, Ruff, MyPy, pytest, Invoke tasks, pre-commit hooks, or README workflow instructions added or cleaned up, even if they only ask for "developer experience", "tooling", "linting", or "project setup."
---

# Python Developer Workflow Setup

## Goal

Create one predictable local workflow for installing dependencies, formatting code, linting, type checking, running tests, and enforcing the same baseline at commit time.

The important outcome is not "the repo has tools installed." The outcome is that a developer can open the repository, read one short section of the README, and know exactly which commands to run.

## When To Use This Skill

Use this skill for:

- new Python repositories that need a clean local workflow
- existing Python repositories with partial or inconsistent tooling
- repos where the user wants stable `inv ...` commands for routine work
- repos that should use Poetry as the dependency source of truth

Do not force this workflow onto a repository that already has an established alternative such as `uv`, `tox`, `nox`, or a non-Poetry dependency layout unless the user explicitly wants that migration.

## Start With Repo Reality

Before changing files, inspect the repository and decide what you are standardizing around.

Check:

- whether the repo already uses Poetry, pip, uv, Hatch, Conda, or another environment manager
- whether the package layout is `src/`, flat package, or script-based
- whether tests are synchronous, asynchronous, or mixed
- whether the repo expects an activated environment or prefers `poetry run ...`
- whether CI already runs linting, type checks, or tests
- whether strict docstring rules are realistic right now or would just create churn

Pick one environment policy and keep it consistent across `tasks.py`, the README, and any contributor guidance.

## Required Deliverables

Leave the repository with these pieces in place:

1. `pyproject.toml` defines developer dependencies and tool behavior.
2. `tasks.py` exposes stable command names.
3. `invoke.yaml` makes task execution predictable.
4. `.pre-commit-config.yaml` enforces the same baseline before commit.
5. `README.md` documents installation and the everyday workflow.

If the repository already has equivalents, improve them instead of duplicating them.

## 1. Configure `pyproject.toml`

Use `pyproject.toml` as the source of truth for both dependencies and tool configuration.

Recommended developer tools:

- `ruff` for formatting and linting
- `mypy` for type checking
- `pytest` for tests
- `pytest-asyncio` only if the repo has async tests
- `invoke` for stable local commands
- `pre-commit` for commit-time enforcement

Example Poetry groups:

```toml
[tool.poetry.group.checks.dependencies]
mypy = "^1.10.1"
pytest = "^8.2.2"
ruff = "^0.5.0"

[tool.poetry.group.dev.dependencies]
invoke = "^2.2.0"

[tool.poetry.group.commits.dependencies]
pre-commit = "^3.7.1"
```

Add `pytest-asyncio` only when the test suite actually needs it.

Add tool configuration that matches the repository layout rather than copying defaults blindly:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

[tool.ruff]
fix = true
indent-width = 4
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

Adjust these decisions per repo:

- whether tests should be type-checked
- which directories MyPy and Ruff should exclude
- whether imports require a `src/` path adjustment
- how strict linting should be on day one
- whether docstring rules are part of the initial rollout or a later cleanup step

Prefer a baseline the repository can realistically pass after a focused cleanup pass. Do not introduce a rule set so strict that the team immediately learns to ignore it.

If you mark Poetry groups as optional, document the exact install command the repo now expects:

```bash
poetry install --with checks,dev,commits
```

Do not copy a lockfile from another repository. Declare the dependencies and generate the lockfile in the target repo.

## 2. Add Stable Invoke Tasks In `tasks.py`

The point of `tasks.py` is to give every repository the same command surface even when the internal paths differ.

Required commands:

- `inv format`
- `inv lint`
- `inv check-quality`
- `inv test`

Useful optional commands:

- `inv check-format`
- `inv clean`
- `inv list-python-files`

Default behavior:

- `format` mutates files with Ruff formatting
- `lint` runs Ruff checks and MyPy
- `check-quality` is non-mutating and should fail fast
- `test` runs pytest
- `clean` removes caches and local build artifacts

Example structure using Poetry-run commands:

```python
from invoke import task
from invoke.context import Context

QUALITY_PATHS = "./src ./tests ./main.py"


@task
def format(c: Context) -> None:
    c.run(f"poetry run ruff format {QUALITY_PATHS}")


@task
def lint(c: Context) -> None:
    c.run(f"poetry run ruff check {QUALITY_PATHS}")
    c.run("poetry run mypy ./src ./main.py --config-file pyproject.toml --show-error-codes")


@task(name="check-quality")
def check_quality(c: Context) -> None:
    c.run(f"poetry run ruff format --check {QUALITY_PATHS}")
    lint(c)


@task
def test(c: Context) -> None:
    c.run("poetry run pytest tests")
```

If the repo assumes an activated environment, drop the `poetry run` prefix everywhere instead of mixing both styles.

Adjust commands for the repo shape:

- add `pytest-asyncio` support only when needed
- widen or narrow target paths to match the actual package layout
- set `PYTHONPATH=.` in the test command only when the layout requires it
- keep helper tasks small and obvious rather than building a task framework

## 3. Add `invoke.yaml`

Keep Invoke behavior explicit and easy to read:

```yaml
run:
  echo: true
project:
  name: Example Project
  repository: example-project
```

`run.echo: true` matters because it makes local task execution transparent. Developers should be able to see the exact commands behind `inv check-quality` or `inv test` without guessing.

## 4. Add `.pre-commit-config.yaml`

Commit-time hooks prevent easy-to-avoid drift.

Recommended baseline:

```yaml
default_language_version:
  python: python3.12
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
      - id: check-case-conflict
      - id: check-merge-conflict
      - id: check-toml
      - id: check-yaml
      - id: debug-statements
      - id: end-of-file-fixer
      - id: mixed-line-ending
      - id: trailing-whitespace
  - repo: https://github.com/python-poetry/poetry
    rev: 1.8.3
    hooks:
      - id: poetry-check
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
      - id: ruff-format
```

Add more hooks only when they reflect the repo's actual workflow. Pre-commit should reinforce the normal path, not become a second independent toolchain.

## 5. Document The Workflow In `README.md`

If the workflow is not documented, it is not standard.

The README should show:

- how to activate or prepare the expected environment
- how to install dependencies
- how to start the application or service if relevant
- how to run `inv check-quality`
- how to run `inv test`
- what contributors should run before opening a PR

Example structure:

````md
## Local Development

```bash
poetry install --with checks,dev,commits
python main.py
```

## Developer Workflow

```bash
inv check-quality
inv format
inv lint
inv test
pre-commit run --all-files
```
````

Keep the README aligned with the environment policy you chose. If you document `poetry run ...`, use it consistently. If you document an activated environment, do not sprinkle in `poetry run` commands arbitrarily.

## Rollout Sequence

Use this order when applying the workflow:

1. Update `pyproject.toml` with the dependency groups and tool configuration.
2. Add or refine `tasks.py`.
3. Add `invoke.yaml`.
4. Add `.pre-commit-config.yaml`.
5. Update `README.md`.
6. Install dependencies.
7. Run the formatter once.
8. Run `inv check-quality`.
9. Run `inv test`.
10. Fix the issues surfaced by the stricter tooling.

The cleanup pass is part of the work. Tooling that immediately fails and stays broken does not improve developer experience.

## Validation Checklist

Validate the workflow end to end:

```bash
inv --list
inv check-quality
inv test
pre-commit run --all-files
```

If the repository standard is Poetry-wrapped commands, run the documented `poetry run ...` equivalents instead.

## Common Mistakes

- copying tool configuration from another repo without adjusting paths or strictness
- mixing activated-environment commands with `poetry run` commands in the same repo docs
- enabling async tooling in a fully synchronous repo
- forgetting to account for `src/` layout imports during tests
- creating different task names in each repo and losing the shared `inv` workflow
- adding strict checks but not budgeting time to make the repo pass them

## Core Principle

Set up a workflow, not just a package list.

The repository should make these things obvious from day one:

- how developers install it
- which commands they run every day
- how quality checks are enforced
- how tests are executed
- how commit hooks match the same baseline

That consistency is the real developer experience.