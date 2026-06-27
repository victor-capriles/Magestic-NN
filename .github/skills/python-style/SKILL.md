---
name: python-style
description: General Python style guidance for writing, refactoring, and reviewing Python code. Use this skill whenever the task needs conventions for imports, docstrings, typing, naming, or lint-aware cleanup, even if the user does not explicitly ask for "style." Use it across repositories, but defer to a project's existing conventions and tool configuration when they are clear. Do not use it for non-Python files.
---

# Python Style

Use this skill to keep Python changes consistent, readable, and practical
across codebases. Prefer local project consistency over copying an entire
style guide into every task.

## Workflow

1. Match the surrounding file first.
2. Keep changes small and task-focused.
3. Read only the reference section you need.
4. Do not introduce tooling or boilerplate the project does not use.

## Default Conventions

- Use 4 spaces, not tabs.
- Prefer implicit line joining inside parentheses over backslashes.
- Keep functions focused and names descriptive.
- Use `snake_case` for functions, variables, and modules.
- Use `CapWords` for classes and `UPPER_CASE` for constants.
- Add type annotations where they improve clarity, especially for public APIs
  and non-obvious data shapes.
- Use context managers for files and similar resources.
- Raise specific exceptions and avoid bare `except:`.
- Avoid mutable default arguments.
- Prefer f-strings for user-facing messages unless a logging API expects
  %-style formatting.
- Follow project lint and typing configuration when it exists.
- Do not add license headers, copyright notices, or other boilerplate.

## Read The Right Reference

Read the smallest reference that answers the current question.

- For lint behavior and suppression patterns, read `references/lint.md`.
- For import structure and typing-import exceptions, read
  `references/imports.md`.
- For module, class, and function docstrings, read
  `references/docstrings.md`.
- For annotations, `None`, generics, and `TYPE_CHECKING`, read
  `references/typing.md`.

If multiple sections apply, start with the one that affects correctness most:
typing, imports, docstrings, then lint.

## Review Posture

When reviewing Python code:

- Prioritize correctness, clarity, and maintainability over cosmetic nits.
- Surface style issues only when they materially improve the code.
- Suggest the smallest change that fixes the issue.

## Escalation Rules

- If the surrounding file already follows a clear local pattern, follow it
  unless it is causing bugs or confusion.
- If a reference conflicts with an explicit project convention or tool config,
  follow the project convention.
- If a lint or typing rule would force awkward code with no practical benefit,
  prefer the simpler implementation and explain the tradeoff briefly.