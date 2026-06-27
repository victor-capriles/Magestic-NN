# Lint

Use lint as a safety net, not as the driver of the design.

## Default Approach

- If the project uses pylint, run it on touched Python files.
- If the project uses another linter as the established tool, follow that tool
  while keeping the same general principles.
- If the project provides a `.pylintrc` or a documented lint command, use it.
- Fix the underlying issue instead of suppressing the warning when practical.
- If a warning must be suppressed, make the suppression as narrow as possible.

## Suppressions

- Prefer `# pylint: disable=` with symbolic names.
- Use line-level disables first.
- Use block-level disables only when the whole block genuinely needs the same
  exception.
- Add a brief reason when the warning name alone is not self-explanatory.

Example:

```python
def do_PUT(self):  # WSGI name, so pylint: disable=invalid-name
  ...
```

## Unused Parameters

If a parameter must remain in the signature for interface compatibility,
delete it near the top of the function and explain why.

Example:

```python
def viking_cafe_order(
  spam: str,
  beans: str,
  eggs: str | None = None,
) -> str:
  del beans, eggs  # Unused by vikings.
  return spam + spam + spam
```

Avoid renaming parameters to `_` or `unused_...` when callers may pass them by
name.

## Useful Commands

```bash
pylint path/to/file.py
pylint --list-msgs
pylint --help-msg=invalid-name
```

## Decision Rule

If lint pushes the code toward a more complex design with no meaningful gain,
keep the simpler code and use the smallest justified suppression.