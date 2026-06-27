# Typing

Use type annotations to clarify interfaces and non-obvious data shapes.

## Priorities

- Annotate public APIs first.
- Annotate complex transformations and reusable helpers next.
- Do not force annotations onto trivial local variables unless they add
  clarity.
- Prefer clear, maintainable types over maximal cleverness.

## Core Rules

- Use built-in generic syntax such as `list[str]` and `dict[str, int]`.
- Use `X | None` for optional values.
- Prefer abstract container types from `collections.abc` in signatures.
- Use annotated assignments instead of old-style type comments.
- Use `Any` when a precise type would be misleading or noisy.

Example:

```python
from collections.abc import Sequence


def transform_coordinates(
    original: Sequence[tuple[float, float]],
) -> Sequence[tuple[float, float]]:
    ...
```

## Optional Values

Be explicit when `None` is allowed.

Example:

```python
def modern_or_union(a: str | int | None, b: str | None = None) -> str:
    ...
```

Avoid implicit optional patterns like `def f(x: str = None)`.

## Forward References

Use `from __future__ import annotations` when it simplifies forward references
or reduces quoting noise.

Example:

```python
from __future__ import annotations


class MyClass:
    def __init__(self, stack: Sequence[MyClass], item: OtherClass) -> None:
        ...


class OtherClass:
    ...
```

## Typing Imports

Import typing symbols directly when that improves readability.

Example:

```python
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias
```

Use `TYPE_CHECKING` only for imports that are truly needed for annotations and
should not run at runtime.

Example:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sketch


def f(x: "sketch.Sketch") -> None:
    ...
```

## Type Aliases And Generics

Use descriptive names for public aliases and private `_T` or `_P` names only
for private, unconstrained type variables.

Example:

```python
from typing import ParamSpec, TypeAlias, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

Coordinate: TypeAlias = tuple[float, float]
```

## Escape Hatches

- Use `# type: ignore` only when the checker is wrong or the interface is
  genuinely dynamic.
- Keep the ignore narrow and explain it if the reason is not obvious.
- Prefer fixing the type surface over stacking ignores.