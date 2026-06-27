# Docstrings

Document behavior that a reader cannot safely infer from the signature alone.

## General Rules

- Use triple double quotes.
- Keep the summary line concise and end it with punctuation.
- Describe behavior and semantics, not line-by-line implementation.
- Stay consistent within a file: imperative or descriptive style is fine, but
  do not mix them casually.

## Module Docstrings

Use a module docstring for non-trivial modules and scripts.

- Explain what the module does.
- Mention important inputs, outputs, or side effects.
- Skip boilerplate that adds no information.

Example:

```python
"""A one-line summary of the module or program.

Leave one blank line. The rest of this docstring should contain an
overall description of the module or program.
"""
```

## Function And Method Docstrings

Add a docstring when a function is public, non-trivial, or contains logic that
is not obvious from the name and signature.

Use these sections when they help:

- `Args:` for parameters that need explanation
- `Returns:` or `Yields:` for non-obvious output semantics
- `Raises:` for interface-relevant exceptions

Example:

```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table row to
            fetch.
        require_all_keys: If True, only rows with values set for all keys are
            returned.

    Returns:
        A dict mapping keys to the corresponding table row data.

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

If a private helper is short and obvious, a docstring is optional.

## Class Docstrings

Public classes should describe what the instance represents. Document public
attributes when they are part of the interface.

Example:

```python
class SampleClass:
    """Summary of class here.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
```

## Overrides

If an overridden method keeps the base behavior contract and the override is
explicitly marked, a new docstring is optional. Add one when the override
changes behavior, adds side effects, or needs extra explanation.

Example:

```python
from typing_extensions import override


class Parent:
    def do_something(self):
        """Parent method, includes docstring."""


class Child(Parent):
    @override
    def do_something(self):
        pass
```