# Imports

Keep imports explicit, stable, and easy to scan.

## Import Order

Place imports at the top of the file, after the module docstring.

Use this grouping order:

1. `from __future__` imports
2. Standard library imports
3. Third-party imports
4. Application-local imports

Within each group, sort lexicographically.

Example:

```python
from __future__ import annotations

import collections
import sys

from absl import flags
import tensorflow as tf

from myproject.backend import huxley
from myproject.backend.state_machine import main_loop
```

## What To Import

- Prefer importing modules or packages over importing individual functions.
- Use direct symbol imports from `typing`, `typing_extensions`, and
  `collections.abc` when that keeps annotations readable.
- Use standard abbreviations such as `import numpy as np` when they are widely
  recognized.
- Keep one import per line, except grouped symbol imports from typing-focused
  modules.

Example:

```python
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from sound.effects import echo

echo.EchoFilter(input_stream, output_stream, delay=0.7, atten=4)
```

## Local Imports

- Do not assume a package hierarchy the project does not have.
- When a project is packaged, prefer explicit absolute imports over ambiguous
  relative imports.
- Import inside a function only when there is a concrete reason, such as an
  optional dependency or a circular-import escape hatch.

Example:

```python
def create_plot(values: list[float]) -> None:
    import matplotlib.pyplot as plt  # Optional dependency for plotting.

    plt.plot(values)
    plt.show()
```

## Anti-Patterns

Avoid these forms:

```python
import os, sys
from module import helper_function
from . import helpers
```

Use them only when there is a strong local reason and the project already
follows that pattern.