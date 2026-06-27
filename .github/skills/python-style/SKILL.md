---
name: python-style
description: "Use when writing, refactoring, or reviewing Python code. Provides a shared Python baseline for services, APIs, automations, integrations, libraries, and tests. Use this skill for Python work even when the requested change is small. This skill focuses on portable rules such as typing, imports, logging, testing, documentation, and code formatting."
---

# Python Standards

## Goal

Keep Python code consistent, safe, and understandable — for humans and AI agents alike.
Standardize new code. Normalize existing code only if the user explicitly agrees cleanup is in scope.
Do the task at hand first. If you notice adjacent improvements, explain the scope and get approval before refactoring them.
Never do large style-only rewrites in files you are not already modifying.

## Repository-Specific References

This skill is the general baseline.
---

## 1. Design Principles

| Principle | When to apply |
| --- | --- |
| **Single Responsibility** | Every function, class, and module does one thing. Splitting is driven by SRP, not line count. |
| **Open/Closed** | Use when extension points are realistic. Don't build registries for hypothetical future needs. |
| **Liskov Substitution** | Subclasses must honor the parent contract. |
| **Interface Segregation** | Compose focused models — don't force consumers to depend on fields they don't use. |
| **Dependency Inversion** | Depend on abstractions (`StorageProvider`, `LLMClient`), not concretions. |
| **Functional core / Imperative shell** | Pure transforms in helpers (easy to test). I/O, network, storage, streaming in the shell. |
| **Complexity heuristic** | Before adding any pattern: does it make the code easier to understand, test, or extend in likely ways? If all three are no, skip it. |

Example:

```python
def normalize_email(email: str) -> str:
    return email.strip().lower()
```

Do not introduce a `NormalizerFactory`, strategy object, or registry for a one-line transform unless the code already has real variants to select between.

---

## 2. Naming Conventions

| What | Convention | Example |
| --- | --- | --- |
| Module / file | `snake_case` | `storage_client.py` |
| Class | `PascalCase` | `ProcessingState` |
| Exception class | `PascalCase` + `Error` suffix | `InvalidInputError` |
| Function / method | Verb-first `snake_case` | `build_context()`, `validate_manifest()` |
| Variable / attribute | `snake_case` | `user_details` |
| Constant | `SCREAMING_SNAKE_CASE`, module-level | `MAX_RETRIES` |
| Boolean | `is_` / `has_` / `can_` / `should_` prefix | `is_retryable`, `has_context` |
| Date / time | `_at` / `_date` / `_time` suffix | `created_at`, `end_date_str` |
| Exception variable | Always `e` | `except ValueError as e:` |
| Test function | `test_<scenario>_<expected>` | `test_empty_history_returns_default()` |
| Private / internal | Leading underscore | `_parse_row()` |
| Factory function | `get_<name>_<thing>()` when factory semantics matter | `get_storage_client()` |
| Domain abbreviations | Use only established, documented abbreviations | `llm`, `api`, `msg` |

## 3. Typing

All new code should use modern Python typing consistently.

| Rule | Convention | Example |
| --- | --- | --- |
| Generics | Built-in only | `list[str]`, `dict[str, Any]`, `tuple[int, ...]` |
| Nullable | Union syntax | `str \| None` — not `Optional[str]` in new code |
| `Any` | Minimize and justify | `payload: Any  # untyped external SDK response` |
| Parameters (read-only) | Prefer abstract input types | `items: Sequence[str]`, `config: Mapping[str, Any]` |
| Return types | Use concrete return types and annotate every function | `-> list[str]`, `-> dict[str, Any]` |
| Closed domain set | Prefer `str, Enum` | `class ApplicationMode(str, Enum)` |
| Inline constraint | Use `Literal` for narrow choices | `role: Literal["system", "user"]` |
| Complex alias | Use `TypeAlias` | `JsonDict: TypeAlias = dict[str, Any]` |
| `collections.abc` source | Import abstract containers from `collections.abc` | `from collections.abc import Sequence` |
| Pydantic defaults | When using Pydantic, default new models to `ConfigDict(extra="forbid", frozen=True)` unless there is a real reason not to | See repository overrides when present |
| Type-only imports | Use `if TYPE_CHECKING:` only for type-only imports or unavoidable runtime cycles | `if TYPE_CHECKING: from my_project.storage.backends import DynamoStorage` |
| Quoted forward references | If a type is imported only under `TYPE_CHECKING`, annotate it as a string | `storage: "DynamoStorage"` |
| No future annotations | Do not rely on postponed annotation evaluation via `from __future__ import annotations` unless a repository explicitly requires it | Match local repo compatibility rules |
| Placement | Keep `TYPE_CHECKING` blocks immediately after the normal import section | Makes type-only dependencies visible and predictable |
| Scope | Put only type-only imports and typing aliases inside `TYPE_CHECKING` blocks | Avoid mixing runtime logic into the block |
| Circular imports | Prefer extracting shared types first; otherwise use `TYPE_CHECKING` plus quoted annotations to break the runtime cycle safely | Preserve type safety without triggering import-time failures |

Quoted forward references are not just style here. They are the safe way to keep runtime imports from resolving names too early when a type-only import would otherwise create a cycle.

## 4. Truthiness & None Checks

| Rule | Convention | Example |
| --- | --- | --- |
| Collections / strings | Prefer implicit truthiness | `if items:`, `if not items:` — not `if len(items) > 0` |
| `None` checks | Always use identity checks | `if value is None:` / `if value is not None:` |
| Missing vs empty | Do not collapse them unless intentional | Avoid `value = value or []` when `None` and `[]` mean different things |
| Integers | Compare explicitly when zero is meaningful | `if retry_count == 0:` not `if not retry_count:` |

---

## 5. Expression Simplicity

| Rule | Convention | Example |
| --- | --- | --- |
| Comprehensions | Keep them simple and optimize for readability | Prefer `[item.id for item in items if item.active]` over multi-`for` comprehensions |
| Multi-`for` comprehensions | Avoid them | Use a regular loop when the expression needs more than one `for` clause |
| Generator expressions | Prefer over `map()` / `filter()` with `lambda` for simple cases | `sum(item.value for item in items)` |
| Lambdas | Use only for short, obvious one-liners | If logic grows, extract a named helper |
| Conditional expressions | Use only for short, one-line cases | `status = "ok" if is_ready else "pending"` |
| Generator docstrings | Use `Yields:` instead of `Returns:` | Document yielded items, not the generator object |

---

## 6. Nested Helpers, Scope, and Decorators

| Rule | Convention | Example |
| --- | --- | --- |
| Nested helpers | Use nested functions or classes when they genuinely close over local state or keep a helper tightly scoped | Move reusable logic back to module level |
| Lexical scope | Keep data flow easy to follow; avoid surprising outer-scope mutation or rebinding | Prefer explicit parameters in callbacks and loops |
| Decorators | Use decorators when they remove repetition or enforce cross-cutting behavior clearly | Test decorators directly |
| Decorator docstrings | A decorator's docstring should say that it is a decorator | Makes call semantics clear |
| `@classmethod` | Prefer for named constructors or explicit lifecycle routines | Avoid using it as the default for ordinary helpers |
| `@staticmethod` | Prefer a module-level helper when no instance or class state is needed | Improves reuse and testing |

## 7. Iteration & Membership Idioms

| Rule | Convention | Example |
| --- | --- | --- |
| Default iteration | Iterate directly over containers that support it | `for key in mapping:`, `for line in file:` |
| Dict keys | Do not loop over `.keys()` when you only need keys | `for key in mapping:` — not `for key in mapping.keys():` |
| Dict key/value pairs | Use `.items()` when you need both key and value | `for key, value in mapping.items():` |
| Membership tests | Use `in` / `not in` directly on the container | `if key in mapping:` — not `if key in mapping.keys():` |
| Indexed iteration | Use `enumerate()` when you need both position and value | `for idx, item in enumerate(items):` — not `for idx in range(len(items)):` |
| Parallel iteration | Use `zip()` when walking aligned iterables together | `for left, right in zip(names, values):` |
| File iteration | Iterate over the file object directly | `for line in file:` — not `for line in file.readlines():` |
| Mutation caveat | Do not mutate a container while iterating over it | Iterate over a snapshot if mutation is required |

This follows Google’s default iterators and operators guidance: prefer the container’s native iteration and membership behavior because it is simpler, more generic, and usually more efficient.

---

## 8. Suppression Policy

| Rule | Convention | Example |
| --- | --- | --- |
| Fix first | Prefer a real fix before suppression: refine types, add annotations, use `cast()`, extract a helper, or adjust configuration | Do not suppress an error you can model directly |
| Smallest scope | Suppress at the narrowest possible scope | Prefer a line-level suppression over a file-level disable |
| Specific mypy ignores | Use error-code-qualified ignores in new code | `# type: ignore[no-any-return]` — not bare `# type: ignore` |
| Specific Ruff suppressions | Use explicit Ruff rule codes | `# noqa: F401` — not bare `# noqa` |
| Reason comment | Add a short reason when the suppression is not self-evident | `# noqa: F401 - re-exported for package API` |
| Prefer typed escape hatches | Use `cast()` or a typed adapter when you know the shape but the checker cannot infer it | Prefer that over ignoring the entire line |
| File-wide disables | Avoid inline file-wide disables like `# mypy: ignore-errors` or `# ruff: noqa` in source files | Use config-level exceptions only for generated or structurally unsupported code |
| Config-level exceptions | Put systematic tool limitations in `pyproject.toml` with explicit scope and rationale | Existing module-level mypy override for `override` mismatches |
| Broad suppressions | Do not add broad or bare suppressions in new code | Prefer `# type: ignore[attr-defined]` over bare `# type: ignore` when the cause is known |

Suppressions are debt markers, not normal style tools. In new code they must be specific, minimal, and justified so they do not hide unrelated problems.

---

## 9. Function Design

| Rule | Convention |
| --- | --- |
| Single Responsibility | Each function does exactly one thing |
| Line target | 40–50 lines. Exceeding, you must justify or break up |
| Parameter limit | ≤ 5 (excluding `self` / `ctx` / `logger`). Group related params into a model or dataclass |
| Nesting limit | 3 target, 4 max. Flatten with guard clauses, `continue`, or extract to helper |
| Guard clauses | Validate early, return / raise fast |
| Mutable defaults | Never use `[]` or `{}` — use `values: list[str] \| None = None` |
| Immutability (hard rule) | Never mutate function inputs — operate on copies. Retries re-use the same arguments, so a mutated input silently corrupts every subsequent attempt |
| Immutability (preference) | Prefer creating new objects over mutating local state |
| Functional core | Extracted helpers are pure — no I/O, no side effects |
| Imperative shell | Keep I/O and side effects in thin coordinating layers |

For larger orchestration methods, prefer a shape where the main method reads like a table of contents:

```python
async def run(self, context: RequestContext) -> ProcessingResult:
    request = self._validate(context)
    payload = self._build_payload(request)
    dependencies = await self._load_dependencies(payload)
    response = await self._execute(payload, dependencies)
    return self._build_result(response)
```

Each lifecycle helper ≤ 50 lines, single responsibility, independently testable.

Example:

```python
def build_retry_summary(attempts: Sequence[int]) -> str:
    if not attempts:
        return "no retries"
    latest_attempt = attempts[-1]
    return f"retried {latest_attempt} times"
```

---

## 10. Error Handling

| Rule | Convention | Example |
| --- | --- | --- |
| Specific catches | Catch the narrowest exception type | `except json.JSONDecodeError as e:` |
| Broad catch | Only if re-raising: `raise` or `raise ... from e` | `except Exception as e: raise NodeError(...) from e` |
| Exception chaining | Always `from e` when wrapping | `raise InvalidResultError("parse failed") from e` |
| Bare re-raise | `raise` — never `raise e`. `raise e` creates a new traceback starting at the except block, losing the original crash site | `except TooManyRequestsError: raise` |
| Try block scope | Minimal — wrap only the line(s) that can throw | Wrap only the code that is expected to fail |
| Custom exceptions | `Error` suffix. Prefer built-in types when they fit semantically | `ValueError` for bad input, custom for domain concepts |
| New exceptions | Follow the repository's error-module convention if it exists, if not create a new module | Keep error ownership local to the boundary that knows the context |
| No `assert` for validation | `raise ValueError(...)` for runtime checks | `assert` only in tests / debug invariants |
| Cleanup | `finally` or `with` statement | `with open(path) as f:` — never rely on `except` alone |
| Silent suppression | Only for non-critical telemetry + comment | `except Exception: pass  # telemetry failure must not break workflow` |
| Justify broad catch | Comment required | `except Exception as e:  # external SDK type unknown` |
| Boundary ownership | Wrap external contract or data-shape failures once, at the boundary that has the missing domain context. A raw `KeyError: 'impressions'` is meaningless for debugging; `"PrintBeat Jobs API unexpected response: 'impressions'"` identifies the tool, domain, and failure type immediately | Make messages precise, grep-friendly, and honest about what actually failed, Do not claim a root cause you have not established |

---

## 11. Logging

| Rule | Convention | Example |
| --- | --- | --- |
| String format | `%`-formatting always, never f-strings. Lazy evaluation means the string is only formatted if the level is enabled; consistency eliminates judgment calls about whether an expression is "expensive enough" to worry about | `logger.info("Processing %s items", count)` |
| Logger naming | `logging.getLogger(__name__)` at module level | `logger = logging.getLogger(__name__)` |
| No dynamic loggers | Never `setup_logger(f"name_{id}")` — memory leak | Pass ID in the message |
| No root logger | Never `logging.getLogger().error(...)` | Use module-level `logger` |
| No `print()` | Never for diagnostics in production code | `logger.debug(...)` instead |
| No module-level funcs | Never `logging.info()` directly | `logger.info(...)` via module-level instance |
| DEBUG | Internal state, flow tracing | `logger.debug("Raw token count: %s", count)` |
| INFO | Milestones, business events | `logger.info("Intent classified as %s", intent)` |
| WARNING | Unexpected but handled | `logger.warning("Primary API timed out, using cache")` |
| ERROR | Operation broken — use `logger.exception()` | `logger.exception("Failed to classify intent")` — auto stack trace |
| CRITICAL | System at risk | `logger.critical("Cannot connect to DB after %s retries", n)` |
| Exception: real failure | `logger.exception()` in `except` block | Preserves full stack trace |
| Exception: trace at non-ERROR | `exc_info=True` kwarg when you need a stack trace at WARNING or DEBUG level | `logger.warning("Retrying %s", svc, exc_info=True)` |
| Exception: handled | `logger.warning()` or `logger.debug()` | No stack trace needed |
| Never `logger.error(str(e))` | Converts the exception to a flat string, losing the stack trace entirely — use `logger.exception()` instead | 90+ legacy instances exist; fix via Boy Scout Rule |
| Never log | Secrets, tokens, passwords, PII | Redact using length + first 8 + last 4 + hash pattern |
| Payload dumps | DEBUG only, verify no secrets | `logger.debug("Response: %s", sanitized_payload)` |
| Log decisions, not data | Outcomes and strategies, not raw payloads | `"Tool %s returned %s results"` — not the full payload |
| Log at boundaries | Entry / exit of significant operations | Node start, API call, external service response |
| No loop logging | Summary after loop, not per-iteration | `logger.info("Processed %s items", count)` |
| Domain context | Include IDs in business operations | `logger.info("Tool %s for conversation %s", tool, conv_id)` |

---

## 12. Documentation

Every module, class, function, method, property, and test must have a docstring. No exceptions. AI agents are first-class consumers, so explicit docstrings help both humans and tooling.

| Rule | Convention |
| --- | --- |
| Module docstring | Mandatory, 1-5 lines describing the module's purpose and any important usage constraints |
| Function / method | Mandatory. Include enough context to use the callable without reading its implementation. One-liner for simple functions. `Args:` + `Returns:` + `Raises:` for complex functions |
| Property docstring | Write it like an attribute description |
| Test docstring | Mandatory, explain the scenario and the expected behavior |
| Overridden methods | Prefer a docstring; `"""See base class."""` is acceptable only when the contract is unchanged |
| Class docstring | Describe what an instance represents and document public attributes when they are part of the contract |
| Style | Use Google-style sections such as `Args:`, `Returns:`, `Yields:`, `Raises:`, and `Attributes:` when they add value |
| Block comments | Explain the approach, constraint, or tradeoff, not Python syntax |
| Inline comments | Use them only for short, non-obvious clarifications, with two spaces before `#` |
| TODO format | Use `# TODO(username): description` and include a ticket when relevant |
| Spelling / grammar | Write docstrings and comments as real prose with correct spelling, capitalization, and punctuation |

**Comment quality**
- Correct: `# Execute tools in parallel so the LLM sees all results next turn.`
- Incorrect: `# Loop through tools and append to messages.`

Example:

```python
def build_cache_key(user_id: str, region: str) -> str:
    """Build a stable cache key for a user in a specific region.

    Args:
        user_id: The stable user identifier.
        region: The deployment region code.

    Returns:
        The cache key used by downstream storage.
    """
    return f"{region}:{user_id}"
```

## 14. Code Formatting

| Rule | Convention |
| --- | --- |
| Statements | Use one statement per line and avoid semicolons |
| Line length | Follow the repository's configured maximum; otherwise keep lines comfortably short |
| Parentheses | Use them for grouping, clarity, or implicit line continuation, not redundancy |
| Indentation | Use 4 spaces per level and never use tabs |
| Trailing commas | Use them in multi-line collections, argument lists, and imports when they help diffs and formatters |
| Blank lines | Use two blank lines between top-level definitions and one between methods |
| Whitespace | Follow normal Python spacing; do not vertically align tokens across lines |
| Shebangs | Add a shebang only to directly executable scripts |

## 15. File Layout

| Rule | Convention |
| --- | --- |
| Import groups | stdlib → third-party → local. Blank line between. Alphabetical within |
| Full package path | Always absolute — `from my_project.api.schemas import Foo`, never relative imports |
| No wildcard imports | Never `from X import *` |
| `import module` | For stdlib used as namespace: `import json`, `import logging`, `import os` |
| `from X import Y` | For specific symbols. Parenthesized multi-line for multiple |
| Module structure | Docstring → imports → constants → type aliases → helpers → classes → factories |
| `TYPE_CHECKING` block | Place it immediately after normal imports |
| File size | Treat ~400 lines as a soft limit; split only when it improves responsibility boundaries |
| Constants | Keep constants module-level, named `UPPER_CASE`, and avoid magic strings |
| One class per file | Prefer one major class per file for services or adapters; group related schema types when that is clearer |
| `__all__` | Not required for internal application code |

---

## 16. Main Guard & Import-Time Side Effects

| Rule | Convention | Example |
| --- | --- | --- |
| Executable logic | Put utility script entrypoints in `main()` and call them only under the main guard | `if __name__ == "__main__": main()` |
| Import safety | Keep modules safe to import by tests, scripts, and docs tooling | Avoid top-level API calls, background threads, or client initialization |
| Top-level code | Limit it to imports, definitions, constants, cheap configuration, and logger setup | Importing a module must not start real work |
| Startup separation | Keep CLI glue separate from reusable library logic | Parse args and read env in the startup layer |
| Expensive setup | Defer network clients, model loading, and cache warmup to explicit startup paths | Not during module import |

---

## 17. Properties & Accessors

| Rule | Convention | Example |
| --- | --- | --- |
| Property use | Use `@property` only for cheap, local, unsurprising behavior | A derived label or count based on already-loaded state |
| No hidden I/O | Do not hide network calls, DB lookups, initialization, or cache warmup behind property access | Use explicit methods instead |
| Stored vs derived state | Keep stored state in fields and use properties only for cheap derived views | Applies to models, service objects, and context carriers |
| No trivial wrappers | Do not add properties that only expose a private field unchanged | Make the field directly accessible instead |
| Setter side effects | If assignment invalidates caches or triggers work, use an explicit method | `reconfigure_model()` is clearer than a side-effecting setter |
| Descriptor complexity | Prefer `@property` over custom descriptors for normal cases | Keep class behavior inspectable to humans and AI tools |
| Inheritance | Prefer methods over properties for behavior that subclasses may need to extend | Keeps extension points explicit |
| Property docstrings | Properties still need docstrings, and they should read like attributes | `"""The current retry delay."""` |

## 18. Global State & Shared Services

| Rule | Convention |
| --- | --- |
| Default state flow | Pass mutable state through constructor injection, request context, dependency containers, or explicit service objects |
| Constants vs state | Module-level constants are fine; writable module-level caches, registries, and clients are not the default |
| Shared-state exception | Shared mutable state is allowed only with a documented owner, narrow access surface, and explicit lifecycle or reset behavior |
| Internal access | Keep mutable shared state internal and expose it through methods or helper functions, not arbitrary external writes |
| Constructor injection | Services receive dependencies via the constructor rather than creating them internally |
| Service lifecycle | If shared service state is necessary, make setup idempotent and shutdown explicit |
| DI abstractions | Depend on abstractions, not concrete implementations |
| Power features | Avoid metaclasses, import hacks, dynamic inheritance, `exec`, `eval`, `__del__`, and broad runtime patching unless an integration truly forces them |
| Prefer ordinary abstractions | Reach for functions, classes, context managers, dataclasses, enums, and adapters before clever runtime machinery |

Example:

```python
class ReportService:
    def __init__(self, storage: ReportStorage) -> None:
        self._storage = storage
```

Prefer constructor injection like the example above over a module-global client such as `_storage = S3ReportStorage()` that is created at import time and shared implicitly.

## 19. Models & Schemas

| Rule | Convention | Example |
| --- | --- | --- |
| Default config | When using Pydantic, default new models to `ConfigDict(extra="forbid", frozen=True)` | Override only with a documented reason |
| Explicit `extra` | Every model declares its `extra` behavior explicitly | Comment when using `"ignore"` or `"allow"` |
| `frozen=False` | Add a short reason comment | `# frozen=False: accumulator updated across steps` |
| Validators at boundaries | Use `Field()` for simple constraints and validators for cross-field rules | `Field(ge=0, max_length=300)` |
| Naming: inputs | Use `*Request` when it clarifies API or service input boundaries | `SearchRequest` |
| Naming: outputs | Use `*Response`, `*Result`, or domain names as appropriate | `SearchResponse`, `ProcessingResult` |
| Naming: config | Use `*Config` or `*Settings` | `StorageSettings` |
| Pydantic vs dataclass | Use Pydantic at validation/serialization boundaries and dataclasses for simple internal containers | Match the repo's established pattern |
| Composition | Prefer focused sub-models over large flat god models | Compose context out of smaller types |
| No monkey-patching | Do not dynamically add attributes to models | Use constructor fields or subclasses |
| Immutable updates | Use `model_copy(update={...})` for immutable updates | Avoid in-place mutation after creation |

Example:

```python
class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    limit: int = Field(default=10, ge=1, le=100)
```

## 20. Testing

| Rule | Convention |
| --- | --- |
| Organization | Prefer class-based grouping with `Test<UnitUnderTest>` when it adds structure |
| Method naming | Use `test_<scenario>_<expected>` |
| No `__init__` | Never define `__init__` in test classes |
| Shared setup | Prefer fixtures over `setUp` / `tearDown` |
| Fakes and patches | Prefer concrete fakes or `monkeypatch` for simple replacement and `Mock(spec=...)` when call assertions matter |
| Async doubles | Use `AsyncMock` for async collaborators |
| Async tests | If the project already configures automatic asyncio support, write async tests as `async def` without adding `@pytest.mark.asyncio` |
| Error assertions | Use `pytest.raises(..., match=...)` |
| AAA structure | Use `# Arrange`, `# Act`, and `# Assert` in longer tests |
| Factories | Put shared factories in `conftest.py` as `create_*()` helpers and local factories in the test module as `_make_*()` helpers |

Coverage policy:
- Always test business logic, validators, retry logic, decorators, data transformations, and error paths.
- Thin wrappers and trivial config modules may not need dedicated tests.
- New logic in a pull request should ship with tests unless there is a real emergency.

Example:

```python
def test_build_retry_summary_returns_default_for_empty_attempts() -> None:
    """Return the default message when no retries occurred."""
    # Arrange
    attempts: list[int] = []

    # Act
    result = build_retry_summary(attempts)

    # Assert
    assert result == "no retries"
```

## 21. Async Patterns

| Rule | Convention |
| --- | --- |
| Async handlers | Use `async def` for handlers and endpoints in async frameworks |
| No blocking in async code | Wrap sync I/O with `asyncio.to_thread()` or another explicit boundary |
| Shared mutable state | Do not rely on accidental atomicity across threads or tasks; use explicit ownership or synchronization |
| Task ownership | Prefer explicit task creation, cancellation, and coordination when lifecycle matters |
| Streaming cleanup | If a generator or stream owns expensive resources, define cleanup and cancellation boundaries explicitly |

Example:

```python
async def load_settings(path: Path) -> dict[str, Any]:
    return await asyncio.to_thread(read_settings_file, path)
```

## 22. API Boundaries

| Rule | Convention |
| --- | --- |
| Dependency injection | In frameworks that support DI, use the framework mechanism instead of repeated manual extraction and validation |
| Contracts | Keep request and response contracts explicit and typed |
| Error responses | Use the repository's structured error response contract for API failures |
| Validation tiers | Use schema validation for shape errors and explicit business rules for domain constraints |
| Event/message types | Use constants or enums instead of repeated inline string literals |
| Middleware | Add middleware only when required by the actual deployment and security architecture |

Example:

```python
@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, service: SearchService = Depends(get_search_service)) -> SearchResponse:
    if request.limit > 50:
        raise HTTPException(status_code=400, detail={"code": "LIMIT_TOO_HIGH"})
    return await service.search(request)
```

Keep the request and response models explicit, and use the repository's structured error payload shape when returning API failures.

## 23. Final Check

Before finishing Python changes, confirm:

1. Typing is modern in the code you touched.
2. Pure logic is extracted from side effects where that improves clarity.
3. Logging is safe (`%`-format, no secrets, right level).
4. Error handling is specific and not duplicated unnecessarily.
5. Modules, classes, functions, methods, properties, and tests have docstrings.
6. New logic has focused test coverage.
7. Imports are ordered, absolute, alphabetically, and consistent with repository tooling.
8. Modules stay safe to import and keep startup logic behind explicit entrypoints.
9. Properties stay cheap and local; I/O and lifecycle work stay explicit.
10. Shared mutable state has an explicit owner and lifecycle.
11. When the repository uses Pydantic, new models declare `extra` explicitly and justify mutable models.
12. If the repository cannot be inferred from context, confirm it before loading a repository-specific architecture reference.