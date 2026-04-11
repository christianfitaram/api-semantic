# Contributing

## Setup

```bash
poetry install --with dev
cp .env.example .env
```

## Development workflow

1. Create a feature branch from `main`.
2. Keep changes focused and include tests when behavior changes.
3. Run local checks before opening a PR:

```bash
poetry run ruff check .
poetry run pytest
```

## Pull requests

- Describe the motivation and behavior change.
- Link related issues.
- Include test coverage for new logic when practical.
- Keep pull requests small and reviewable.

## Code style

- Follow existing project patterns.
- Prefer explicit, readable code over premature abstraction.
