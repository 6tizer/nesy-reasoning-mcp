# Development Guide

This project uses Python, `uv`, pytest, Ruff, and the official MCP Python SDK.

## Local Setup

```bash
uv sync
```

Run the full local gate:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
printf '' | env PYTHONPATH=src uv run nesy-reasoning-mcp --transport stdio
```

## Branch Workflow

Use short-lived branches:

```bash
git switch -c feat/v0.4-persistence
```

Preferred branch prefixes:

- `feat/` for public behavior.
- `fix/` for bug fixes.
- `docs/` for docs only.
- `chore/` for CI, packaging, repository maintenance.

Open a pull request before merging to `main`. Required checks are:

- `Test Python 3.11`
- `Test Python 3.13`
- `Smoke checks`

## Version Scope Rules

Each version should have one primary goal.

- Do not add future tools before their planned version.
- Do not change existing public output shape without tests and docs.
- Keep `structuredContent` and `content[0].text` mirrored for every tool result.
- Keep business constants and tool names single-sourced in code.
- Keep stdout clean in stdio mode.

## Adding A Tool

Checklist:

- Add Pydantic input and output models.
- Add JSON schema metadata.
- Add tool registration.
- Add handler dispatch.
- Return `CallToolResult` with mirrored structured output.
- Add unit tests for valid input, invalid input, and MCP shape.
- Add smoke coverage if the tool is public.
- Update install docs and README tool list.

## Storage Work

Persistence changes need extra care:

- Memory store remains default unless version plan says otherwise.
- SQLite or file stores must preserve existing relation IDs.
- Load failures must not partially mutate store unless a mode explicitly allows it.
- File reads and writes must stay inside allowed roots.
- Tests must cover restart behavior and corrupt input.

## Hook Work

Hook integration must not depend on process memory. Use SQLite, JSON file, or HTTP
daemon state sharing.

Rules:

- Stop hooks must handle `stop_hook_active=true`.
- Hook outputs must be deterministic and small.
- Hook samples must not write secrets to repo.
- Any blocking hook must explain the contradiction and the affected context.

## Release Checklist

- Bump version in `pyproject.toml` and `src/nesy_reasoning_mcp/__init__.py`.
- Update README and docs for changed tools or config.
- Run full local gate.
- Confirm GitHub Actions passes.
- Build package with `rm -rf dist && uv build`.
- Attach artifacts only when cutting a GitHub release.
