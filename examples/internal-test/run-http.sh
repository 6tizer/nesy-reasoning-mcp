#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${NESY_REPO_DIR:-/Users/mac-mini/Documents/nesy-reasoning-mcp}"
export NESY_CONFIG="${NESY_CONFIG:-$REPO_DIR/examples/internal-test/nesy-config.json}"
export NESY_LOCAL_TOKEN="${NESY_LOCAL_TOKEN:-nesy-internal-test-token}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR/src}"

exec uv --directory "$REPO_DIR" run nesy-reasoning-mcp --transport http
