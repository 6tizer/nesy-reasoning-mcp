#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${NESY_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export NESY_CONFIG="${NESY_CONFIG:-$REPO_DIR/examples/internal-test/nesy-config.json}"
export NESY_LOCAL_TOKEN="${NESY_LOCAL_TOKEN:-nesy-internal-test-token}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR/src}"

exec uv --directory "$REPO_DIR" run nesy-reasoning-mcp --transport http
