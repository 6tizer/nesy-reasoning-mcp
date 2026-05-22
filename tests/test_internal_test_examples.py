import json
import os
import subprocess
import sys
from pathlib import Path

from nesy_reasoning_mcp.config import NesyConfig, StorageBackend
from nesy_reasoning_mcp.schemas import RelationInput

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "internal-test"


def test_internal_test_json_examples_parse_and_config_loads() -> None:
    config_data = json.loads((EXAMPLES / "nesy-config.json").read_text(encoding="utf-8"))
    mcp_data = json.loads((EXAMPLES / "mcp-stdio-config.json").read_text(encoding="utf-8"))
    hooks_data = json.loads((EXAMPLES / "claude-hooks.json").read_text(encoding="utf-8"))

    config = NesyConfig.model_validate(config_data)

    assert config.storage.backend == StorageBackend.SQLITE
    assert config.storage.sqlite_path == "~/.nesy-reasoning/internal-test/nesy.db"
    assert config.logging.audit_log is True
    assert mcp_data["mcpServers"]["nesy-reasoning"]["env"]["NESY_CONFIG"].endswith(
        "examples/internal-test/nesy-config.json"
    )
    assert hooks_data["hooks"]["PreToolUse"][0]["hooks"][0]["command"].endswith(
        "hook-pretooluse.sh"
    )


def test_internal_test_wrappers_use_current_cli() -> None:
    for name, hook_name in {
        "hook-pretooluse.sh": "hook pretooluse",
        "hook-stop.sh": "hook stop",
        "run-http.sh": "--transport http",
    }.items():
        text = (EXAMPLES / name).read_text(encoding="utf-8")
        assert 'uv --directory "$REPO_DIR" run nesy-reasoning-mcp' in text
        assert hook_name in text
        assert 'PYTHONPATH="${PYTHONPATH:-$REPO_DIR/src}"' in text


def test_internal_test_wrapper_shell_syntax() -> None:
    for name in ["hook-pretooluse.sh", "hook-stop.sh", "run-http.sh"]:
        subprocess.run(["bash", "-n", str(EXAMPLES / name)], check=True)


def test_internal_test_report_template_mentions_agent_eval_and_mypy() -> None:
    text = (ROOT / "docs" / "internal-test-report-template.md").read_text(encoding="utf-8")

    assert "uv run mypy src/nesy_reasoning_mcp" in text
    assert "nesy-reasoning-mcp eval agent" in text
    assert "PostToolBatch hook is not part of v1.0 internal testing" in text


def test_agent_instructions_nesy_facts_example_is_relation_input() -> None:
    text = (EXAMPLES / "agent-instructions.md").read_text(encoding="utf-8")
    start = "<!-- NESY_FACTS_EXAMPLE_START -->"
    end = "<!-- NESY_FACTS_EXAMPLE_END -->"
    block = text.split(start, 1)[1].split(end, 1)[0]
    json_text = block.split("```json", 1)[1].split("```", 1)[0]
    facts = json.loads(json_text)

    parsed = [RelationInput.model_validate(item) for item in facts]

    assert parsed[0].source == "降价"
    assert parsed[0].target == "销量增加"


def test_internal_test_smoke_script_runs() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "smoke.py")],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout.strip() == "internal-test smoke ok"
