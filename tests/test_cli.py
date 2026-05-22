import os
import subprocess
import sys
from pathlib import Path


def test_help_writes_no_stdout_banner() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "--help"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "usage: nesy-reasoning-mcp" in completed.stdout
    assert "http" in completed.stdout
    assert "eval" in completed.stdout
    assert completed.stderr == ""


def test_eval_help_lists_llm_subcommand() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "eval", "--help"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "run" in completed.stdout
    assert "llm" in completed.stdout
    assert completed.stderr == ""
