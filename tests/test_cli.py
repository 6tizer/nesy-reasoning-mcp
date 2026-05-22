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
