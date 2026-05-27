import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_guard_passes_clean_example() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "architecture_guard.py"),
            "--rules",
            str(ROOT / "examples" / "architecture-guard-rules.json"),
            "--facts",
            str(ROOT / "examples" / "architecture-guard-facts.json"),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["violations"] == []
    assert report["checked_rules"] == 3
    assert completed.stderr == ""


def test_architecture_guard_fails_when_gitnexus_fact_implies_violation(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(
        json.dumps(
            {
                "anchor": "ObservedArchitectureFacts",
                "relations": [
                    {
                        "source": "ObservedArchitectureFacts",
                        "target": "MCPServerCallsAgentSDK",
                        "relation_type": "sufficient",
                        "confidence": 1.0,
                        "provenance": {
                            "source": "gitnexus",
                            "command": "npx gitnexus context run_stdio_server",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "architecture_guard.py"),
            "--rules",
            str(ROOT / "examples" / "architecture-guard-rules.json"),
            "--facts",
            str(facts_path),
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
    )

    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["status"] == "fail"
    assert [item["id"] for item in report["violations"]] == ["no-agent-sdk-inside-mcp-server"]
    assert report["violations"][0]["best_path"]["nodes"] == [
        "ObservedArchitectureFacts",
        "MCPServerCallsAgentSDK",
    ]
    assert completed.stderr == ""


def test_architecture_guard_ignores_reverse_violation_path(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(
        json.dumps(
            {
                "anchor": "ObservedArchitectureFacts",
                "relations": [
                    {
                        "source": "MCPServerCallsAgentSDK",
                        "target": "ObservedArchitectureFacts",
                        "relation_type": "sufficient",
                        "confidence": 1.0,
                        "provenance": {
                            "source": "gitnexus",
                            "command": "npx gitnexus context run_stdio_server",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "architecture_guard.py"),
            "--rules",
            str(ROOT / "examples" / "architecture-guard-rules.json"),
            "--facts",
            str(facts_path),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["violations"] == []
    assert completed.stderr == ""


def test_architecture_guard_fails_when_facts_directly_imply_architecture_violation(
    tmp_path: Path,
) -> None:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(
        json.dumps(
            {
                "anchor": "ObservedArchitectureFacts",
                "relations": [
                    {
                        "source": "ObservedArchitectureFacts",
                        "target": "ArchitectureViolation",
                        "relation_type": "sufficient",
                        "confidence": 1.0,
                        "provenance": {"source": "gitnexus"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "architecture_guard.py"),
            "--rules",
            str(ROOT / "examples" / "architecture-guard-rules.json"),
            "--facts",
            str(facts_path),
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
    )

    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["status"] == "fail"
    assert [item["id"] for item in report["violations"]] == ["architecture-violation"]
    assert report["violations"][0]["best_path"]["nodes"] == [
        "ObservedArchitectureFacts",
        "ArchitectureViolation",
    ]
    assert completed.stderr == ""


def test_architecture_guard_rejects_rules_without_checks(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "anchor": "ObservedArchitectureFacts",
                "relations": [],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "architecture_guard.py"),
            "--rules",
            str(rules_path),
            "--facts",
            str(ROOT / "examples" / "architecture-guard-facts.json"),
        ],
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "checks must contain at least one object" in completed.stderr
