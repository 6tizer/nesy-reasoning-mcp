import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from nesy_reasoning_mcp.evaluation import BenchmarkFixture, run_eval_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "benchmarks" / "fixtures" / "core.json"
SCHEMA_PATH = ROOT / "benchmarks" / "fixtures" / "core.schema.json"


def test_benchmark_fixture_matches_json_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(fixture)
    parsed = BenchmarkFixture.model_validate(fixture)

    assert parsed.name == "core-phase7-offline"
    assert len(parsed.cases) >= 8


def test_benchmark_schema_rejects_invalid_category() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["cases"][0]["category"] = "unknown"

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(fixture)


def test_benchmark_schema_rejects_empty_expected() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["cases"][0]["expected"] = {}

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(fixture)


def test_benchmark_model_rejects_invalid_baseline_score() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["cases"][0]["baselines"]["llm_only"] = 2

    with pytest.raises(ValueError):
        BenchmarkFixture.model_validate(fixture)


def test_benchmark_model_rejects_empty_expected() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["cases"][0]["expected"] = {}

    with pytest.raises(ValueError):
        BenchmarkFixture.model_validate(fixture)


@pytest.mark.asyncio
async def test_eval_runner_scores_core_fixture() -> None:
    report = await run_eval_file(FIXTURE_PATH)

    assert report["status"] == "pass"
    assert report["case_count"] == 9
    assert report["passed"] == 9
    assert report["full_mcp_score"] == 1.0
    assert report["metrics"]["logical_accuracy"] == 1.0
    assert report["metrics"]["contradiction_recall"] == 1.0
    assert report["metrics"]["false_contradiction_rate"] == 0.0
    assert report["metrics"]["counterfactual_conservatism"] == 1.0
    assert report["marginal_contribution"]["no_classify"] > 0
    assert report["marginal_contribution"]["no_contradiction"] > 0
    assert report["marginal_contribution"]["no_counterfactual"] == 1.0
    assert report["marginal_contribution"]["no_verify_chain"] == 1.0


@pytest.mark.asyncio
async def test_eval_runner_min_score_failure() -> None:
    report = await run_eval_file(FIXTURE_PATH, min_score=1.01)

    assert report["status"] == "fail"
    assert report["failed"] == []


def test_eval_cli_outputs_parseable_json() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "eval",
            "run",
            "--fixture",
            str(FIXTURE_PATH),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["case_count"] == 9
    assert completed.stderr == ""
