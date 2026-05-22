"""Offline benchmark evaluation for NeSy Reasoning."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nesy_reasoning_mcp.schemas import RelationSetData
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import LOAD_RELATIONS, call_tool

EvalCategory = Literal[
    "classification",
    "transitive",
    "contradiction",
    "counterfactual",
    "business",
]
EvalProvider = Literal["openai"]
OPENAI_LLM_ONLY = "openai_llm_only"


class ExpectedSpec(BaseModel):
    """Expected result matcher for one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    equals: dict[str, Any] = Field(default_factory=dict)
    contains: dict[str, Any] = Field(default_factory=dict)
    min_count: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_matcher(self) -> ExpectedSpec:
        """Require each benchmark case to assert at least one observable result."""
        if not self.equals and not self.contains and not self.min_count:
            raise ValueError("expected must define at least one matcher")
        return self


class BenchmarkCase(BaseModel):
    """One offline benchmark case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: EvalCategory
    description: str
    relation_set: RelationSetData = Field(default_factory=RelationSetData)
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    expected: ExpectedSpec
    baselines: dict[str, float] = Field(default_factory=dict)

    @field_validator("baselines")
    @classmethod
    def validate_baseline_scores(cls, value: dict[str, float]) -> dict[str, float]:
        """Ensure static baseline scores are normalized."""
        for name, score in value.items():
            if score < 0 or score > 1:
                raise ValueError(f"baseline score for {name} must be between 0 and 1")
        return value


class BenchmarkFixture(BaseModel):
    """A versioned benchmark fixture."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    name: str
    cases: list[BenchmarkCase] = Field(min_length=1)


def run_eval_cli(args: Any) -> int:
    """Run evaluation from argparse args and write the selected output."""
    report = anyio.run(
        run_eval_file,
        Path(args.fixture),
        args.min_score,
    )
    status_ok = report["status"] == "pass"
    text = _json_report(report) if args.format == "json" else _text_report(report)
    if args.output:
        Path(args.output).write_text(text + ("\n" if text else ""), encoding="utf-8")
    else:
        print(text)
    return 0 if status_ok else 1


def run_llm_eval_cli(args: Any) -> int:
    """Run live LLM baseline evaluation from argparse args."""
    try:
        report = anyio.run(
            run_llm_eval_file,
            Path(args.fixture),
            args.provider,
            args.model,
            args.case_id,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    status_ok = report["status"] == "pass"
    text = _json_report(report) if args.format == "json" else _llm_text_report(report)
    if args.output:
        Path(args.output).write_text(text + ("\n" if text else ""), encoding="utf-8")
    else:
        print(text)
    return 0 if status_ok else 1


async def run_eval_file(path: Path, min_score: float = 1.0) -> dict[str, Any]:
    """Evaluate a fixture file and return a structured report."""
    fixture = BenchmarkFixture.model_validate_json(path.read_text(encoding="utf-8"))
    return await run_fixture(fixture, fixture_path=str(path), min_score=min_score)


async def run_llm_eval_file(
    path: Path,
    provider: EvalProvider = "openai",
    model: str = "gpt-5.2",
    case_ids: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a live LLM-only baseline against a benchmark fixture."""
    fixture = BenchmarkFixture.model_validate_json(path.read_text(encoding="utf-8"))
    filtered = _filter_fixture_cases(fixture, case_ids or [])
    return await run_llm_fixture(
        filtered,
        fixture_path=str(path),
        provider=provider,
        model=model,
        env=env,
        client_factory=client_factory,
    )


async def run_fixture(
    fixture: BenchmarkFixture,
    *,
    fixture_path: str | None = None,
    min_score: float = 1.0,
) -> dict[str, Any]:
    """Evaluate all cases in a benchmark fixture."""
    case_results = [await _run_case(case) for case in fixture.cases]
    full_score = _average(item["score"] for item in case_results)
    baseline_scores = _baseline_scores(fixture.cases)
    marginal = {
        name: round(full_score - score, 6) for name, score in sorted(baseline_scores.items())
    }
    metrics = _metrics(case_results)
    all_cases_passed = all(item["passed"] for item in case_results)
    status = "pass" if full_score >= min_score and all_cases_passed else "fail"
    return {
        "status": status,
        "fixture": fixture.name,
        "fixture_path": fixture_path,
        "case_count": len(case_results),
        "passed": sum(1 for item in case_results if item["passed"]),
        "failed": [item["id"] for item in case_results if not item["passed"]],
        "full_mcp_score": round(full_score, 6),
        "baseline_scores": baseline_scores,
        "marginal_contribution": marginal,
        "metrics": metrics,
        "cases": case_results,
    }


async def run_llm_fixture(
    fixture: BenchmarkFixture,
    *,
    fixture_path: str | None = None,
    provider: EvalProvider = "openai",
    model: str = "gpt-5.2",
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a live LLM-only baseline and compare it with the deterministic MCP run."""
    if provider != "openai":
        raise ValueError(f"unsupported eval provider: {provider}")
    env_map = os.environ if env is None else env
    api_key = env_map.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for eval llm with provider=openai")

    mcp_report = await run_fixture(fixture, fixture_path=fixture_path)
    llm_results = [
        await _run_llm_case(case, model=model, api_key=api_key, client_factory=client_factory)
        for case in fixture.cases
    ]
    llm_score = round(_average(item["score"] for item in llm_results), 6)
    return {
        "status": "pass" if mcp_report["status"] == "pass" else "fail",
        "fixture": fixture.name,
        "fixture_path": fixture_path,
        "provider": provider,
        "model": model,
        "case_count": len(llm_results),
        "llm_passed": sum(1 for item in llm_results if item["passed"]),
        "llm_failed": [item["id"] for item in llm_results if not item["passed"]],
        "full_mcp_score": mcp_report["full_mcp_score"],
        "live_baseline_scores": {OPENAI_LLM_ONLY: llm_score},
        "live_marginal_contribution": {
            OPENAI_LLM_ONLY: round(mcp_report["full_mcp_score"] - llm_score, 6)
        },
        "metrics": mcp_report["metrics"],
        "cases": llm_results,
        "mcp_report": mcp_report,
    }


async def _run_case(case: BenchmarkCase) -> dict[str, Any]:
    store = RelationStore()
    if case.relation_set != RelationSetData():
        loaded = await call_tool(
            LOAD_RELATIONS,
            {
                "source_type": "inline",
                "data": case.relation_set.model_dump(mode="json"),
                "check_contradictions": False,
            },
            store,
        )
        if loaded.isError:
            return _failed_case(case, ["relation_set failed to load"], 0.0)
    started = time.perf_counter()
    try:
        result = await call_tool(case.tool_name, case.tool_input, store)
    except Exception as exc:  # pragma: no cover - defensive fixture error path
        return _failed_case(case, [f"tool call failed: {exc}"], 0.0)
    latency_ms = (time.perf_counter() - started) * 1000
    structured = result.structuredContent or {}
    failures = _match_failures(structured, case.expected)
    passed = not failures and not result.isError
    return {
        "id": case.id,
        "category": case.category,
        "tool_name": case.tool_name,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "latency_ms": round(latency_ms, 3),
        "failures": failures,
        "expected_equals": case.expected.equals,
        "observed": _observed_fields(structured),
    }


def _failed_case(
    case: BenchmarkCase,
    failures: list[str],
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "tool_name": case.tool_name,
        "passed": False,
        "score": 0.0,
        "latency_ms": round(latency_ms, 3),
        "failures": failures,
        "expected_equals": case.expected.equals,
        "observed": {},
    }


async def _run_llm_case(
    case: BenchmarkCase,
    *,
    model: str,
    api_key: str,
    client_factory: Callable[[str], Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        prediction = await _openai_prediction(case, model, api_key, client_factory)
    except json.JSONDecodeError:
        latency_ms = (time.perf_counter() - started) * 1000
        return _failed_llm_case(case, model, ["model output was not valid JSON"], latency_ms)
    except ValueError:
        raise
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        return _failed_llm_case(case, model, ["provider call failed"], latency_ms)

    latency_ms = (time.perf_counter() - started) * 1000
    failures = _match_failures(prediction, case.expected)
    passed = not failures
    return {
        "id": case.id,
        "category": case.category,
        "tool_name": case.tool_name,
        "model": model,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "latency_ms": round(latency_ms, 3),
        "failures": failures,
        "prediction_keys": sorted(str(key) for key in prediction),
    }


def _failed_llm_case(
    case: BenchmarkCase,
    model: str,
    failures: list[str],
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "tool_name": case.tool_name,
        "model": model,
        "passed": False,
        "score": 0.0,
        "latency_ms": round(latency_ms, 3),
        "failures": failures,
        "prediction_keys": [],
    }


def _match_failures(payload: dict[str, Any], expected: ExpectedSpec) -> list[str]:
    failures: list[str] = []
    for path, expected_value in expected.equals.items():
        values = _values_at_path(payload, path)
        if not values or not any(value == expected_value for value in values):
            failures.append(f"{path} expected {expected_value!r}, got {values!r}")
    for path, expected_value in expected.contains.items():
        values = _values_at_path(payload, path)
        if expected_value not in values:
            failures.append(f"{path} expected to contain {expected_value!r}, got {values!r}")
    for path, minimum in expected.min_count.items():
        values = _values_at_path(payload, path)
        if len(values) < minimum:
            failures.append(f"{path} expected at least {minimum}, got {len(values)}")
    return failures


async def _openai_prediction(
    case: BenchmarkCase,
    model: str,
    api_key: str,
    client_factory: Callable[[str], Any] | None,
) -> dict[str, Any]:
    client = client_factory(api_key) if client_factory else _openai_client(api_key)
    prompt = _llm_prompt(case)
    response = await anyio.to_thread.run_sync(
        lambda: client.responses.create(model=model, input=prompt)
    )
    output_text = getattr(response, "output_text", "")
    if not isinstance(output_text, str):
        raise json.JSONDecodeError("output_text is not a string", "", 0)
    parsed = _parse_json_prediction(output_text)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("model output must be a JSON object", output_text, 0)
    return parsed


def _openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ValueError(
            "openai package is required for eval llm; install with `uv sync --extra eval`"
        ) from exc
    return OpenAI(api_key=api_key)


def _llm_prompt(case: BenchmarkCase) -> str:
    payload = {
        "id": case.id,
        "category": case.category,
        "description": case.description,
        "relation_set": case.relation_set.model_dump(mode="json"),
        "tool_name": case.tool_name,
        "tool_input": case.tool_input,
        "expected_matchers": {
            "equals": case.expected.equals,
            "contains": case.expected.contains,
            "min_count": case.expected.min_count,
        },
    }
    return (
        "You are evaluating a structured reasoning case without calling MCP tools. "
        "Infer the expected tool result from the relation_set and tool_input. "
        "Return only one JSON object. The object should include the fields needed "
        "to satisfy expected_matchers. Do not use markdown.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _parse_json_prediction(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(stripped[start : end + 1])


def _filter_fixture_cases(
    fixture: BenchmarkFixture,
    case_ids: list[str],
) -> BenchmarkFixture:
    if not case_ids:
        return fixture
    requested = set(case_ids)
    selected = [case for case in fixture.cases if case.id in requested]
    found = {case.id for case in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"unknown benchmark case id(s): {', '.join(missing)}")
    return BenchmarkFixture(version=fixture.version, name=fixture.name, cases=selected)


def _values_at_path(payload: Any, path: str) -> list[Any]:
    values = [payload]
    for part in path.split("."):
        is_array = part.endswith("[]")
        key = part[:-2] if is_array else part
        next_values: list[Any] = []
        for value in values:
            selected = value
            if key:
                if not isinstance(value, dict) or key not in value:
                    continue
                selected = value[key]
            if is_array:
                if isinstance(selected, list):
                    next_values.extend(selected)
            else:
                next_values.append(selected)
        values = next_values
    return values


def _observed_fields(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace")
    return {
        "status": payload.get("status"),
        "classification": payload.get("classification"),
        "relation_type": payload.get("relation_type"),
        "has_contradictions": payload.get("has_contradictions"),
        "world_mode": payload.get("world_mode"),
        "trace_count": len(trace) if isinstance(trace, list) else 0,
    }


def _baseline_scores(cases: list[BenchmarkCase]) -> dict[str, float]:
    names = sorted({name for case in cases for name in case.baselines})
    return {
        name: round(_average(case.baselines[name] for case in cases if name in case.baselines), 6)
        for name in names
    }


def _metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    logical = [
        item["score"]
        for item in case_results
        if item["category"] in {"classification", "transitive", "business"}
    ]
    contradiction_positive = [
        item
        for item in case_results
        if item["category"] == "contradiction"
        and item["expected_equals"].get("has_contradictions") is True
    ]
    contradiction_negative = [
        item
        for item in case_results
        if item["category"] == "contradiction"
        and item["expected_equals"].get("has_contradictions") is False
    ]
    false_positives = [
        item
        for item in contradiction_negative
        if item["observed"].get("has_contradictions") is True
    ]
    counterfactual = [
        item["score"] for item in case_results if item["category"] == "counterfactual"
    ]
    trace_complete = [
        1.0 if item["observed"].get("trace_count", 0) > 0 else 0.0 for item in case_results
    ]
    return {
        "logical_accuracy": round(_average(logical), 6),
        "contradiction_recall": round(
            _average(item["score"] for item in contradiction_positive),
            6,
        ),
        "false_contradiction_rate": round(
            len(false_positives) / len(contradiction_negative) if contradiction_negative else 0.0,
            6,
        ),
        "counterfactual_conservatism": round(_average(counterfactual), 6),
        "trace_completeness": round(_average(trace_complete), 6),
        "latency_ms_avg": round(_average(item["latency_ms"] for item in case_results), 3),
    }


def _average(values: Any) -> float:
    items = list(values)
    return sum(float(item) for item in items) / len(items) if items else 0.0


def _text_report(report: dict[str, Any]) -> str:
    lines = [
        f"NeSy evaluation: {report['status']}",
        f"fixture: {report['fixture']}",
        f"cases: {report['passed']}/{report['case_count']}",
        f"full_mcp_score: {report['full_mcp_score']:.3f}",
    ]
    lines.append("metrics:")
    lines.extend(f"- {key}: {value}" for key, value in report["metrics"].items())
    lines.append("marginal_contribution:")
    lines.extend(f"- {key}: {value}" for key, value in report["marginal_contribution"].items())
    if report["failed"]:
        lines.append(f"failed: {', '.join(report['failed'])}")
    return "\n".join(lines)


def _llm_text_report(report: dict[str, Any]) -> str:
    live_score = report["live_baseline_scores"][OPENAI_LLM_ONLY]
    marginal = report["live_marginal_contribution"][OPENAI_LLM_ONLY]
    lines = [
        f"NeSy live evaluation: {report['status']}",
        f"fixture: {report['fixture']}",
        f"provider: {report['provider']}",
        f"model: {report['model']}",
        f"cases: {report['llm_passed']}/{report['case_count']}",
        f"full_mcp_score: {report['full_mcp_score']:.3f}",
        f"openai_llm_only: {live_score:.3f}",
        f"marginal_contribution: {marginal:.3f}",
    ]
    if report["llm_failed"]:
        lines.append(f"llm_failed: {', '.join(report['llm_failed'])}")
    return "\n".join(lines)


def _json_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
