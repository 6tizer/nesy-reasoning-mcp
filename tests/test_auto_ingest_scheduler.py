import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import pytest

from nesy_reasoning_mcp.auto_ingest import cli as ingest_cli
from nesy_reasoning_mcp.auto_ingest.scheduler import (
    ScheduledIngestionJob,
    ScheduledIngestionJobStatus,
    ScheduledIngestionProviderConfig,
    ScheduledIngestionRunStatus,
    ScheduledIngestionRunTrigger,
    ScheduledIngestionSourceConfig,
    ScheduledIngestionState,
    ScheduledIngestionWriteConfig,
    next_cron_run,
)
from nesy_reasoning_mcp.auto_ingest.schemas import IngestionReport
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.store import RelationStore


def _job(tmp_path: Path, **kwargs: Any) -> ScheduledIngestionJob:
    name = kwargs.pop("name", "docs ingestion")
    cron = kwargs.pop("cron", "*/15 * * * *")
    state = kwargs.pop(
        "state",
        ScheduledIngestionState(next_run_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()),
    )
    provider_config = kwargs.pop("provider_config", ScheduledIngestionProviderConfig())
    write_config = kwargs.pop(
        "write_config",
        ScheduledIngestionWriteConfig(report_dir=str(tmp_path / "reports")),
    )
    return ScheduledIngestionJob(
        name=name,
        cron=cron,
        source_config=ScheduledIngestionSourceConfig(urls=["https://example.com/source"]),
        provider_config=provider_config,
        write_config=write_config,
        state=state,
        **kwargs,
    )


def test_next_cron_run_supports_steps_and_timezone() -> None:
    after = datetime(2026, 1, 1, 0, 7, tzinfo=UTC)

    assert next_cron_run("*/15 * * * *", "UTC", after=after) == ("2026-01-01T00:15:00+00:00")


def test_next_cron_run_supports_aliases_and_sparse_schedule() -> None:
    after = datetime(2026, 1, 1, 0, 7, tzinfo=UTC)

    assert next_cron_run("@hourly", "UTC", after=after) == "2026-01-01T01:00:00+00:00"
    assert next_cron_run("@yearly", "UTC", after=after) == "2027-01-01T00:00:00+00:00"


def test_scheduled_job_strips_cron_and_timezone(tmp_path: Path) -> None:
    job = _job(tmp_path, cron="  @daily  ", timezone="  UTC  ")

    assert job.cron == "@daily"
    assert job.timezone == "UTC"


def test_scheduled_job_rejects_unknown_fields() -> None:
    payload = _job(Path("/tmp")).model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValueError):
        ScheduledIngestionJob.model_validate(payload)


def test_scheduled_write_requires_multi_reviewer_by_default(tmp_path: Path) -> None:
    args = argparse.Namespace(
        name="write job",
        cron="*/30 * * * *",
        timezone="UTC",
        input=None,
        url=["https://example.com/source"],
        retrieval_input=None,
        task=None,
        question=None,
        timeout_seconds=3.0,
        max_url_bytes=1000,
        search_queries=[],
        search_provider="exa",
        search_limit=5,
        search_timeout_seconds=3.0,
        search_include_domains=[],
        search_exclude_domains=[],
        search_api_key_env="EXA_API_KEY",
        crawl=False,
        crawl_max_depth=1,
        crawl_max_pages=10,
        crawl_max_page_bytes=1000,
        crawl_max_total_bytes=5000,
        crawl_timeout_seconds=3.0,
        crawl_allow_domains=[],
        model=None,
        provider=None,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        disable_tracing=False,
        reviewer_models=["reviewer-a"],
        voting_policy="risk_tiered",
        high_priority_reviewer_models=[],
        auto_write=True,
        allow_scheduled_writes=True,
        allow_single_reviewer_write=False,
        min_write_confidence=0.85,
        report_dir=str(tmp_path / "reports"),
        max_retries=0,
        retry_backoff_seconds=60,
    )

    with pytest.raises(ValueError, match="at least two reviewer models"):
        ingest_cli._scheduled_job_from_args(args)


def test_scheduled_provider_thinking_overrides_round_trip(tmp_path: Path) -> None:
    args = argparse.Namespace(
        name="deepseek job",
        cron="*/30 * * * *",
        timezone="UTC",
        input=None,
        url=["https://example.com/source"],
        retrieval_input=None,
        task=None,
        question=None,
        timeout_seconds=3.0,
        max_url_bytes=1000,
        search_queries=[],
        search_provider="exa",
        search_limit=5,
        search_timeout_seconds=3.0,
        search_include_domains=[],
        search_exclude_domains=[],
        search_api_key_env="EXA_API_KEY",
        crawl=False,
        crawl_max_depth=1,
        crawl_max_pages=10,
        crawl_max_page_bytes=1000,
        crawl_max_total_bytes=5000,
        crawl_timeout_seconds=3.0,
        crawl_allow_domains=[],
        model=None,
        provider="deepseek",
        base_url=None,
        api_key_env=None,
        provider_header=[],
        provider_thinking="disabled",
        provider_reasoning_effort="max",
        disable_tracing=False,
        reviewer_models=[],
        voting_policy="risk_tiered",
        high_priority_reviewer_models=[],
        auto_write=False,
        allow_scheduled_writes=False,
        allow_single_reviewer_write=False,
        min_write_confidence=0.85,
        report_dir=str(tmp_path / "reports"),
        max_retries=0,
        retry_backoff_seconds=60,
    )

    job = ingest_cli._scheduled_job_from_args(args)
    round_tripped = ingest_cli._scheduled_job_args(job)

    assert job.provider_config.provider_thinking == "disabled"
    assert job.provider_config.provider_reasoning_effort == "max"
    assert round_tripped.provider_thinking == "disabled"
    assert round_tripped.provider_reasoning_effort == "max"


async def test_scheduled_dry_run_writes_report_without_graph_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run(args: argparse.Namespace) -> IngestionReport:
        assert args.auto_write is False
        return IngestionReport(metadata={"scheduled": True})

    monkeypatch.setattr(ingest_cli, "_run_agent_dry_run", fake_run)
    store = RelationStore()
    job = _job(tmp_path)
    store.upsert_scheduled_ingestion_job(job)

    run = await ingest_cli._run_scheduled_ingestion_job(
        job,
        store,
        trigger=ScheduledIngestionRunTrigger.MANUAL,
    )

    updated = store.get_scheduled_ingestion_job(job.id)
    assert run.status == ScheduledIngestionRunStatus.SUCCEEDED
    assert run.report_path is not None
    assert json.loads(Path(run.report_path).read_text(encoding="utf-8"))["metadata"] == {
        "scheduled": True
    }
    assert updated is not None
    assert updated.state.last_run_id == run.id
    assert updated.state.retry_count == 0
    assert store.list_relations() == []


async def test_scheduled_runtime_skips_unsafe_auto_write(tmp_path: Path) -> None:
    store = RelationStore()
    job = _job(
        tmp_path,
        provider_config=ScheduledIngestionProviderConfig(reviewer_models=["reviewer-a"]),
        write_config=ScheduledIngestionWriteConfig(
            auto_write=True,
            allow_scheduled_writes=False,
            report_dir=str(tmp_path / "reports"),
        ),
    )
    store.upsert_scheduled_ingestion_job(job)

    run = await ingest_cli._run_scheduled_ingestion_job(
        job,
        store,
        trigger=ScheduledIngestionRunTrigger.DUE,
    )

    updated = store.get_scheduled_ingestion_job(job.id)
    assert run.status == ScheduledIngestionRunStatus.SKIPPED
    assert run.diagnostics[0].code == "SCHEDULED_WRITE_NOT_ALLOWED"
    assert updated is not None
    assert updated.status == ScheduledIngestionJobStatus.FAILED


async def test_scheduled_write_warning_does_not_block_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run(args: argparse.Namespace) -> IngestionReport:
        return IngestionReport()

    monkeypatch.setattr(ingest_cli, "_run_agent_dry_run", fake_run)
    store = RelationStore()
    job = _job(
        tmp_path,
        write_config=ScheduledIngestionWriteConfig(
            auto_write=False,
            allow_scheduled_writes=True,
            report_dir=str(tmp_path / "reports"),
        ),
    )
    store.upsert_scheduled_ingestion_job(job)

    run = await ingest_cli._run_scheduled_ingestion_job(
        job,
        store,
        trigger=ScheduledIngestionRunTrigger.MANUAL,
    )

    assert run.status == ScheduledIngestionRunStatus.SUCCEEDED
    assert run.diagnostics[0].code == "SCHEDULED_WRITE_CONFIG"


async def test_run_due_runs_only_due_active_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run(args: argparse.Namespace) -> IngestionReport:
        return IngestionReport()

    monkeypatch.setattr(ingest_cli, "_run_agent_dry_run", fake_run)
    store = RelationStore()
    due = _job(tmp_path, name="due")
    future = _job(
        tmp_path,
        name="future",
        state=ScheduledIngestionState(
            next_run_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat()
        ),
    )
    disabled = _job(tmp_path, name="disabled", status=ScheduledIngestionJobStatus.DISABLED)
    for job in [due, future, disabled]:
        store.upsert_scheduled_ingestion_job(job)

    runs = await ingest_cli._run_due_scheduled_jobs(
        store,
        trigger=ScheduledIngestionRunTrigger.DUE,
        limit=10,
    )

    assert [run.job_id for run in runs] == [due.id]


async def test_failed_scheduled_run_records_retry_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run(args: argparse.Namespace) -> IngestionReport:
        return IngestionReport(
            diagnostics=[Diagnostic(level="error", code="INGESTION_ERROR", message="failed")]
        )

    monkeypatch.setattr(ingest_cli, "_run_agent_dry_run", fake_run)
    store = RelationStore()
    job = _job(tmp_path)
    store.upsert_scheduled_ingestion_job(job)

    run = await ingest_cli._run_scheduled_ingestion_job(
        job,
        store,
        trigger=ScheduledIngestionRunTrigger.DUE,
    )

    updated = store.get_scheduled_ingestion_job(job.id)
    assert run.status == ScheduledIngestionRunStatus.FAILED
    assert updated is not None
    assert updated.state.retry_count == 1
    assert updated.status == ScheduledIngestionJobStatus.ACTIVE


async def test_scheduled_run_skips_when_job_claim_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run(args: argparse.Namespace) -> IngestionReport:
        pytest.fail("stale scheduled job should not run")

    monkeypatch.setattr(ingest_cli, "_run_agent_dry_run", fake_run)
    store = RelationStore()
    job = _job(tmp_path)
    store.upsert_scheduled_ingestion_job(job)
    store.update_scheduled_ingestion_job_state(
        job.id,
        state=job.state.model_copy(update={"current_run_id": "other-run"}),
        status=ScheduledIngestionJobStatus.RUNNING,
    )

    run = await ingest_cli._run_scheduled_ingestion_job(
        job,
        store,
        trigger=ScheduledIngestionRunTrigger.WORKER,
    )

    assert run.status == ScheduledIngestionRunStatus.SKIPPED
    assert run.diagnostics[0].code == "SCHEDULED_JOB_CLAIM_CONFLICT"


async def test_cancelled_scheduled_run_persists_failure_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cancelled_exc_class = anyio.get_cancelled_exc_class()

    async def fake_run(args: argparse.Namespace) -> IngestionReport:
        raise cancelled_exc_class()

    monkeypatch.setattr(ingest_cli, "_run_agent_dry_run", fake_run)
    store = RelationStore()
    job = _job(tmp_path)
    store.upsert_scheduled_ingestion_job(job)

    with pytest.raises(cancelled_exc_class):
        await ingest_cli._run_scheduled_ingestion_job(
            job,
            store,
            trigger=ScheduledIngestionRunTrigger.WORKER,
        )

    runs = store.list_scheduled_ingestion_runs()
    updated = store.get_scheduled_ingestion_job(job.id)
    assert len(runs) == 1
    assert runs[0].status == ScheduledIngestionRunStatus.FAILED
    assert runs[0].diagnostics[0].code == "SCHEDULED_INGESTION_RUN_CANCELLED"
    assert updated is not None
    assert updated.status == ScheduledIngestionJobStatus.ACTIVE
    assert updated.state.current_run_id is None
