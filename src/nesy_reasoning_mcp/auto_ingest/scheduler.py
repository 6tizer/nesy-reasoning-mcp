"""Scheduled Agent SDK ingestion job schemas and cron helpers."""

from __future__ import annotations

import json
from bisect import bisect_left
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nesy_reasoning_mcp.auto_ingest.crawler import (
    DEFAULT_CRAWL_MAX_DEPTH,
    DEFAULT_CRAWL_MAX_PAGE_BYTES,
    DEFAULT_CRAWL_MAX_PAGES,
    DEFAULT_CRAWL_MAX_TOTAL_BYTES,
    DEFAULT_CRAWL_TIMEOUT_SECONDS,
)
from nesy_reasoning_mcp.auto_ingest.fetcher import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_FETCH_BYTES,
)
from nesy_reasoning_mcp.auto_ingest.schemas import ReviewVotingPolicy
from nesy_reasoning_mcp.auto_ingest.search import (
    DEFAULT_SEARCH_API_KEY_ENV,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_SEARCH_TIMEOUT_SECONDS,
)
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.storage.common import _utc_now_iso

DEFAULT_SCHEDULE_TIMEZONE = "UTC"
DEFAULT_SCHEDULE_MAX_RETRIES = 2
DEFAULT_SCHEDULE_RETRY_BACKOFF_SECONDS = 300
DEFAULT_SCHEDULE_POLL_SECONDS = 60.0
MAX_SCHEDULE_POLL_SECONDS = 3600.0
MAX_CRON_LOOKAHEAD_MINUTES = 366 * 24 * 60

CRON_ALIASES = {
    "@annually": "0 0 1 1 *",
    "@yearly": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}


class ScheduledIngestionJobStatus(StrEnum):
    """Lifecycle status for a scheduled ingestion job."""

    ACTIVE = "active"
    DISABLED = "disabled"
    RUNNING = "running"
    FAILED = "failed"


class ScheduledIngestionRunStatus(StrEnum):
    """Lifecycle status for one scheduled ingestion run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScheduledIngestionRunTrigger(StrEnum):
    """Trigger source for a scheduled ingestion run."""

    MANUAL = "manual"
    DUE = "due"
    WORKER = "worker"


class ScheduledIngestionSourceConfig(BaseModel):
    """Persisted source inputs for one scheduled Agent SDK ingestion job."""

    model_config = ConfigDict(extra="forbid")

    input_path: str | None = None
    urls: list[str] = Field(default_factory=list)
    retrieval_input_path: str | None = None
    task: str | None = None
    question: str | None = None
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS
    max_url_bytes: int = DEFAULT_MAX_FETCH_BYTES
    search_queries: list[str] = Field(default_factory=list)
    search_provider: str = DEFAULT_SEARCH_PROVIDER
    search_limit: int = DEFAULT_SEARCH_LIMIT
    search_timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS
    search_include_domains: list[str] = Field(default_factory=list)
    search_exclude_domains: list[str] = Field(default_factory=list)
    search_api_key_env: str = DEFAULT_SEARCH_API_KEY_ENV
    crawl: bool = False
    crawl_max_depth: int = DEFAULT_CRAWL_MAX_DEPTH
    crawl_max_pages: int = DEFAULT_CRAWL_MAX_PAGES
    crawl_max_page_bytes: int = DEFAULT_CRAWL_MAX_PAGE_BYTES
    crawl_max_total_bytes: int = DEFAULT_CRAWL_MAX_TOTAL_BYTES
    crawl_timeout_seconds: float = DEFAULT_CRAWL_TIMEOUT_SECONDS
    crawl_allow_domains: list[str] = Field(default_factory=list)

    @field_validator("input_path", "retrieval_input_path", "task", "question", "search_api_key_env")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip optional text values and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator(
        "urls",
        "search_queries",
        "search_include_domains",
        "search_exclude_domains",
        "crawl_allow_domains",
    )
    @classmethod
    def strip_string_lists(cls, value: list[str]) -> list[str]:
        """Strip string list values and reject empty entries."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("must not contain empty values")
        return stripped

    @model_validator(mode="after")
    def require_source(self) -> ScheduledIngestionSourceConfig:
        """Require at least one configured source path, URL, retrieval batch, or search query."""
        if not (self.input_path or self.urls or self.retrieval_input_path or self.search_queries):
            raise ValueError(
                "scheduled ingestion job requires an input, retrieval input, URL, or search query"
            )
        if self.crawl and not self.urls:
            raise ValueError("--crawl requires at least one scheduled URL")
        return self


class ScheduledIngestionProviderConfig(BaseModel):
    """Persisted model/provider inputs for one scheduled Agent SDK ingestion job."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    provider_headers: list[str] = Field(default_factory=list)
    provider_thinking: str | None = None
    provider_reasoning_effort: str | None = None
    disable_tracing: bool = False
    reviewer_models: list[str] = Field(default_factory=list)
    reviewers: list[str] = Field(default_factory=list)
    voting_policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED
    high_priority_reviewer_models: list[str] = Field(default_factory=list)
    high_priority_reviewers: list[str] = Field(default_factory=list)

    @field_validator(
        "model",
        "provider",
        "base_url",
        "api_key_env",
        "provider_thinking",
        "provider_reasoning_effort",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip optional text values and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator(
        "provider_headers",
        "reviewer_models",
        "reviewers",
        "high_priority_reviewer_models",
        "high_priority_reviewers",
    )
    @classmethod
    def strip_string_lists(cls, value: list[str]) -> list[str]:
        """Strip string list values and reject empty entries."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("must not contain empty values")
        return stripped

    @field_validator("provider_thinking")
    @classmethod
    def validate_provider_thinking(cls, value: str | None) -> str | None:
        """Validate provider thinking mode overrides."""
        if value is None or value in {"enabled", "disabled"}:
            return value
        raise ValueError("must be enabled or disabled")

    @field_validator("provider_reasoning_effort")
    @classmethod
    def validate_provider_reasoning_effort(cls, value: str | None) -> str | None:
        """Validate provider reasoning effort overrides."""
        if value is None or value in {"high", "max"}:
            return value
        raise ValueError("must be high or max")


class ScheduledIngestionWriteConfig(BaseModel):
    """Persisted write controls for one scheduled Agent SDK ingestion job."""

    model_config = ConfigDict(extra="forbid")

    auto_write: bool = False
    allow_scheduled_writes: bool = False
    allow_single_reviewer_write: bool = False
    min_write_confidence: float = Field(default=0.85, ge=0, le=1)
    report_dir: str | None = None

    @field_validator("report_dir")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip optional report directory and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ScheduledIngestionRetryPolicy(BaseModel):
    """Retry policy for failed scheduled ingestion runs."""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=DEFAULT_SCHEDULE_MAX_RETRIES, ge=0, le=10)
    retry_backoff_seconds: int = Field(
        default=DEFAULT_SCHEDULE_RETRY_BACKOFF_SECONDS,
        ge=1,
        le=24 * 60 * 60,
    )


class ScheduledIngestionState(BaseModel):
    """Mutable run state for one scheduled ingestion job."""

    model_config = ConfigDict(extra="forbid")

    last_run_at: str | None = None
    next_run_at: str | None = None
    last_status: ScheduledIngestionRunStatus | None = None
    retry_count: int = Field(default=0, ge=0)
    current_run_id: str | None = None
    last_run_id: str | None = None
    last_report_run_id: str | None = None
    last_report_path: str | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @field_validator(
        "last_run_at",
        "next_run_at",
        "current_run_id",
        "last_run_id",
        "last_report_run_id",
        "last_report_path",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip optional text values and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ScheduledIngestionJob(BaseModel):
    """A persisted scheduled Agent SDK ingestion job."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"sched_{uuid4().hex}", min_length=1)
    name: str = Field(min_length=1)
    status: ScheduledIngestionJobStatus = ScheduledIngestionJobStatus.ACTIVE
    cron: str = Field(min_length=1)
    timezone: str = DEFAULT_SCHEDULE_TIMEZONE
    source_config: ScheduledIngestionSourceConfig
    provider_config: ScheduledIngestionProviderConfig = Field(
        default_factory=ScheduledIngestionProviderConfig
    )
    write_config: ScheduledIngestionWriteConfig = Field(
        default_factory=ScheduledIngestionWriteConfig
    )
    retry_policy: ScheduledIngestionRetryPolicy = Field(
        default_factory=ScheduledIngestionRetryPolicy
    )
    state: ScheduledIngestionState = Field(default_factory=ScheduledIngestionState)
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)

    @model_validator(mode="before")
    @classmethod
    def set_default_timestamps(cls, value: Any) -> Any:
        """Use one timestamp for generated creation/update defaults."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("created_at") is None and data.get("updated_at") is None:
            timestamp = _utc_now_iso()
            data["created_at"] = timestamp
            data["updated_at"] = timestamp
        elif data.get("created_at") is None:
            data["created_at"] = data["updated_at"]
        elif data.get("updated_at") is None:
            data["updated_at"] = data["created_at"]
        return data

    @field_validator("id", "name", "created_at", "updated_at")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Strip required text and reject empty values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("cron")
    @classmethod
    def validate_cron_expression(cls, value: str) -> str:
        """Validate cron syntax."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        validate_cron(stripped)
        return stripped

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Validate schedule timezone."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        _zoneinfo(stripped)
        return stripped


class ScheduledIngestionRun(BaseModel):
    """One persisted scheduled ingestion run attempt."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"srun_{uuid4().hex}", min_length=1)
    job_id: str = Field(min_length=1)
    trigger: ScheduledIngestionRunTrigger
    status: ScheduledIngestionRunStatus = ScheduledIngestionRunStatus.RUNNING
    attempt: int = Field(default=1, ge=1)
    started_at: str = Field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    report_run_id: str | None = None
    report_path: str | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "id",
        "job_id",
        "started_at",
        "finished_at",
        "report_run_id",
        "report_path",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text values and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ScheduledIngestionJobFilter(BaseModel):
    """Filter for listing scheduled ingestion jobs."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(default_factory=list)
    status: ScheduledIngestionJobStatus | None = None
    due_before: str | None = None


class ScheduledIngestionRunFilter(BaseModel):
    """Filter for listing scheduled ingestion runs."""

    model_config = ConfigDict(extra="forbid")

    job_id: str | None = None
    status: ScheduledIngestionRunStatus | None = None


def validate_cron(cron: str) -> None:
    """Validate a supported five-field cron expression."""
    _parse_cron(cron)


def next_cron_run(
    cron: str,
    timezone: str,
    *,
    after: datetime | None = None,
) -> str:
    """Return the next cron fire time after `after` as an ISO UTC timestamp."""
    fields = _parse_cron(cron)
    zone = _zoneinfo(timezone)
    base = after or datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    current = base.astimezone(zone).replace(second=0, microsecond=0) + timedelta(minutes=1)
    deadline = current + timedelta(minutes=MAX_CRON_LOOKAHEAD_MINUTES)
    minute_values = sorted(fields["minute"])
    hour_values = sorted(fields["hour"])
    month_values = sorted(fields["month"])
    while current <= deadline:
        if current.month not in fields["month"]:
            current = _next_allowed_month(
                current,
                month_values,
                hour=hour_values[0],
                minute=minute_values[0],
            )
            continue
        if not _cron_day_matches(current, fields):
            current = _next_day_start(current, hour=hour_values[0], minute=minute_values[0])
            continue
        if current.hour not in fields["hour"]:
            next_hour = _next_value_at_or_after(current.hour, hour_values)
            current = (
                current.replace(hour=next_hour, minute=minute_values[0])
                if next_hour is not None
                else _next_day_start(current, hour=hour_values[0], minute=minute_values[0])
            )
            continue
        if current.minute not in fields["minute"]:
            next_minute = _next_value_at_or_after(current.minute, minute_values)
            if next_minute is not None:
                current = current.replace(minute=next_minute)
                continue
            next_hour = _next_value_after(current.hour, hour_values)
            current = (
                current.replace(hour=next_hour, minute=minute_values[0])
                if next_hour is not None
                else _next_day_start(current, hour=hour_values[0], minute=minute_values[0])
            )
            continue
        if _cron_fields_match(current, fields):
            return current.astimezone(UTC).isoformat()
    raise ValueError("cron expression has no matching run time in the next year")


def scheduler_report_path(
    job: ScheduledIngestionJob,
    run: ScheduledIngestionRun,
) -> Path:
    """Return the report path for a scheduled run."""
    base = (
        Path(job.write_config.report_dir).expanduser()
        if job.write_config.report_dir
        else Path.home() / ".nesy-reasoning" / "ingestion-reports"
    )
    return base / job.id / f"{run.id}.json"


def write_scheduled_report(
    job: ScheduledIngestionJob,
    run: ScheduledIngestionRun,
    payload: dict[str, Any],
) -> str:
    """Persist one scheduled ingestion report and return its path."""
    path = scheduler_report_path(job, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(path)


def scheduled_write_diagnostics(job: ScheduledIngestionJob) -> list[Diagnostic]:
    """Return runtime safety diagnostics for unsafe scheduled write settings."""
    diagnostics: list[Diagnostic] = []
    if not job.write_config.auto_write and job.write_config.allow_scheduled_writes:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="SCHEDULED_WRITE_CONFIG",
                message="scheduled writes are configured but auto-write is disabled",
            )
        )
    if not job.write_config.auto_write:
        return diagnostics
    if not job.write_config.allow_scheduled_writes:
        diagnostics.append(
            Diagnostic(
                level="error",
                code="SCHEDULED_WRITE_NOT_ALLOWED",
                message="scheduled auto-write requires allow_scheduled_writes=true",
                related_ids=[job.id],
            )
        )
    reviewer_count = scheduled_reviewer_count(job.provider_config)
    if reviewer_count < 2 and not job.write_config.allow_single_reviewer_write:
        diagnostics.append(
            Diagnostic(
                level="error",
                code="SCHEDULED_WRITE_REQUIRES_MULTI_REVIEWER",
                message=(
                    "scheduled auto-write requires at least two reviewer models or "
                    "allow_single_reviewer_write=true"
                ),
                related_ids=[job.id],
            )
        )
    return diagnostics


def scheduled_reviewer_count(provider_config: ScheduledIngestionProviderConfig) -> int:
    """Return the number of distinct scheduled reviewers."""
    return len(dict.fromkeys([*provider_config.reviewer_models, *provider_config.reviewers]))


def job_due(job: ScheduledIngestionJob, *, now: datetime | None = None) -> bool:
    """Return whether a job is active and due at `now`."""
    if job.status != ScheduledIngestionJobStatus.ACTIVE or not job.state.next_run_at:
        return False
    threshold = now or datetime.now(UTC)
    due_at = datetime.fromisoformat(job.state.next_run_at)
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=UTC)
    return due_at <= threshold.astimezone(UTC)


def next_state_for_success(
    job: ScheduledIngestionJob,
    run: ScheduledIngestionRun,
    *,
    now: datetime | None = None,
) -> ScheduledIngestionState:
    """Build the next job state after a successful run."""
    finished_at = run.finished_at or _utc_now_iso()
    return job.state.model_copy(
        deep=True,
        update={
            "last_run_at": finished_at,
            "next_run_at": next_cron_run(
                job.cron,
                job.timezone,
                after=now or _parse_iso(finished_at),
            ),
            "last_status": ScheduledIngestionRunStatus.SUCCEEDED,
            "retry_count": 0,
            "current_run_id": None,
            "last_run_id": run.id,
            "last_report_run_id": run.report_run_id,
            "last_report_path": run.report_path,
            "diagnostics": run.diagnostics,
        },
    )


def next_state_for_failure(
    job: ScheduledIngestionJob,
    run: ScheduledIngestionRun,
    *,
    now: datetime | None = None,
) -> tuple[ScheduledIngestionJobStatus, ScheduledIngestionState]:
    """Build the next job status/state after a failed run."""
    finished_at = run.finished_at or _utc_now_iso()
    base_time = now or _parse_iso(finished_at)
    if job.state.retry_count < job.retry_policy.max_retries:
        retry_count = job.state.retry_count + 1
        next_run_at = (
            base_time.astimezone(UTC) + timedelta(seconds=job.retry_policy.retry_backoff_seconds)
        ).isoformat()
        status = ScheduledIngestionJobStatus.ACTIVE
    else:
        retry_count = job.state.retry_count + 1
        next_run_at = next_cron_run(job.cron, job.timezone, after=base_time)
        status = ScheduledIngestionJobStatus.FAILED
    return status, job.state.model_copy(
        deep=True,
        update={
            "last_run_at": finished_at,
            "next_run_at": next_run_at,
            "last_status": ScheduledIngestionRunStatus.FAILED,
            "retry_count": retry_count,
            "current_run_id": None,
            "last_run_id": run.id,
            "last_report_run_id": run.report_run_id,
            "last_report_path": run.report_path,
            "diagnostics": run.diagnostics,
        },
    )


def next_state_for_skip(
    job: ScheduledIngestionJob,
    run: ScheduledIngestionRun,
    *,
    now: datetime | None = None,
) -> ScheduledIngestionState:
    """Build the next job state after a skipped run."""
    finished_at = run.finished_at or _utc_now_iso()
    return job.state.model_copy(
        deep=True,
        update={
            "last_run_at": finished_at,
            "next_run_at": next_cron_run(
                job.cron,
                job.timezone,
                after=now or _parse_iso(finished_at),
            ),
            "last_status": ScheduledIngestionRunStatus.SKIPPED,
            "current_run_id": None,
            "last_run_id": run.id,
            "last_report_run_id": run.report_run_id,
            "last_report_path": run.report_path,
            "diagnostics": run.diagnostics,
        },
    )


def _parse_cron(cron: str) -> dict[str, set[int]]:
    normalized = cron.strip().lower()
    if normalized in CRON_ALIASES:
        cron = CRON_ALIASES[normalized]
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError("cron must have exactly five fields")
    minute, hour, day, month, weekday = parts
    return {
        "minute": _parse_cron_field(minute, 0, 59, "minute"),
        "hour": _parse_cron_field(hour, 0, 23, "hour"),
        "day": _parse_cron_field(day, 1, 31, "day"),
        "month": _parse_cron_field(month, 1, 12, "month"),
        "weekday": _parse_cron_field(weekday, 0, 7, "weekday", normalize_weekday=True),
    }


def _cron_fields_match(
    value: datetime,
    fields: dict[str, set[int]],
) -> bool:
    """Return whether a timestamp matches cron fields."""
    return (
        value.minute in fields["minute"]
        and value.hour in fields["hour"]
        and value.month in fields["month"]
        and _cron_day_matches(value, fields)
    )


def _cron_day_matches(
    value: datetime,
    fields: dict[str, set[int]],
) -> bool:
    """Return whether a timestamp matches day-of-month and weekday constraints."""
    cron_weekday = (value.weekday() + 1) % 7
    return value.day in fields["day"] and cron_weekday in fields["weekday"]


def _next_day_start(value: datetime, *, hour: int, minute: int) -> datetime:
    """Return the next day at the earliest allowed hour/minute."""
    return (value + timedelta(days=1)).replace(hour=hour, minute=minute)


def _next_allowed_month(
    value: datetime,
    month_values: list[int],
    *,
    hour: int,
    minute: int,
) -> datetime:
    """Return day one of the next allowed month at the earliest allowed time."""
    next_month = _next_value_at_or_after(value.month, month_values)
    if next_month is None:
        return value.replace(
            year=value.year + 1, month=month_values[0], day=1, hour=hour, minute=minute
        )
    return value.replace(month=next_month, day=1, hour=hour, minute=minute)


def _next_value_after(current: int, values: list[int]) -> int | None:
    """Return the next value in sorted `values` after `current`."""
    index = bisect_left(values, current + 1)
    if index < len(values):
        return values[index]
    return None


def _next_value_at_or_after(current: int, values: list[int]) -> int | None:
    """Return the next value in sorted `values` at or after `current`."""
    index = bisect_left(values, current)
    if index < len(values):
        return values[index]
    return None


def _parse_cron_field(
    field: str,
    minimum: int,
    maximum: int,
    label: str,
    *,
    normalize_weekday: bool = False,
) -> set[int]:
    values: set[int] = set()
    for part in field.split(","):
        values.update(_parse_cron_part(part.strip(), minimum, maximum, label))
    if not values:
        raise ValueError(f"cron {label} field must not be empty")
    if normalize_weekday and 7 in values:
        values.remove(7)
        values.add(0)
    return values


def _parse_cron_part(part: str, minimum: int, maximum: int, label: str) -> set[int]:
    if not part:
        raise ValueError(f"cron {label} field contains an empty entry")
    base = part
    step = 1
    if "/" in part:
        base, step_text = part.split("/", 1)
        step = int(step_text)
        if step <= 0:
            raise ValueError(f"cron {label} step must be positive")
    if base == "*":
        start, end = minimum, maximum
    elif "-" in base:
        start_text, end_text = base.split("-", 1)
        start, end = int(start_text), int(end_text)
    else:
        start = end = int(base)
    if start < minimum or end > maximum or start > end:
        raise ValueError(f"cron {label} field value out of range")
    return set(range(start, end + 1, step))


def _zoneinfo(timezone: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown schedule timezone: {timezone}") from exc


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
