"""Static provider registry for OpenAI-compatible ingestion shortcuts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class ProviderStructuredOutputMode(StrEnum):
    """Structured output strategy used by an OpenAI-compatible provider."""

    AGENT_SCHEMA = "agent_schema"
    JSON_OBJECT = "json_object"


@dataclass(frozen=True)
class ProviderRegistryEntry:
    """Known OpenAI-compatible provider settings for CLI shortcuts."""

    name: str
    base_url: str
    api_key_env: str
    default_model: str | None
    docs_url: str
    notes: str
    tracing_disabled: bool = True
    structured_output_mode: ProviderStructuredOutputMode = ProviderStructuredOutputMode.AGENT_SCHEMA
    supported_models: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    extra_body: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "structured_output_mode",
            ProviderStructuredOutputMode(self.structured_output_mode),
        )
        object.__setattr__(self, "supported_models", tuple(self.supported_models))
        object.__setattr__(self, "extra_body", MappingProxyType(dict(self.extra_body)))


PROVIDER_REGISTRY: Mapping[str, ProviderRegistryEntry] = MappingProxyType(
    {
        "deepseek": ProviderRegistryEntry(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            default_model="deepseek-v4-pro",
            docs_url="https://api-docs.deepseek.com/zh-cn/",
            notes=(
                "DeepSeek V4 Pro uses JSON Object mode; thinking is enabled with high "
                "reasoning effort."
            ),
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            supported_models=("deepseek-v4-pro", "deepseek-v4-flash"),
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        "kimi": ProviderRegistryEntry(
            name="kimi",
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            default_model="kimi-k2.6",
            docs_url="https://platform.kimi.com/docs/api/overview",
            notes=("Kimi K2.6 uses JSON Object mode; thinking is enabled by default."),
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            extra_body={"thinking": {"type": "enabled"}},
        ),
        "openrouter": ProviderRegistryEntry(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            default_model=None,
            docs_url="https://openrouter.ai/docs/quickstart",
            notes=(
                "OpenRouter requires an explicit model; optional attribution headers are supported."
            ),
        ),
    }
)


def get_provider_entry(name: str) -> ProviderRegistryEntry:
    """Return a provider registry entry by name."""
    normalized = name.strip().lower()
    try:
        return PROVIDER_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(
            f"unknown provider '{name}'; supported providers: {supported}; "
            "use --list-providers to see details"
        ) from exc


def list_provider_entries() -> list[ProviderRegistryEntry]:
    """Return provider registry entries in stable display order."""
    return [PROVIDER_REGISTRY[name] for name in sorted(PROVIDER_REGISTRY)]
