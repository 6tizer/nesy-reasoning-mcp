"""Static provider registry for OpenAI-compatible ingestion shortcuts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


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


PROVIDER_REGISTRY: Mapping[str, ProviderRegistryEntry] = MappingProxyType(
    {
        "deepseek": ProviderRegistryEntry(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            default_model="deepseek-v4-pro",
            docs_url="https://api-docs.deepseek.com/zh-cn/",
            notes="DeepSeek OpenAI-compatible Chat Completions endpoint.",
        ),
        "kimi": ProviderRegistryEntry(
            name="kimi",
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            default_model="kimi-k2.6",
            docs_url="https://platform.kimi.com/docs/api/overview",
            notes="Moonshot Kimi OpenAI-compatible Chat Completions endpoint.",
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
    """Return a provider registry entry by lowercase exact name."""
    try:
        return PROVIDER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown provider '{name}'; use --list-providers to see supported providers"
        ) from exc


def list_provider_entries() -> list[ProviderRegistryEntry]:
    """Return provider registry entries in stable display order."""
    return [PROVIDER_REGISTRY[name] for name in sorted(PROVIDER_REGISTRY)]
