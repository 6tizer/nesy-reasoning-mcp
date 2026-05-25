"""Shared URL/domain safety helpers for ingestion retrieval layers."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse


def normalize_domain_filters(values: list[str]) -> list[str]:
    """Normalize exact host/subdomain filter values with stable de-duplication."""
    domains: list[str] = []
    seen: set[str] = set()
    for value in values:
        domain = normalize_domain_filter(value)
        if domain in seen:
            continue
        domains.append(domain)
        seen.add(domain)
    return domains


def normalize_domain_filter(value: str) -> str:
    """Normalize one domain filter and reject local/private literal hosts."""
    stripped = value.strip().lower().rstrip(".")
    if not stripped:
        raise ValueError("domain filters must not contain empty values")
    parsed = urlparse(stripped if "://" in stripped else f"//{stripped}")
    domain = (parsed.hostname or "").strip().lower().rstrip(".")
    if not domain:
        raise ValueError(f"invalid domain filter: {value}")
    if is_local_literal(domain):
        raise ValueError("local domain filters are not supported")
    return domain


def host_matches_domain(host: str, domain: str) -> bool:
    """Return whether host exactly matches domain or is its subdomain."""
    return host == domain or host.endswith(f".{domain}")


def is_local_literal(value: str) -> bool:
    """Return whether value is a local/private host literal."""
    if value == "localhost" or value.endswith(".localhost") or value.endswith(".local"):
        return True
    try:
        address = ip_address(value)
    except ValueError:
        return False
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    )
