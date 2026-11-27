"""Cheap live provider probes and model catalog reads.

This module is deliberately the only place that sends a credential to an
upstream endpoint for setup/diagnostics. Results carry status text and model
IDs only: no request body, response body or key value is ever returned or
printed. A rate-limited provider is retried with a fixed backoff instead of
turning a transient 429 into a setup failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import httpx

# Retry a 429 five times before giving up. The first four waits follow the
# documented schedule; the fifth keeps the ceiling rather than growing forever.
RATE_LIMIT_BACKOFF = (30.0, 60.0, 120.0, 240.0, 240.0)
MAX_MODELS = 1_000


@dataclass(frozen=True)
class CatalogResult:
    """Model IDs read from a provider, plus a redacted failure reason."""

    ok: bool
    models: tuple[str, ...] = ()
    detail: str = ""
    supported: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "models": list(self.models),
            "detail": self.detail,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class ProbeResult:
    """Result of a credential check; ``detail`` never contains the key."""

    ok: bool
    detail: str
    models: tuple[str, ...] = ()


def _catalog_url(base_url: str, kind: str) -> str:
    root = base_url.strip().rstrip("/")
    if kind == "anthropic-compatible" and not root.endswith("/v1"):
        return f"{root}/v1/models"
    return f"{root}/models"


def _auth_headers(kind: str, api_key: str | None, extra: dict[str, str] | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    for name, value in (extra or {}).items():
        # Only non-secret literals are accepted here; env-backed headers are
        # handled by the caller and never persisted in this module.
        if isinstance(name, str) and isinstance(value, str):
            headers[name] = value
    if api_key:
        if kind == "anthropic-compatible":
            headers["x-api-key"] = api_key
            headers.setdefault("anthropic-version", "2023-06-01")
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _model_ids(payload: Any) -> tuple[str, ...]:
    rows: Iterable[Any]
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("models") or []
    else:
        rows = payload or []
    ids: list[str] = []
    for row in rows:
        if isinstance(row, str):
            value = row
        elif isinstance(row, dict):
            value = row.get("id") or row.get("name") or ""
        else:
            continue
        text = str(value).strip()
        if text and text not in ids:
            ids.append(text)
        if len(ids) >= MAX_MODELS:
            break
    return tuple(ids)


def fetch_catalog(
    base_url: str,
    *,
    api_key: str | None = None,
    kind: str = "openai-compatible",
    timeout: float = 8.0,
    extra_headers: dict[str, str] | None = None,
    client: Any | None = None,
    sleep: Callable[[float], Any] | None = None,
) -> CatalogResult:
    """GET the provider's model list. ``client`` allows offline tests."""
    url = base_url.strip()
    if not url.startswith(("http://", "https://")):
        return CatalogResult(False, detail="base_url must start with http:// or https://")
    endpoint = _catalog_url(url, kind)
    headers = _auth_headers(kind, api_key, extra_headers)
    pause = sleep or time.sleep
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(len(RATE_LIMIT_BACKOFF) + 1):
            try:
                response = http.get(endpoint, headers=headers)
            except httpx.HTTPError as exc:
                return CatalogResult(
                    False, detail=f"provider unreachable ({type(exc).__name__})"
                )
            status = int(getattr(response, "status_code", 0))
            if status == 429:
                if attempt < len(RATE_LIMIT_BACKOFF):
                    pause(RATE_LIMIT_BACKOFF[attempt])
                    continue
                return CatalogResult(False, detail="provider rate limited after retries")
            if status == 401 or status == 403:
                return CatalogResult(
                    False, detail=f"authentication rejected (HTTP {status})", supported=True
                )
            if status < 200 or status >= 300:
                return CatalogResult(
                    True, detail=f"catalog endpoint unavailable (HTTP {status})",
                    supported=False,
                )
            try:
                payload = response.json()
            except (ValueError, TypeError):
                return CatalogResult(True, detail="catalog returned no JSON", supported=False)
            models = _model_ids(payload)
            return CatalogResult(
                True, models=models,
                detail=f"live catalog: {len(models)} model(s)" if models else "live catalog is empty",
            )
        return CatalogResult(False, detail="provider rate limited after retries")
    finally:
        if owns_client and hasattr(http, "close"):
            http.close()


def probe_provider(
    base_url: str,
    *,
    api_key: str | None = None,
    kind: str = "openai-compatible",
    skip: bool = False,
    timeout: float = 8.0,
    client: Any | None = None,
    sleep: Callable[[float], Any] | None = None,
) -> ProbeResult:
    """Validate a key with one cheap catalog GET; never send a chat request."""
    if skip:
        return ProbeResult(True, "live probe skipped by operator")
    if not base_url.strip() or kind == "mock":
        return ProbeResult(True, "offline provider: no live probe required")
    # Local servers may need no credential at all; a missing key on a remote
    # provider is caught by the wizard before this point and shows up here as
    # an authentication rejection from the provider itself.
    result = fetch_catalog(
        base_url, api_key=api_key, kind=kind, timeout=timeout,
        client=client, sleep=sleep,
    )
    if not result.ok:
        return ProbeResult(False, result.detail)
    if not result.supported:
        return ProbeResult(False, result.detail)
    return ProbeResult(True, result.detail, models=result.models)
