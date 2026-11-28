"""Live model catalog reads and `leo model list --all`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import yaml
from typer.testing import CliRunner

from leo.interfaces import cli as cli_mod
from leo.models import catalog as catalog_mod


class FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class FakeClient:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def get(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
        self.calls.append({"url": url, "headers": dict(headers or {})})
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_fetch_catalog_parses_ids_without_echoing_the_key() -> None:
    client = FakeClient(FakeResponse(200, {"data": [{"id": "m-2"}, {"id": "m-1"}, {"id": "m-2"}]}))
    result = catalog_mod.fetch_catalog(
        "https://example.invalid/v1/", api_key="unit-key-value", client=client
    )
    assert result.ok and result.models == ("m-2", "m-1")
    assert "unit-key-value" not in result.detail
    assert client.calls[0]["url"] == "https://example.invalid/v1/models"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer unit-key-value"
    assert not client.closed  # an injected client stays owned by its caller


def test_anthropic_catalog_uses_its_own_auth_and_path() -> None:
    client = FakeClient(FakeResponse(200, {"data": [{"id": "claude-test"}]}))
    result = catalog_mod.fetch_catalog(
        "https://example.invalid", api_key="anthropic-key",
        kind="anthropic-compatible", client=client,
    )
    assert result.models == ("claude-test",)
    assert client.calls[0]["url"] == "https://example.invalid/v1/models"
    assert "x-api-key" in client.calls[0]["headers"]
    assert "Authorization" not in client.calls[0]["headers"]


def test_authentication_failure_is_redacted_and_rejected() -> None:
    result = catalog_mod.fetch_catalog(
        "https://example.invalid/v1", api_key="bad-key",
        client=FakeClient(FakeResponse(401, {"error": "bad-key leaked in body"})),
    )
    assert not result.ok
    assert result.detail == "authentication rejected (HTTP 401)"
    assert "bad-key" not in result.detail


def test_rate_limit_retries_five_times_with_the_documented_backoff() -> None:
    client = FakeClient(*[FakeResponse(429, {})] * 5, FakeResponse(200, {"data": [{"id": "ok"}]}))
    waits: list[float] = []
    result = catalog_mod.fetch_catalog(
        "https://example.invalid/v1", api_key="k", client=client, sleep=waits.append
    )
    assert result.ok and result.models == ("ok",)
    assert waits == [30.0, 60.0, 120.0, 240.0, 240.0]
    assert len(client.calls) == 6


def test_local_provider_needs_no_key_for_a_live_probe() -> None:
    client = FakeClient(FakeResponse(200, {"data": [{"id": "llama-local"}]}))
    verdict = catalog_mod.probe_provider(
        "http://127.0.0.1:11434/v1", api_key=None, client=client
    )
    assert verdict.ok and verdict.models == ("llama-local",)
    assert "Authorization" not in client.calls[0]["headers"]


def test_unreachable_provider_is_reported_without_a_traceback() -> None:
    class Broken:
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError("offline")

        def close(self) -> None:
            pass

    result = catalog_mod.fetch_catalog(
        "https://example.invalid/v1", api_key="k", client=Broken()
    )
    assert not result.ok and "unreachable" in result.detail


def _config(tmp_path: Any) -> Any:
    path = tmp_path / "leo.yaml"
    path.write_text(yaml.safe_dump({
        "models": {
            "default": "or",
            "providers": {
                "or": {
                    "kind": "openai-compatible",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "UNIT_CATALOG_KEY",
                    "model": "configured-only",
                },
                "offline": {"kind": "mock", "model": "mock-1"},
            },
        }
    }))
    return path


def test_model_list_all_shows_live_catalog_not_only_configured_slots(
    tmp_path: Any, monkeypatch: Any
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("LEO_CONFIG", str(config))
    cli_mod._state.clear()
    monkeypatch.setattr(catalog_mod, "fetch_catalog", lambda *a, **k: catalog_mod.CatalogResult(
        True, ("live-a", "live-b"), "live catalog: 2 model(s)"
    ))
    result = CliRunner().invoke(cli_mod.cli, ["model", "list", "--all"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["all"] is True
    assert payload["models"] == ["live-a", "live-b"]
    assert payload["providers"]["or"]["source"] == "live"
    assert "configured-only" not in payload["models"]
    assert payload["providers"]["offline"]["source"] == "configured"


def test_model_list_without_all_still_lists_configured_slots(
    tmp_path: Any, monkeypatch: Any
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("LEO_CONFIG", str(config))
    cli_mod._state.clear()
    result = CliRunner().invoke(cli_mod.cli, ["model", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["or"]["model"] == "configured-only"
    assert "models" not in payload
