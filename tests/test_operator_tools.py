"""Conversation-callable browser/computer tools: registration, schemas,
permission gating and path containment. No real browser or desktop action is
performed — operators are replaced with in-process stubs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo.config import Settings
from leo.contracts import ToolCall
from leo.events import EventBus
from leo.security import PermissionManagerImpl
from leo.tools import operator_tools
from leo.tools.executor import ToolExecutorImpl
from leo.tools.registry import ToolRegistry

BROWSER = {
    "browser_navigate", "browser_extract_text", "browser_screenshot",
    "browser_click", "browser_fill",
}
COMPUTER = {
    "computer_notify", "computer_screenshot",
    "computer_clipboard_read", "computer_clipboard_write",
}


def build_registry(include_computer: bool = True) -> tuple[ToolRegistry, dict]:
    registry = ToolRegistry()
    settings = Settings()
    permissions = PermissionManagerImpl(settings, EventBus())
    operator_tools.register_operator_tools(
        registry, settings, permissions, include_computer=include_computer
    )
    ctx = {
        "settings": settings,
        "permissions": permissions,
        "actor": "chat:test",
        "dry_run": False,
    }
    return registry, ctx


def test_registration_and_schemas() -> None:
    registry, _ = build_registry()
    names = set(registry.list_specs() and [s.name for s in registry.list_specs()])
    assert BROWSER <= names and COMPUTER <= names
    for spec in registry.list_specs():
        assert spec.input_schema["type"] == "object"
        assert "additionalProperties" in spec.input_schema
        if spec.name.endswith("navigate") or spec.name.endswith("click"):
            assert spec.risk.value == "medium"
    # opt-out (e.g. non-macOS) registers browser tools only
    browser_only, _ = build_registry(include_computer=False)
    assert {s.name for s in browser_only.list_specs()} == BROWSER


@pytest.mark.asyncio
async def test_navigate_and_extract_use_persistent_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, ctx = build_registry()

    class StubBrowser:
        urls: list[str] = []

        async def navigate(self, url: str) -> str:
            self.urls.append(url)
            return url + "#final"

        async def extract_text(self, selector: str = "body") -> str:
            return f"text:{selector}"

    stub = StubBrowser()
    monkeypatch.setattr(operator_tools, "_browser_operator", lambda ctx: stub)

    nav = registry.get("browser_navigate").handler
    ext = registry.get("browser_extract_text").handler
    out = await nav({"url": "https://example.com"}, ctx)
    assert out == {"success": True, "result": "https://example.com#final"}
    assert stub.urls == ["https://example.com"]
    assert await ext({"selector": "main"}, ctx) == {"success": True, "result": "text:main"}


@pytest.mark.asyncio
async def test_computer_actions_use_executor_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, ctx = build_registry()
    seen: list[bool] = []

    class StubComputer:
        def __init__(self, permissions, *, actor="computer", dry_run=True, **kwargs):
            seen.append(dry_run)

        async def notify(self, message: str, *, title: str = "LEO") -> dict:
            return {"success": True, "message": message, "title": title, "executed": True}

        async def clipboard_write(self, text: str) -> dict:
            return {"success": True, "length": len(text)}

    monkeypatch.setattr(operator_tools, "_computer_operator",
                        lambda ctx: StubComputer(ctx["permissions"], dry_run=ctx["dry_run"]))
    notify = registry.get("computer_notify").handler
    write = registry.get("computer_clipboard_write").handler
    assert (await notify({"message": "e2e"}, ctx))["message"] == "e2e"
    assert (await write({"text": "hello"}, ctx))["length"] == 5
    assert seen == [False, False]


@pytest.mark.asyncio
async def test_browser_private_url_is_denied_by_policy() -> None:
    """The real operator rejects private/loopback hosts before any backend use."""
    from leo.integrations.browser import BrowserPermissionError, BrowserOperator

    settings = Settings()
    operator = BrowserOperator(
        PermissionManagerImpl(settings, EventBus()),
        base_dir=settings.root,
        profile_dir=settings.resolve(settings.paths.workspaces) / "browser-profile",
    )
    with pytest.raises(BrowserPermissionError):
        await operator.navigate("http://127.0.0.1:8891/health")
    with pytest.raises(BrowserPermissionError):
        await operator.navigate("file:///etc/passwd")


def test_screenshot_path_containment() -> None:
    registry, ctx = build_registry()
    shot = registry.get("browser_screenshot").handler
    # The handler resolves the target through _resolve_output_path before the
    # operator sees it; an escaping path must be rejected.
    import asyncio

    with pytest.raises(ValueError, match="workspace"):
        asyncio.run(shot({"path": "../../etc/evil.png"}, ctx))


# --------------------------------------------------------------------------- dry-run wiring


def test_computer_specs_opt_in_to_real_dry_run() -> None:
    registry, _ = build_registry()
    computer = {s.name: s for s in registry.list_specs() if s.category == "computer"}
    assert set(computer) == COMPUTER
    assert all(spec.supports_dry_run for spec in computer.values())
    # Browser actions cannot be simulated safely; they keep the executor's
    # short-circuit rather than being dispatched with a dry-run context.
    browser = [s for s in registry.list_specs() if s.category == "browser"]
    assert all(not s.supports_dry_run for s in browser)


def test_production_computer_operator_is_dry_run_without_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[bool] = []

    class CaptureOperator:
        def __init__(self, permissions, **kwargs):
            captured.append(kwargs["dry_run"])

    monkeypatch.setattr(operator_tools, "ComputerOperator", CaptureOperator)
    registry, ctx = build_registry()
    assert registry.get("computer_notify")

    operator_tools._computer_operator(ctx)
    ctx["dry_run"] = True
    operator_tools._computer_operator(ctx)
    ctx["dry_run"] = False
    ctx["settings"] = SimpleNamespace(computer=SimpleNamespace(live_actions=True))
    operator_tools._computer_operator(ctx)
    ctx["dry_run"] = True
    operator_tools._computer_operator(ctx)
    assert captured == [True, True, False, True]


@pytest.mark.asyncio
async def test_dry_run_preview_is_not_relabeled_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, ctx = build_registry()

    class StubComputer:
        def __init__(self, permissions, **kwargs):
            pass

        async def notify(self, message, *, title="LEO"):
            return {"action": "computer.notify", "dry_run": True, "executed": False}

    monkeypatch.setattr(operator_tools, "_computer_operator", lambda ctx: StubComputer(ctx["permissions"]))
    out = await registry.get("computer_notify").handler({"message": "x"}, ctx)
    assert out["executed"] is False and out["success"] is False


@pytest.mark.asyncio
async def test_executor_dispatches_requested_dry_run_to_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, ctx = build_registry()
    seen: list[bool] = []

    class StubComputer:
        def __init__(self, permissions, *, dry_run=True, **kwargs):
            seen.append(dry_run)
            self.dry_run = dry_run

        async def notify(self, message, *, title="LEO"):
            return {
                "action": "computer.notify", "dry_run": self.dry_run,
                "executed": False, "message": message,
            }

    monkeypatch.setattr(operator_tools, "_computer_operator", lambda c: StubComputer(c["permissions"], dry_run=c["dry_run"]))
    executor = ToolExecutorImpl(registry, ctx["permissions"], EventBus(), ctx["settings"])
    result = await executor.execute(
        ToolCall(name="computer_notify", arguments={"message": "preview"}, actor="chat:test", dry_run=True)
    )
    assert seen == [True]  # handler actually ran instead of a canned success
    assert result.metadata["dry_run"] is True
    assert result.output["executed"] is False and result.output["success"] is False
    # The executor's own success means "handler completed without exception";
    # the tool payload carries the truthful no-execution result.


@pytest.mark.asyncio
async def test_navigate_registered_preview_uses_token_bound_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser tool grants loopback only after the preview registry matches."""
    from leo.tools import preview_tools

    registry, ctx = build_registry()
    token = "registeredToken123"
    url = f"http://127.0.0.1:18765/preview/{token}"
    fake_preview = SimpleNamespace(
        token=token, port=18765, actor="chat:test", path="/workspace/page.html",
        content_type="text/html; charset=utf-8",
    )
    monkeypatch.setattr(preview_tools, "registered_preview", lambda value, c: fake_preview)

    seen: list[dict] = []

    class SpyOperator:
        async def navigate(self, url):  # must not be used for a registered preview
            raise AssertionError("registered preview must not use the ordinary path")

        async def navigate_registered_preview(self, url, *, token, port):
            seen.append({"url": url, "token": token, "port": port})
            return url + "#final"

    monkeypatch.setattr(operator_tools, "_browser_operator", lambda c: SpyOperator())
    out = await registry.get("browser_navigate").handler({"url": url}, ctx)
    assert out["success"] is True
    assert out["token_id"] == preview_tools.token_audit_id(token)
    assert out["token_id"] != token and out["registered_token"] is True
    assert seen == [{"url": url, "token": token, "port": 18765}]

    # Without a registry match the same loopback URL stays on the ordinary,
    # SSRF-blocked path (the real operator's denial is covered above and in
    # test_browser_private_url_is_denied_by_policy).
    monkeypatch.setattr(preview_tools, "registered_preview", lambda value, c: None)
    ordinary: list[str] = []

    class OrdinaryOperator:
        async def navigate(self, url):
            ordinary.append(url)
            return "ordinary"

    monkeypatch.setattr(operator_tools, "_browser_operator", lambda c: OrdinaryOperator())
    plain = await registry.get("browser_navigate").handler({"url": url}, ctx)
    assert plain == {"success": True, "result": "ordinary"}
    assert ordinary == [url]
