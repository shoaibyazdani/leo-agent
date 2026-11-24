"""Tests for the operator-facing WhatsApp device-link wiring.

No Node process, WhatsApp socket or external network is used. The manager is
replaced with an in-memory fake; the page test uses only a loopback socket
bound to 127.0.0.1.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from leo.integrations import whatsapp_link
from leo.integrations.whatsapp_link import (
    LinkPage,
    LinkUnavailable,
    bridge_status,
    run_device_link,
)


def test_bridge_status_reports_missing_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(whatsapp_link.shutil, "which", lambda _name: "/usr/bin/node")
    monkeypatch.setattr(whatsapp_link, "BRIDGE_DIR", tmp_path)

    report = bridge_status("owner")

    assert report["node"] == "/usr/bin/node"
    assert report["dependencies_installed"] is False
    assert report["ready"] is False
    assert "npm ci" in report["hint"]


def test_ensure_ready_rejects_a_host_without_node(monkeypatch):
    monkeypatch.setattr(whatsapp_link.shutil, "which", lambda _name: None)

    with pytest.raises(LinkUnavailable):
        whatsapp_link.ensure_ready()


def test_link_page_serves_html_and_state_only_on_its_token_path():
    page = LinkPage("owner")
    url = page.start()
    try:
        page.update({
            "state": "waiting_for_scan",
            "linked": False,
            "error": None,
            "user": None,
            "qr": {"data_uri": "data:image/png;base64,AAAA", "expires_in_ms": 30_000},
        })

        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8")
            assert "Link WhatsApp" in body
            assert response.headers["Cache-Control"] == "no-store"

        with urllib.request.urlopen(url + "state", timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["state"] == "waiting_for_scan"
        assert payload["qr"].startswith("data:image/png;base64,")
        assert payload["expires_in_ms"] == 30_000

        wrong = url.replace(page.path, "/not-the-token/")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(wrong, timeout=5)
        assert excinfo.value.code == 404

        parsed = urllib.parse.urlsplit(url)
        conn = http.client.HTTPConnection("127.0.0.1", parsed.port, timeout=5)
        try:
            conn.request("GET", parsed.path, headers={"Host": "evil.example"})
            assert conn.getresponse().status == 421
        finally:
            conn.close()
    finally:
        page.close()


class _FakePage:
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.path = "/fake/"
        self.updates: list[dict] = []
        self.closed = False

    def start(self) -> str:
        return "http://127.0.0.1:1/fake/"

    def update(self, status: dict) -> None:
        self.updates.append(status)

    def close(self) -> None:
        self.closed = True


class _FakeManager:
    instances: list["_FakeManager"] = []

    def __init__(self, config) -> None:
        self.config = config
        self._auth = "a" * 32
        self.stopped = False
        self.session_dir = None
        self._states = [
            {
                "state": "waiting_for_scan",
                "linked": False,
                "error": None,
                "user": None,
                "qr": {"data_uri": "data:image/png;base64,AAAA", "expires_in_ms": 30_000},
            },
            {
                "state": "connected",
                "linked": True,
                "error": None,
                "user": {"id": "15551234567@s.whatsapp.net", "name": "Owner"},
                "qr": None,
            },
        ]
        _FakeManager.instances.append(self)

    @property
    def session_auth(self) -> str:
        return self._auth

    async def start(self, owner_id=None, *, auth=None):
        return {"state": "starting", "linked": False, "error": None, "user": None, "qr": None}

    async def status(self, owner_id=None, *, auth=None):
        if self._states:
            return self._states.pop(0)
        return {
            "state": "connected",
            "linked": True,
            "error": None,
            "user": {"id": "15551234567@s.whatsapp.net", "name": "Owner"},
            "qr": None,
        }

    async def cancel(self, owner_id=None, *, auth=None):
        return {"state": "cancelled", "linked": False}

    async def stop(self, owner_id=None, *, auth=None):
        self.stopped = True

    def diagnostics(self):
        return ()


async def test_run_device_link_completes_on_connected(monkeypatch):
    _FakeManager.instances.clear()
    monkeypatch.setattr(whatsapp_link, "ensure_ready", lambda: None)
    monkeypatch.setattr(whatsapp_link, "LinkPage", _FakePage)
    monkeypatch.setattr(whatsapp_link, "WhatsAppBridgeManager", _FakeManager)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(whatsapp_link.asyncio, "sleep", _no_sleep)

    seen: list[tuple[str, object]] = []
    result = await run_device_link(
        "owner", timeout=30, open_page=False, notify=lambda kind, data: seen.append((kind, data))
    )

    assert result.state == "connected"
    assert result.linked is True
    assert result.user == {"id": "15551234567@s.whatsapp.net", "name": "Owner"}
    manager = _FakeManager.instances[-1]
    assert manager.stopped is True
    assert ("page", "http://127.0.0.1:1/fake/") in seen
    assert result.as_dict()["ok"] is True
