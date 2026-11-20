"""Optional live browser verification (Phase 10, DoD #12).

Run with `LEO_LIVE_BROWSER=1` after installing Playwright/Chromium. Normal test
runs skip this file so the suite stays offline and deterministic.
"""

from __future__ import annotations

import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from leo.contracts import Decision, RiskLevel
from leo.integrations.browser import BrowserOperator


class _Allow:
    async def authorize(self, actor, action, resource=None, risk=RiskLevel.LOW, **context):
        return Decision(allowed=True, reason="test allow", risk=risk)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib API
        body = b"<html><body><h1 id='title'>LEO browser OK</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test server
        pass


@pytest.mark.skipif(os.getenv("LEO_LIVE_BROWSER") != "1", reason="set LEO_LIVE_BROWSER=1 for live browser check")
@pytest.mark.asyncio
async def test_live_browser_navigates_extracts_and_screenshots(tmp_path: Path) -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    operator = BrowserOperator(
        _Allow(),
        profile_dir=tmp_path / "profile",
        base_dir=tmp_path,
        allow_private_hosts=True,
        timeout=15.0,
    )
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        assert (await operator.navigate(url)).startswith("http://127.0.0.1")
        text = await operator.extract_text("#title")
        assert "LEO browser OK" in text
        shot = await operator.screenshot()
        assert shot.startswith(b"\x89PNG")
        await operator.click("#title")
        assert operator.record().status in {"operational", "configured", "active", "ok"}
    finally:
        await operator.close()
        server.shutdown()
        thread.join(timeout=5)
