"""Desktop shell static verification (GUI launch is environment-dependent)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"


def test_desktop_shell_security_configuration() -> None:
    package = json.loads((DESKTOP / "package.json").read_text(encoding="utf-8"))
    assert package["main"] == "main.js"
    main = (DESKTOP / "main.js").read_text(encoding="utf-8")
    preload = (DESKTOP / "preload.js").read_text(encoding="utf-8")
    assert "nodeIntegration: false" in main
    assert "contextIsolation: true" in main
    assert "sandbox: true" in main
    assert "127.0.0.1" in main
    assert "contextBridge.exposeInMainWorld" in preload


def test_desktop_javascript_syntax() -> None:
    for name in ("main.js", "preload.js"):
        result = subprocess.run(["node", "--check", str(DESKTOP / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
