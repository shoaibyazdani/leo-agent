"""Static validation of the installers and README Quick Install section."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL_INSTALLER = (ROOT / "install.sh").read_text(encoding="utf-8")
PS_INSTALLER = (ROOT / "install.ps1").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
LEGACY = re.compile(r"openclaw|hermes", re.IGNORECASE)


def test_shell_installer_parses_and_installs_then_onboards() -> None:
    if shutil.which("sh"):
        check = subprocess.run(
            ["sh", "-n", str(ROOT / "install.sh")], capture_output=True, text=True
        )
        assert check.returncode == 0, check.stderr
    for marker in (
        'uname -s', 'uv pip install', '.venv/bin/python -m pip install', 'cli init',
        'onboard "$@"', 'cd "\\$repo"', 'LEO_SKIP_ONBOARD',
    ):
        assert marker in SHELL_INSTALLER, marker
    assert not re.search(r"curl[^\n]*\|\s*(sh|bash)", SHELL_INSTALLER)
    assert "iex" not in SHELL_INSTALLER.lower()


def test_powershell_installer_installs_and_onboards_without_remote_code() -> None:
    for marker in (
        'Windows_NT', 'pyproject.toml', '-m venv', '-m pip install -e .',
        'cli onboard', 'apply_env_file', 'SetEnvironmentVariable', 'SkipOnboard',
    ):
        assert marker in PS_INSTALLER, marker
    for forbidden in ("Invoke-Expression", "Invoke-WebRequest", "iwr", "curl "):
        assert forbidden.lower() not in PS_INSTALLER.lower(), forbidden
    assert PS_INSTALLER.count("@'") == 1 and PS_INSTALLER.count("\n'@") == 1  # balanced here-string
    assert not LEGACY.search(PS_INSTALLER)


def test_readme_documents_both_one_command_installers() -> None:
    section = README.split("## Quick Install", 1)[1].split("## ", 1)[0]
    assert "install.sh | sh" in section
    assert "install.ps1 | iex" in section
    assert "leo onboard" in section
    assert "leo.yaml" in section and "0600" in section
    assert not LEGACY.search(section)
