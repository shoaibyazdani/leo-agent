"""SOUL.md / USER.md / MEMORY.md persona loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from leo.core.app import LeoApp


def _blocks(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("LEO_HOME", str(root))
    return LeoApp.__new__(LeoApp)._identity_blocks()


def test_identity_blocks_follow_declared_order_and_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "SOUL.md").write_text("s" * 9_000, encoding="utf-8")
    (tmp_path / "USER.md").write_text("u" * 2_000, encoding="utf-8")
    (tmp_path / "MEMORY.md").write_text("m" * 2_200, encoding="utf-8")

    blocks = _blocks(tmp_path, monkeypatch)

    labels = [
        "CHARACTER (SOUL.md)",
        "ABOUT THE OPERATOR (USER.md)",
        "DURABLE FACTS (MEMORY.md)",
    ]
    positions = [blocks.index(label) for label in labels]
    assert positions == sorted(positions)

    # Declared character caps: 8000 / 1375 / 2200.
    assert "s" * 8_000 in blocks
    assert "s" * 8_001 not in blocks
    assert "u" * 1_375 in blocks
    assert "u" * 1_376 not in blocks
    assert "m" * 2_200 in blocks
    assert blocks.count("[truncated]") == 2


def test_missing_identity_files_are_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "SOUL.md").write_text("only the character file", encoding="utf-8")

    blocks = _blocks(tmp_path, monkeypatch)

    assert "CHARACTER (SOUL.md)" in blocks
    assert "ABOUT THE OPERATOR (USER.md)" not in blocks
    assert "DURABLE FACTS (MEMORY.md)" not in blocks


def test_reference_files_are_not_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTS.md and IDENTITY.md are reference material, not prompt content."""
    (tmp_path / "AGENTS.md").write_text("workspace rules", encoding="utf-8")
    (tmp_path / "IDENTITY.md").write_text("identity card", encoding="utf-8")

    assert _blocks(tmp_path, monkeypatch) == ""


def test_identity_dir_prefers_leo_home_then_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEO_HOME", str(tmp_path / "persona"))
    assert LeoApp._identity_dir() == tmp_path / "persona"

    monkeypatch.delenv("LEO_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert LeoApp._identity_dir() == tmp_path / "home" / ".config" / "leo"
