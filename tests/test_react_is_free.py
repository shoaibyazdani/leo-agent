"""A reaction must never ask a human for permission.

Two gates used to sit in front of it: the tool was already LOW risk, but a FAILED
reaction triggered the self-healing engine, which proposed a HIGH-risk repair and
opened an approval. That is how "react to my message" turned into an approval
request for work nobody asked for.
"""

from __future__ import annotations

import pytest

from leo.events import Event
from leo.selfheal.engine import COSMETIC_TOOLS


def test_reaction_actions_are_declared_cosmetic() -> None:
    assert "channel_react" in COSMETIC_TOOLS
    assert "gateway.react" in COSMETIC_TOOLS


@pytest.mark.asyncio
async def test_a_failed_reaction_is_not_an_incident(monkeypatch, tmp_path) -> None:
    from leo import db
    from leo.selfheal.engine import SelfHealingEngine

    db.set_db_path(tmp_path / "sh.db")
    await db.initialize()

    engine = SelfHealingEngine.__new__(SelfHealingEngine)          # no wiring needed
    import hashlib as _h

    engine._load = lambda name: _async_none()                # type: ignore[attr-defined]
    engine._save = lambda name, value: _async_none()          # type: ignore[attr-defined]
    engine._index = lambda: _async_empty_dict()               # type: ignore[attr-defined]
    engine._emit = lambda topic, **payload: _async_none()     # type: ignore[attr-defined]

    event = Event(topic="tool.executed", source="tools", payload={
        "tool": "channel_react", "decision": "failure",
        "error": "provider rejected the reaction",
    })

    assert await engine.observe(event) is None


async def _async_none():
    return None


async def _async_empty_dict():
    return {}
