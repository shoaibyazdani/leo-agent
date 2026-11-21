"""A channel has to follow ``/new``.

``/new`` appends a conversation for the same (channel, subject). The gateway's
get-or-create used to return the OLDEST match, so the channel stayed pinned to
the first conversation forever and ``/new`` produced an id nothing ever used.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from leo import db
from leo.integrations.gateway import Gateway


async def _seed(channel: str, subject: str, conversation_id: str, when: datetime) -> None:
    await db.execute(
        "INSERT INTO conversations(id,channel,subject,created_at,updated_at) VALUES(?,?,?,?,?)",
        (conversation_id, channel, subject, when.isoformat(), when.isoformat()),
    )


async def test_channel_follows_the_newest_conversation():
    now = datetime(2026, 11, 21, 7, 0, tzinfo=timezone.utc)
    await _seed("telegram", "100000001", "conv_first", now)
    await _seed("telegram", "100000001", "conv_after_new", now + timedelta(seconds=30))

    async with db.transaction() as conn:
        resolved = await Gateway._conversation_in(conn, "telegram", "100000001")

    assert resolved == "conv_after_new"


async def test_unrelated_subject_still_gets_its_own_conversation():
    now = datetime(2026, 11, 21, 7, 0, tzinfo=timezone.utc)
    await _seed("telegram", "100000001", "conv_owner", now)

    async with db.transaction() as conn:
        created = await Gateway._conversation_in(conn, "telegram", "999")

    assert created != "conv_owner"
