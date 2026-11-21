#!/usr/bin/env python3
"""Ingest local orchestration-agent transcripts into LEO's activity table.

Usage (from the repository root):

    .venv/bin/python scripts/agent_activity_ingest.py
    .venv/bin/python scripts/agent_activity_ingest.py --dir /tmp/transcripts
    .venv/bin/python scripts/agent_activity_ingest.py --interval 2

The script never contacts the network and never reads a transcript outside the
configured root. Re-running it is safe: rows are upserted from a fingerprint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leo import db  # noqa: E402
from leo.agents import activity  # noqa: E402
from leo.config import Settings  # noqa: E402


async def _run(interval: float | None) -> int:
    if db.database_path_configured():
        await db.initialize()
    else:
        await db.initialize(Settings.load().db_path)
    first = await activity.sync_agent_activity()
    print(json.dumps(first, sort_keys=True))
    if interval is None:
        return 0
    try:
        while True:
            await asyncio.sleep(interval)
            result = await activity.sync_agent_activity()
            print(json.dumps(result, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        dest="transcript_dir",
        help="Override LEO_AGENT_TRANSCRIPT_DIR (still strictly path-contained).",
    )
    parser.add_argument("--db", dest="db_path", help="Override the SQLite database path.")
    parser.add_argument(
        "--interval",
        type=float,
        help="Repeat ingestion every N seconds; omit for a single pass.",
    )
    args = parser.parse_args(argv)
    if args.transcript_dir:
        os.environ["LEO_AGENT_TRANSCRIPT_DIR"] = args.transcript_dir
    if args.interval is not None and args.interval < 0.5:
        parser.error("--interval must be >= 0.5 seconds")
    if args.db_path:
        db.set_db_path(args.db_path)
    return asyncio.run(_run(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
