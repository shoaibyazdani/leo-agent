# Running LEO as a service

`leo serve` is one process: API, dashboard, channel adapters, inbound polling and
the outbound queue all live inside it. It ships no supervisor of its own, so
restarting it needs a service manager.

## Install the unit

```sh
sudo cp deploy/leo-agent@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now leo-agent@$USER.service
```

The unit is a template: `%i` is the operator account, so the config, database and
media stay under that user's home and the unit can be copied verbatim to another
machine.

Secrets go in `~/.config/leo/env` (mode 600), never in the unit:

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USER_IDS=123456789
LEO_OPENCODE_SESSION=...
```

## Restarting

```sh
leo service status     # state, pid, restart count, start time
leo service restart    # the same restart, without remembering the unit name
leo service logs 100   # journal tail
```

Or from a chat surface, which is delivery-safe: the restart waits until the
reply leaves.

```
/restart
```

`/restart` is not an in-process reload. A process cannot replace itself in
place, so the command spawns a detached child that sleeps briefly and then calls
`systemctl restart` — the reply is delivered first, and the restart survives this
process dying.

Set these to match your installation:

| variable | default | meaning |
| --- | --- | --- |
| `LEO_SERVICE_UNIT` | `leo-agent.service` | unit `leo service` and `/restart` drive |
| `LEO_RESTART_DELAY_SECONDS` | `3` | how long the child waits before restarting |

## Config is read once, at startup

`LEO_CONFIG` (default `~/.config/leo/leo.yaml`) is loaded when the process
starts, not per session and not per turn. A new conversation changes nothing and
a restart re-reads the same file.

If your checkout ships an updated `leo.yaml`, copying it blindly can wipe runtime
settings that must not be published (approval toggles, channel allow-lists). Keep
the private copy as the source of truth and merge only the model block, or point
`LEO_CONFIG` straight at the file you want — but never assume a restart alone
picks up a repo-side change.
