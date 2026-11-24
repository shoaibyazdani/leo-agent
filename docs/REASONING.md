# Reasoning settings

`/reasoning` controls how much reasoning effort LEO requests from a model, which
configuration scope a change belongs to, and whether a concise reasoning summary
is shown in the reply. It is available in the CLI, API, dashboard and messaging
channels; Telegram renders it as an inline keyboard.

## Command

```text
/reasoning                         show the interactive menu
/reasoning <level>                 set the level in the current edit scope
/reasoning scope <global|session>  choose where later edits land
/reasoning global | session        shorthand for the scope command
/reasoning show | hide             show/hide concise summaries
/reasoning detail                  toggle concise/detailed summaries
/reasoning reset                   clear session overrides (or global with --global)
/reasoning --global <level>        one-shot global write
```

Aliases: `/think`, `/thinking`, `/t`.

The menu shows the active effort and where it came from, the edit scope, whether
display is on, which provider will receive the next request, and the last
translated level. Every level has a button. There are buttons for switching
scope, toggling summaries, toggling summary detail, resetting session overrides
and clearing the global default.

## Levels

From least to most effort:

`none · minimal · low · medium · high · xhigh · max · ultra`

- `none` is an explicit disable where the provider supports one.
- `ultra` is LEO's internal ceiling. No provider wire accepts it, so it is
  clamped to that provider's highest supported level (`max` on Anthropic,
  `xhigh`/`high` on narrower OpenAI-compatible wires).
- Clamping never escalates cost. An unsupported level resolves to the nearest
  *weaker* supported level; if nothing weaker exists, the provider's weakest
  level is used. The status line reports the requested and applied values.

Aliases accepted by the parser: `off`, `false`, `disable`, `min`, `med`, `hi`,
`xh`, `extreme`.

## Scopes

| Scope | Stored in | Applies to | Survives restart |
|---|---|---|---|
| global | `kv` key `reasoning:global` | every conversation without a session override, plus background work | yes |
| session | `kv` key `reasoning:session:<conversation_id>` | that conversation only | yes |

Resolution order is **session → global → `leo.yaml` default**. A missing session
value inherits; an explicit `hide` is not the same as unset. Session keys are
conversation ids, so one chat or user can never inherit another's override.

The edit scope itself is per conversation and is remembered with that
conversation's session state. A surface with no conversation id cannot write a
session value; it is told to use `--global` instead of inventing a shared key.

## Provider mapping

| Provider | Parameter sent |
|---|---|
| Anthropic-compatible, legacy (`reasoning_style: budget`) | `thinking: {type: enabled, budget_tokens: N}`; budgets rise from 1,024 to 32,768 and are always below `max_tokens` |
| Anthropic-compatible, modern Claude (`auto` detects Claude 4.6+ / 5.x) | `thinking: {type: adaptive, display: summarized\|omitted}` + `output_config.effort` |
| Anthropic-compatible, `none` | `thinking: {type: disabled}` where allowed |
| OpenAI-compatible | `reasoning_effort: <level>` |
| Provider with `reasoning_style: off` | nothing |

The style is configurable per provider:

```yaml
models:
  providers:
    my-gateway:
      kind: anthropic-compatible
      reasoning_style: auto        # auto | effort | budget | off
      reasoning_effort: medium     # optional provider default
      reasoning_efforts: [low, medium, high]  # optional narrow wire
      reasoning_overrides: {medium: high}     # optional vendor mapping
      reasoning_summaries: true    # allow provider summaries
```

`reasoning_overrides` is consulted before nearest-level clamping, for wires where
a level's positional meaning differs (for example a model whose middle tier is
`high`). It only applies when the mapped value is itself in the declared wire.

If a gateway rejects a reasoning field with HTTP 400/422, LEO retries that one
request without the reasoning parameters and records why in the usage metadata.
A rejected effort shape disables reasoning parameters for that provider for the
life of the process. A rejected `none` disable token is handled separately: LEO
retries at the lowest real effort (`low`) so the provider cannot substitute its
own higher default, and remembers that floor for later disabled requests. If the
floor is rejected too, reasoning parameters are dropped. The reply still arrives,
and the degradation is visible in `/reasoning` and `model_usage`.

## Display and privacy

Display is **off by default**. When on:

- Anthropic adaptive thinking is requested with `display: summarized`; the API
  returns a readable summary, not the raw chain of thought.
- OpenAI-compatible providers contribute only fields explicitly marked as a
  summary (`reasoning_summary`, a `reasoning.summary` object, or a
  summary-typed `reasoning_details` part). Generic `reasoning`,
  `reasoning_content` and `thinking` fields are treated as private and are
  never shown.
- Streaming surfaces (CLI/dashboard) show the summary in their dedicated
  thinking channel; answer text never includes reasoning.
- Blocking messaging replies (Telegram and other channel adapters) add a short
  footer: the effort and its source, the provider summary when one exists, the
  number of model calls and the tools used. `detail` controls whether the
  summary is shortened (~360 characters) or fuller (~1,200).
- With display off, no reasoning event is emitted and no footer is added.

Raw thinking is not persisted. Assistant rows store no reasoning text; only the
effective effort/source/display state is recorded as metadata.

## Subagents

A chat turn installs its effective selection in a per-turn context variable.
Agent workers inherit it unless the delegation context explicitly provides
`reasoning` (or `reasoning_effort`/`reasoning_show`). Scheduled and background
work outside a turn uses the provider/global default. A child can never write a
session override.

## Configuration defaults

```yaml
models:
  reasoning_effort: medium   # optional global default; unset sends nothing
  reasoning_show: false      # global default for summary display
```

Live `/reasoning` writes are stored in SQLite and take precedence over these
YAML defaults. The global value is pushed into live providers on startup and
immediately after a change, so scheduled work and compaction see it too.

## Limitations

- A provider that cannot disable thinking (some current Claude models) may still
  reason at its minimum level when `none` is selected; LEO reports the
  degradation instead of claiming the parameter was accepted.
- OpenAI-compatible summary display depends on the gateway returning an explicit
  summary field. Many gateways return raw chain-of-thought only; LEO then shows
  the execution footer and no thought text.
- Reasoning parameters can invalidate a provider's prompt cache when the level
  changes. That is expected and is not hidden.
- The provider's configured `timeout_seconds` is unchanged. A very slow `max`
  turn may need that value raised in `leo.yaml`.
- Session scope is keyed by conversation id. Starting a new conversation (`/new`)
  leaves the old conversation's override with the old conversation.
- Some OpenAI-compatible thinking endpoints (notably DeepSeek/Kimi/MiMo thinking
  modes) require the previous assistant turn's `reasoning_content` to be echoed
  on a replayed tool-call turn. LEO deliberately does not persist chain-of-thought,
  so such a provider may reject the replay; the provider-level reasoning fallback
  then retries without reasoning parameters and reports the degradation.
