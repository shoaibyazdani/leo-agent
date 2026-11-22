# LEO Status

Current, code-backed state of the project. History lives in git, not in a
narrative changelog.

## Working today

- Provider-independent agent platform: streaming chat in the dashboard and in
  the terminal (`leo chat`), durable tasks, layered memory, specialist agents,
  native tools and skills, approvals with audit, integrations adapters.
- Measured from the live `LeoApp.initialize()` composition root: 95
  registered tools, 37 available slash commands (126 registered slash
  commands in total) and 24 specialist agents.
- Vision: image attachments route to providers that declare the `vision`
  capability and reach the model as image content blocks.
- Local-first security model: loopback bind by default, `LEO_API_TOKEN`
  required for any non-loopback bind, HIGH-risk actions gated by approvals.

## Limits

- Live messaging delivery needs platform credentials; the adapters are
  mock-tested until then.
- Remote node / VPS deployment needs a host, token and TLS endpoint.
- See [LEO_KNOWN_ISSUES.md](LEO_KNOWN_ISSUES.md) for the itemized list.

## Testing

See [docs/TESTING.md](docs/TESTING.md).
