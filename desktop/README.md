# LEO desktop shell (Electron)

A thin, localhost-only wrapper around the LEO FastAPI dashboard. The Python
backend owns all credentials, tools and filesystem access; the renderer has
`nodeIntegration: false`, `contextIsolation: true` and `sandbox: true`.

```bash
# From the repository root, after creating .venv:
cd desktop && npm install && npm start
```

Set `LEO_API_URL` to connect to an already running backend, or
`LEO_EXTERNAL_API=1` to skip spawning the local backend. Port 8765 on
`127.0.0.1` is used by default.

Status: source and configuration verified; GUI launch was not exercised in this
headless environment. The web dashboard itself is covered by `tests/test_api.py`.
