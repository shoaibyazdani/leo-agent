# LEO — agent persona templates

This directory ships **empty templates only**. Personal persona files belong in
`$LEO_HOME` or `~/.config/leo`, never in this repository.

LEO loads these files in a fixed order, with these character caps:

| File | Prompt block | Cap |
| --- | --- | ---: |
| `SOUL.md` | CHARACTER | 8,000 |
| `USER.md` | ABOUT THE OPERATOR | 1,375 |
| `MEMORY.md` | DURABLE FACTS | 2,200 |

Copy the templates into the runtime location:

```sh
mkdir -p ~/.config/leo
cp agent/templates/SOUL.md \
   agent/templates/USER.md \
   agent/templates/MEMORY.md \
   ~/.config/leo/
```

Then edit those local copies. Do not commit them.

`AGENTS.md` and `IDENTITY.md` are optional reference files; LEO does not inject
them into the prompt. The loader is read-only: durable memory writes stay in
LEO's SQLite memory module, not back in `MEMORY.md`.
