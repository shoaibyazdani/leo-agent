#!/usr/bin/env bash
# Install (or wrap) a local pre-commit secret scan. Run once per clone.
# Existing hooks are NEVER overwritten: they are copied to a timestamped backup
# and invoked by the wrapper before the secret scan.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
hook_dir="$root/.git/hooks"
hook="$hook_dir/pre-commit"
marker="leo-secret-scan-wrapper-v2"
backup_config="$hook_dir/.leo-pre-commit-backup"

if [[ ! -d "$hook_dir" ]]; then
  echo "No .git/hooks directory at $hook_dir (is this a worktree/clone?)" >&2
  exit 1
fi

if [[ -f "$hook" ]] && grep -q "$marker" "$hook"; then
  echo "Pre-commit secret scan wrapper already installed: $hook"
  exit 0
fi

backup=""
if [[ -e "$hook" || -L "$hook" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup="$hook.backup.$stamp"
  cp -p "$hook" "$backup"
  echo "Backed up existing pre-commit hook to: $backup"
fi

# Store the backup path in a sidecar file so the wrapper needs no interpolation
# of user-controlled paths into generated shell code.
printf '%s\n' "$backup" > "$backup_config"

tmp="$(mktemp "$hook_dir/.pre-commit.leo.XXXXXX")"
cat > "$tmp" <<'HOOK'
#!/usr/bin/env bash
# leo-secret-scan-wrapper-v2 — do not edit; run scripts/install-git-hooks.sh
set -euo pipefail
hook_dir="$(cd "$(dirname "$0")" && pwd)"
config="$hook_dir/.leo-pre-commit-backup"
if [[ -f "$config" ]]; then
  backup=""
  IFS= read -r backup < "$config" || backup=""
  if [[ -n "$backup" && -f "$backup" ]]; then
    "$backup" "$@"
  fi
fi
python3 "$(git rev-parse --show-toplevel)/scripts/check-secrets.py"
HOOK

chmod +x "$tmp"
mv "$tmp" "$hook"
echo "Installed pre-commit hook (existing hook preserved and wrapped): $hook"
