#!/usr/bin/env bash
# PostToolUse hook: format and auto-fix a Python file Claude just edited.
# Never blocks the session: always exits 0. Requires ruff as a dev dependency (uv add --dev ruff).
set -u
FILE=$(python3 -c 'import json,sys
try:
    print((json.load(sys.stdin).get("tool_input") or {}).get("file_path") or "")
except Exception:
    print("")')
case "$FILE" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$FILE" ] || exit 0
command -v uv >/dev/null 2>&1 || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
uv run --frozen ruff format --quiet "$FILE" >/dev/null 2>&1 || true
uv run --frozen ruff check --fix --quiet "$FILE" >/dev/null 2>&1 || true
exit 0
