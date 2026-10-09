#!/usr/bin/env bash
# Reproducible Claude Code setup for this repository (macOS first; Linux works too).
#
# What it does (safe to re-run):
#   1. Checks machine prerequisites and prints how to install anything missing.
#      With --install-tools it installs missing CLIs via Homebrew / uv tool (machine-level, opt-in).
#   2. Syncs project Python dependencies with uv (--frozen).
#   3. Installs the project's approved plugins at PROJECT scope (never user scope).
#   4. Verifies that your user-level Claude Code settings file was not modified, and restores it if it was.
#   5. Validates the shared configuration (scripts/check-claude-config.py).
#
# It never writes to ~/.claude/settings.json, never edits .claude/settings.json, and never stores credentials.
set -euo pipefail

INSTALL_TOOLS=0
SKIP_PLUGINS=0
for arg in "$@"; do
  case "$arg" in
    --install-tools) INSTALL_TOOLS=1 ;;
    --skip-plugins) SKIP_PLUGINS=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "Run this inside the repository."; exit 1; }
cd "$ROOT"

# Approved plugins. Keep in sync with "enabledPlugins" in .claude/settings.json (CI checks this list).
PLUGINS=(
  "security-guidance@claude-plugins-official"
  "code-review@claude-plugins-official"
  "pyright-lsp@claude-plugins-official"
  "pydantic-ai@claude-plugins-official"
  "astronomer-data@astronomer"
  "frontend-design@claude-plugins-official"
  "impeccable@impeccable"
)

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; }

echo "== 1. Machine prerequisites"
missing_required=0
need() {  # name, check-command, install-hint, required(1/0), brew-formula-or-empty
  local name="$1" check="$2" hint="$3" required="$4" brew_pkg="$5"
  if eval "$check" >/dev/null 2>&1; then ok "$name"; return; fi
  if [[ $INSTALL_TOOLS -eq 1 && -n "$brew_pkg" ]] && command -v brew >/dev/null 2>&1; then
    echo "    installing $name ($brew_pkg) ..."; eval "$brew_pkg" && { ok "$name (installed)"; return; }
  fi
  if [[ "$required" == "1" ]]; then fail "$name missing: $hint"; missing_required=1; else warn "$name missing (recommended): $hint"; fi
}
need "git"            "command -v git"            "xcode-select --install"                                   1 ""
need "Claude Code"    "command -v claude"         "curl -fsSL https://claude.ai/install.sh | bash"            1 ""
need "uv"             "command -v uv"             "brew install uv"                                          1 "brew install uv"
need "python3 >= 3.11" "python3 -c 'import sys; assert sys.version_info >= (3, 11)'" "uv python install 3.12" 1 ""
need "GitHub CLI (gh)" "command -v gh"            "brew install gh"                                          1 "brew install gh"
need "Docker"         "command -v docker"         "brew install --cask docker"                               0 "brew install --cask docker"
need "gcloud CLI"     "command -v gcloud"         "brew install --cask google-cloud-sdk"                     0 "brew install --cask google-cloud-sdk"
need "Terraform"      "command -v terraform"      "brew tap hashicorp/tap && brew install hashicorp/tap/terraform" 0 "brew tap hashicorp/tap && brew install hashicorp/tap/terraform"
need "pyright-langserver (for pyright-lsp)" "command -v pyright-langserver" "uv tool install pyright"        0 "uv tool install pyright"
if [[ $missing_required -eq 1 ]]; then echo "Install the required tools above, then re-run."; exit 1; fi

echo "== 2. Project dependencies"
if [[ -f pyproject.toml && -f uv.lock ]]; then uv sync --frozen && ok "uv sync --frozen"; else warn "no pyproject.toml/uv.lock yet; skipped"; fi

USER_SETTINGS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json"
BACKUP=""
if [[ -f "$USER_SETTINGS" ]]; then
  BACKUP="$(mktemp)"; cp "$USER_SETTINGS" "$BACKUP"
  BEFORE="$(shasum -a 256 "$USER_SETTINGS" | cut -d' ' -f1)"
else
  BEFORE="absent"
fi

echo "== 3. Project-scoped plugins"
if [[ $SKIP_PLUGINS -eq 1 ]]; then
  warn "skipped (--skip-plugins)"
else
  markets="$(claude plugin marketplace list 2>/dev/null || true)"
  need_session=0
  grep -q "claude-plugins-official" <<<"$markets" || need_session=1
  grep -q "astronomer" <<<"$markets" || need_session=1
  grep -q "impeccable" <<<"$markets" || need_session=1
  if [[ $need_session -eq 1 ]]; then
    warn "Marketplaces not registered on this machine yet."
    echo "    Run 'claude' once in this repository, accept the folder trust prompt, then exit (/exit)."
    echo "    Claude Code registers the official marketplace and clones the ones declared in .claude/settings.json."
    echo "    Then re-run: scripts/setup-claude.sh"
    exit 1
  fi
  installed="$(claude plugin list 2>/dev/null || true)"
  for p in "${PLUGINS[@]}"; do
    if grep -qF "$p" <<<"$installed"; then
      ok "$p (already installed)"
    else
      claude plugin install "$p" --scope project && ok "$p (installed, project scope)" || fail "$p failed; see 'claude plugin list' and claude.setup.md"
    fi
  done
fi

echo "== 4. Global settings untouched"
if [[ -f "$USER_SETTINGS" ]]; then AFTER="$(shasum -a 256 "$USER_SETTINGS" | cut -d' ' -f1)"; else AFTER="absent"; fi
if [[ "$BEFORE" == "$AFTER" ]]; then
  ok "$USER_SETTINGS unchanged"
else
  if [[ -n "$BACKUP" ]]; then cp "$BACKUP" "$USER_SETTINGS"; warn "user settings changed during setup and were restored from backup"; else warn "a user settings file was created during setup; review $USER_SETTINGS"; fi
fi
if git diff --quiet -- .claude/settings.json; then ok ".claude/settings.json unchanged"; else warn ".claude/settings.json changed; review with 'git diff .claude/settings.json' and do not commit unintended edits"; fi
[[ -n "$BACKUP" ]] && rm -f "$BACKUP"

echo "== 5. Shared configuration"
chmod +x .claude/hooks/*.py .claude/hooks/*.sh 2>/dev/null || true
python3 scripts/check-claude-config.py

cat <<'NEXT'

Next (each member, once, on your own machine; nothing here is committed):
  claude            # sign in (/login) if prompted; trust this folder
  gh auth login     # GitHub access for gh and the code-review plugin
  gcloud auth login && gcloud auth application-default login   # only if you work on GCP resources
Then inside Claude Code: /context, /hooks, /agents, /permissions and type "/" to see the skills.
NEXT
