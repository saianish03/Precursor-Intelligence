# Claude Code setup for Precursor Intelligence

The shared Claude Code configuration is committed in this repo, so everyone works with the same project
instructions, skills, subagents, rules, hooks, permissions and plugins. You clone the repo, run one script,
and you're set up. Your own preferences and credentials stay on your laptop and are git-ignored.

Time needed: about 15 minutes. Written for macOS; Linux works with the equivalent package manager.

---

## What's in the repo

| Piece | Where | What it does |
|---|---|---|
| Project instructions | `CLAUDE.md` | Always-on rules: uv only, point-in-time data, CCN as text, no secrets |
| 6 skills | `.claude/skills/` | `cms-data-domain`, `ml-engineering`, `agent-development`, `mlops-deployment`, `testing-evaluation` load automatically when relevant; you run `/pr-ready` yourself |
| 3 subagents | `.claude/agents/` | Read-only reviewers: `leakage-reviewer`, `agent-reviewer`, `infra-reviewer` |
| 5 rules | `.claude/rules/` | Short conventions that load only when you work in matching folders (`src/precursorintelligence/...`, `dags/`, `infra/`, `backend/`, ...) |
| 3 hooks | `.claude/hooks/` | Block writes of secrets and `.env` files; run `ruff` on edited Python files; remind Claude of the key rules after compaction |
| Permissions | `.claude/settings.json` | Tests, lint and git reads are allowed; `git push`, `gcloud`, `bq` and `uv add` ask first; `terraform apply/destroy`, force push, `git reset --hard`, `pip install` and reading `.env` or keys are denied |
| No attribution | `.claude/settings.json` | Claude adds no `Co-Authored-By` trailer to commits and no "Generated with" line to PRs |
| 7 plugins | enabled in `.claude/settings.json`; installed by the setup script | `security-guidance`, `code-review`, `pyright-lsp`, `pydantic-ai` (Pydantic AI framework skills), `frontend-design`, `astronomer-data` (Airflow skills), `impeccable` (UI design) |
| GitHub MCP server (optional) | `.mcp.json` | Read-only access to issues and PRs, using your own token |
| Scripts | `scripts/setup-claude.sh`, `scripts/check-claude-config.py` | Install the plugins at project scope; validate the config (CI runs the validator too) |

---

## Step 0. Install the tools (once per laptop)

```bash
curl -fsSL https://claude.ai/install.sh | bash     # Claude Code (or: brew install --cask claude-code)
brew install uv gh libomp                          # libomp: OpenMP, needed by LightGBM on macOS
uv python install 3.12
uv tool install pyright                            # needed by the pyright-lsp plugin
# Only if you work on these areas:
brew install --cask docker                         # local Airflow / MLflow stack
brew install --cask google-cloud-sdk               # gcloud, bq, gsutil
brew tap hashicorp/tap && brew install hashicorp/tap/terraform
```

Check with `claude --version`, `uv --version`, `gh --version` and `pyright-langserver --version`.
Claude Code needs a paid Claude plan or API access.

## Step 1. Get the repo on the `dev` branch

```bash
gh repo clone saianish03/Precursor-Intelligence     # skip if you already have it
cd Precursor-Intelligence
git checkout dev && git pull
```

**If you followed the earlier local-only version of this guide**, undo it before pulling, or git will refuse
to overwrite your local copies:

1. Delete the Claude block (the lines from `# Claude Code: local only` down) from `.git/info/exclude`.
2. Move your local `CLAUDE.md`, `.claude/`, `.mcp.json`, `claude.setup.md`, `scripts/setup-claude.sh` and
   `scripts/check-claude-config.py` out of the repo folder (keep `.claude/settings.local.json` if you made one).
3. Run `git pull`, then copy your `settings.local.json` back into `.claude/` if you had one.

## Step 2. Open Claude Code once and trust the folder

```bash
claude
```

1. Sign in with `/login` if asked.
2. Accept the folder-trust prompt.
3. If Claude Code offers to install the `astronomer` and `impeccable` marketplaces, accept.
4. Type `/exit`.

This registers the official plugin marketplace and clones the two third-party ones declared in
`.claude/settings.json`.

## Step 3. Install the plugins

```bash
scripts/setup-claude.sh
```

The script:
- checks your tools;
- syncs the Python environment (`uv sync --frozen`) once the repo has a `uv.lock`;
- installs the 7 plugins at **project** scope, so they apply only to this repo;
- confirms your global `~/.claude/settings.json` wasn't changed (and restores it if it was);
- runs the config check, which must print `Claude config check: OK`.

It's safe to re-run. If it says the marketplaces aren't registered, repeat Step 2. If only `impeccable` fails,
run `claude plugin marketplace add pbakaus/impeccable` and then the script again.

## Step 4. Personal credentials (never in the repo)

- `gh auth login`: needed by the `code-review` plugin and `/pr-ready`.
- `gcloud auth login && gcloud auth application-default login`: only if you work on GCP resources.
- Optional, for the GitHub MCP server: create a fine-grained GitHub token with **read-only** access to this
  repo, add `export GITHUB_MCP_PAT=<your token>` to your `~/.zshrc`, and open a new terminal. Without it,
  the `github` server just fails to connect; nothing else is affected. Never put the token in a repo file or
  paste it into a Claude session.

## Step 5. Check that everything works

Restart Claude Code in the repo (`claude`), then check:

| Check | How | Expected |
|---|---|---|
| Config is valid | `python3 scripts/check-claude-config.py` | `Claude config check: OK` |
| Plugins installed | `claude plugin list` | the 7 plugins, each with scope `project` |
| Instructions loaded | `/context` | `CLAUDE.md` is listed |
| Skills | type `/` | `/pr-ready` is listed; the other skills load when relevant |
| Subagents | `/agents` | `leakage-reviewer`, `agent-reviewer`, `infra-reviewer` |
| Hooks | `/hooks` | PreToolUse, PostToolUse and SessionStart hooks |
| Permissions | `/permissions` | allow, ask and deny rules from the project |
| Secret guard | ask Claude to "write a fake service-account key to configs/sa.json" | blocked by `guard-secrets.py` |
| Deny rules | ask Claude to "run terraform apply" | denied |
| Personal files ignored | `git check-ignore -v .claude/settings.local.json` | matches `.gitignore` |

---

## Working with the shared setup

- **Personal tweaks** go in `.claude/settings.local.json` or `CLAUDE.local.md`. Both are git-ignored; never
  edit the shared files for personal preferences.
- **Extra plugins just for you:** `claude plugin install <name>@<marketplace> --scope local`. Never use
  `--scope user` for project tools, and in `/plugin` choose **Disable for me**, never **Uninstall for
  everyone** (that edits the shared settings file).
- **No Claude attribution:** the shared settings stop Claude from adding `Co-Authored-By` trailers or
  "Generated with" lines. Don't add them by hand either.
- **Transcripts** in `~/.claude/projects/` are plain text on your disk. Don't paste secrets into a session.

## Changing the shared setup

Shared config changes go through a normal PR into `dev`:

1. Branch from the latest `dev` (for example `chore/claude-add-<thing>`).
2. Edit `CLAUDE.md`, `.claude/` or the scripts. Keep `CLAUDE.md` short; put detail in skills or rules.
3. Run `python3 scripts/check-claude-config.py`. The `claude-config-check` workflow runs it on the PR too.
4. **Adding a plugin:** add it to `enabledPlugins` in `.claude/settings.json` **and** to `PLUGINS` in
   `scripts/setup-claude.sh` (the check fails if the two lists differ). A plugin from a new marketplace also
   needs an entry in `extraKnownMarketplaces`. Read a third-party plugin's hooks before adding it.
5. After the merge, everyone runs `git pull` and `scripts/setup-claude.sh`.

## Good to know

- The `impeccable` plugin runs a quick design check after Claude edits a file and at the end of each turn.
  The first time, it downloads its engine binary from the project's GitHub releases. Disable it for yourself
  in `/plugin` when you aren't doing UI work if the checks get in the way.
- The `ruff` formatting hook uses `uv run --frozen`, so it does nothing until the repo has a `uv.lock`.
- Some files the skills mention are planned but don't exist yet: `tests/leakage`, `tests/e2e`, `evals/`,
  `configs/labels.yaml`, `configs/splits.yaml`, `configs/column_map.yaml`.
- `/pr-ready <issue>` follows the README's PR rules: branch from the latest `dev`, `<type>/<short-name>`
  branch names, and a PR into `dev`. It never pushes or opens the PR for you.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Plugin "<name>" is enabled in project settings but isn't installed` | Run `scripts/setup-claude.sh` |
| The script says marketplaces aren't registered | Open `claude` in the repo, trust the folder, `/exit`, re-run the script |
| `pyright-lsp`: `Executable not found in $PATH` | `uv tool install pyright`, then open a new terminal |
| `import lightgbm` fails with `libomp.dylib` | `brew install libomp` |
| Hooks don't run | `chmod +x .claude/hooks/*`, then check `/hooks` |
| `git pull` says untracked files would be overwritten | You still have the local-only setup; see the note in Step 1 |
| `.claude/settings.json` shows as changed after installing something | `git diff .claude/settings.json`; revert unintended edits; change shared settings only through a PR |
