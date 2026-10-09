#!/usr/bin/env python3
"""Validate the shared Claude Code configuration in this repository.

Checks JSON validity, plugin entry format, skill/agent frontmatter, hook scripts,
that personal files are not tracked, and that no obvious secrets are committed in
Claude config. Standard library only. Exit code 1 on any error. Used by CI and setup-claude.sh.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []
warnings: list[str] = []


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip().strip('"')
    return meta


def check_settings() -> None:
    p = ROOT / ".claude" / "settings.json"
    if not p.exists():
        errors.append(".claude/settings.json is missing")
        return
    try:
        s = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        errors.append(f".claude/settings.json is not valid JSON: {e}")
        return
    if s.get("permissions", {}).get("defaultMode") == "bypassPermissions":
        errors.append("settings.json must not set defaultMode to bypassPermissions")
    for key in s.get("enabledPlugins", {}):
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*@[a-z0-9][a-z0-9._-]*", key):
            errors.append(f"enabledPlugins key '{key}' should look like <plugin>@<marketplace>")
        mk = key.split("@", 1)[-1]
        if mk != "claude-plugins-official" and mk not in s.get("extraKnownMarketplaces", {}):
            errors.append(f"plugin '{key}' uses marketplace '{mk}' that is not declared in extraKnownMarketplaces")
    for event, groups in s.get("hooks", {}).items():
        for g in groups:
            for h in g.get("hooks", []):
                cmd = h.get("command", "")
                for rel in re.findall(r'\$CLAUDE_PROJECT_DIR"?(/[^\s"]+)', cmd):
                    script = ROOT / rel.lstrip("/")
                    if not script.exists():
                        errors.append(f"hook for {event} points to missing file {rel.lstrip('/')}")
                    elif script.suffix in {".py", ".sh"} and not script.stat().st_mode & 0o111:
                        errors.append(f"hook script {rel.lstrip('/')} is not executable (chmod +x)")


def check_setup_script_in_sync() -> None:
    script = ROOT / "scripts" / "setup-claude.sh"
    settings = ROOT / ".claude" / "settings.json"
    if not (script.exists() and settings.exists()):
        return
    block = re.search(r"PLUGINS=\((.*?)\)", script.read_text(), re.S)
    listed = set(re.findall(r'"([^"]+@[^"]+)"', block.group(1))) if block else set()
    enabled = {k for k, v in json.loads(settings.read_text()).get("enabledPlugins", {}).items() if v}
    if listed != enabled:
        errors.append(f"setup-claude.sh PLUGINS {sorted(listed)} differs from enabledPlugins {sorted(enabled)}")


def check_skills_agents() -> None:
    for skill in sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md")):
        meta = frontmatter(skill)
        name = skill.parent.name
        if not meta.get("description"):
            errors.append(f"skill {name}: missing description in frontmatter")
        if meta.get("name") and meta["name"] != name:
            errors.append(f"skill {name}: frontmatter name '{meta['name']}' does not match folder")
        if len(meta.get("description", "")) > 1024:
            warnings.append(f"skill {name}: description over 1024 characters")
    for agent in sorted((ROOT / ".claude" / "agents").glob("*.md")):
        meta = frontmatter(agent)
        for field in ("name", "description"):
            if not meta.get(field):
                errors.append(f"agent {agent.name}: missing '{field}' in frontmatter")
    claude_md = ROOT / "CLAUDE.md"
    if not claude_md.exists():
        errors.append("CLAUDE.md is missing")
    elif len(claude_md.read_text().splitlines()) > 200:
        warnings.append("CLAUDE.md is over 200 lines; move detail into skills or rules")


def check_git_hygiene() -> None:
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.append("not a git checkout; skipped tracked-file checks")
        return
    personal = {".claude/settings.local.json", "CLAUDE.local.md"}
    for f in tracked:
        if f in personal or f.startswith(".claude/agent-memory-local/"):
            errors.append(f"{f} is personal and must not be committed")
        if re.search(r"(^|/)\.env($|\.)", f) and not f.endswith(".env.example"):
            errors.append(f"{f} looks like an env file and must not be committed")


SECRET = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----|sk-ant-[A-Za-z0-9_\-]{20,}|"
                    r"\b(?:ghp|gho|ghs)_[A-Za-z0-9]{36}\b|AIza[0-9A-Za-z_\-]{35}")


def check_secrets() -> None:
    for p in [ROOT / "CLAUDE.md", *(ROOT / ".claude").rglob("*")]:
        if p.is_file() and p.suffix in {".md", ".json", ".txt", ".py", ".sh"}:
            if p.name == "guard-secrets.py" or p == Path(__file__):
                continue
            if SECRET.search(p.read_text(errors="ignore")):
                errors.append(f"possible secret in {p.relative_to(ROOT)}")


def main() -> int:
    check_settings()
    check_setup_script_in_sync()
    check_skills_agents()
    check_git_hygiene()
    check_secrets()
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("Claude config check:", "FAILED" if errors else "OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
