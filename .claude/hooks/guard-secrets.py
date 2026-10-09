#!/usr/bin/env python3
"""PreToolUse hook: block writes to secret files and content that looks like a credential.

Exit 2 blocks the tool call and sends stderr to Claude as feedback. Exit 0 lets the normal
permission flow continue. Standard library only (no jq dependency).
"""
import json
import re
import sys

PROTECTED_PATH_PATTERNS = [
    r"(^|/)\.env(\.[^/]*)?$",
    r"\.pem$",
    r"credentials[^/]*\.json$",
    r"service[-_]?account[^/]*\.json$",
    r"(^|/)\.config/gcloud/",
]
ALLOWED_PATHS = [r"(^|/)\.env\.example$"]
SECRET_PATTERNS = {
    "private key block": r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----",
    "GCP service-account key": r'"private_key"\s*:\s*"-----BEGIN',
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "GitHub token": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|github_pat_[A-Za-z0-9_]{40,}",
    "Anthropic API key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "Slack token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
}


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    tool_input = data.get("tool_input") or {}
    path = str(tool_input.get("file_path") or "").replace("\\", "/")

    if path and not any(re.search(p, path) for p in ALLOWED_PATHS):
        for pattern in PROTECTED_PATH_PATTERNS:
            if re.search(pattern, path):
                print(f"Blocked: {path} is a protected secrets path. Keep credentials out of the repo; "
                      "use Secret Manager or a local, gitignored env file you edit yourself.", file=sys.stderr)
                return 2

    chunks = [tool_input.get("content"), tool_input.get("new_string")]
    chunks += [e.get("new_string") for e in tool_input.get("edits") or [] if isinstance(e, dict)]
    text = "\n".join(c for c in chunks if isinstance(c, str))
    for label, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, text):
            print(f"Blocked: the new content contains what looks like a {label}. Remove it and reference "
                  "the secret through an environment variable or Secret Manager instead.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
