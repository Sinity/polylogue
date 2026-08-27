#!/usr/bin/env bash
# Deny test invocations that bypass the managed harness.
#
# `devtools test` and `devtools verify` are the only supported way to run this
# repository's tests. Going around them opts out of the checkout guard, project
# selection semantics, structured pytest outcomes, and the receipts the
# execution host captures.
#
# Applies only to commands running inside this checkout, and only to argv --
# heredoc bodies and quoted strings are data, so a commit message or a file
# being written may say "pytest" freely.
#
# Escape hatch: POLYLOGUE_ALLOW_BARE_PYTEST=1 <command>
# Needing it twice means the harness is missing something; fix devtools instead.
set -euo pipefail

payload="$(cat)"

verdict="$(printf '%s' "$payload" | python3 -c '
import json
import os
import re
import sys
from pathlib import Path

try:
    payload = json.load(sys.stdin)
    command = (payload.get("tool_input") or {}).get("command", "")
except Exception:
    print("allow")
    raise SystemExit(0)

# The harness this guards is this checkout\x27s. A command run elsewhere is not
# its business, and denying there blocks work in repositories that have no
# devtools at all.
root = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
cwd = Path(payload.get("cwd") or os.getcwd())
try:
    cwd = cwd.resolve()
except OSError:
    print("allow")
    raise SystemExit(0)
if root != cwd and root not in cwd.parents:
    print("allow")
    raise SystemExit(0)

if not command:
    print("allow")
    raise SystemExit(0)

if "POLYLOGUE_ALLOW_BARE_PYTEST=1" in command:
    print("allow")
    raise SystemExit(0)

# Drop heredoc bodies: they are content being written, not commands.
kept, skip_until = [], None
for line in command.splitlines():
    if skip_until is not None:
        if line.strip() == skip_until:
            skip_until = None
        continue
    opener = re.search(r"<<-?\s*[\x27\"]?([A-Za-z_][A-Za-z0-9_]*)[\x27\"]?", line)
    if opener:
        skip_until = opener.group(1)
        kept.append(line[: opener.start()])
        continue
    kept.append(line)
text = "\n".join(kept)

# Drop quoted strings: a commit message or a --description is data too.
text = re.sub(r"\x27[^\x27]*\x27", " ", text)
text = re.sub(r"\"(?:[^\"\\\\]|\\\\.)*\"", " ", text)

# A devtools call legitimately runs pytest downstream.
if re.search(r"(^|[;&|(]|\s)devtools(\s|$)", text):
    print("allow")
    raise SystemExit(0)

# pytest must sit at a command position -- start of a line, or after a
# separator -- optionally behind env assignments or an interpreter. Matching it
# after any whitespace would flag prose that merely contains the word.
command_position = r"(?:^|[\n;&|(]|&&|\|\|)\s*(?:\w+=\S+\s+)*"
if re.search(command_position + r"(?:\S*/)?pytest(?:\s|$)", text) or re.search(
    command_position + r"\S*python\S*\s+(?:-\S+\s+)*-m\s+pytest(?:\s|$)", text
):
    print("deny")
else:
    print("allow")
' 2>/dev/null || echo allow)"

[ "$verdict" = "deny" ] || exit 0

cat >&2 <<'MSG'
Bare pytest is not how this repository runs tests.

  use:  devtools test <file-or-nodeid>      (forwards arbitrary pytest args)
        devtools test -k <expr>
        devtools verify                     (full baseline)
        devtools why                        (explain the last run)

Running pytest directly opts out of the checkout guard, project selection
semantics, and structured outcomes.

If devtools is what is blocking you, fix devtools. That is the supported move.

Genuinely need it once: prefix the command with POLYLOGUE_ALLOW_BARE_PYTEST=1
MSG
exit 2
