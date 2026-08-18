#!/usr/bin/env bash
# Deny test invocations that bypass the managed harness.
#
# `devtools test` and `devtools verify` are the only supported way to run this
# repository's tests. Going around them silently opts out of the checkout guard
# (which catches a worktree running another checkout's code), containment
# (systemd scope, stall and runtime caps), and the receipts that `devtools why`
# and the merge gate read.
#
# This exists because knowing the rule was demonstrably not enough: bare pytest
# was reached for dozens of times in one session, every time for a real local
# reason. A rule that loses to friction every time it is tested is not a rule.
#
# ONLY EXECUTABLE TEXT IS INSPECTED, and that took two corrections to get right.
# The first version matched the whole command string, so writing a file whose
# CONTENT mentioned pytest was refused. The second stripped heredocs but still
# matched quoted arguments, so a `git commit -m` message mentioning pytest was
# refused. Both times the guard blocked authoring the fix for the thing it
# guards, which is worse than no guard. Heredoc bodies AND quoted strings are
# data, not commands, and are removed before matching.
#
# It matches argv, a structured carrier -- not prose, so this is not the
# natural-language pattern-matching this repo forbids elsewhere.
#
# Escape hatch: POLYLOGUE_ALLOW_BARE_PYTEST=1 <command>
# Needing it twice means the harness is missing something; fix devtools instead.
set -euo pipefail

payload="$(cat)"

verdict="$(printf '%s' "$payload" | python3 -c '
import json
import re
import sys

try:
    command = (json.load(sys.stdin).get("tool_input") or {}).get("command", "")
except Exception:
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

if re.search(r"(^|[;&|(]|\s)(-m\s+pytest(\s|$)|(\S*/)?pytest(\s|$))", text):
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

Running pytest directly opts out of the checkout guard, containment, and the
receipts the merge gate and `devtools why` read -- and it is SLOWER: measured on
the same 1342 tests, 124s through devtools versus 393s bare.

If devtools is what is blocking you, fix devtools. That is the supported move.

Genuinely need it once: prefix the command with POLYLOGUE_ALLOW_BARE_PYTEST=1
MSG
exit 2
