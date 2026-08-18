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
# was reached for dozens of times in a single session, every time for a real
# local reason -- the wrapper failing inside a worktree, lock contention with a
# running gate, or simply fewer characters in a tight loop. A rule that loses to
# friction every time it is tested is not a rule, so this makes the bypass fail
# closed instead.
#
# It matches the COMMAND, not prose: argv is a structured carrier, so this is not
# the natural-language pattern-matching this repo forbids elsewhere.
#
# Escape hatch, for the cases the harness genuinely cannot serve:
#   POLYLOGUE_ALLOW_BARE_PYTEST=1 <command>
# If you reach for that more than once, the harness is missing something --
# fix devtools rather than keep escaping it.
set -euo pipefail

payload="$(cat)"
command="$(printf '%s' "$payload" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("tool_input") or {}).get("command",""))' 2>/dev/null || true)"

[ -n "$command" ] || exit 0
case "$command" in
  *POLYLOGUE_ALLOW_BARE_PYTEST=1*) exit 0 ;;
esac

# `-m pytest` in any form, or pytest as a bare command word.
if printf '%s' "$command" | grep -qE '(^|[;&|[:space:]])(-m[[:space:]]+pytest\b|[^[:space:]/]*pytest)([[:space:]]|$)'; then
  # `devtools test` legitimately contains the word downstream; only deny direct calls.
  if printf '%s' "$command" | grep -qE '(^|[;&|[:space:]])devtools([[:space:]]|$)'; then
    exit 0
  fi
  cat >&2 <<'MSG'
Bare pytest is not how this repository runs tests.

  use:  devtools test <file-or-nodeid>      (forwards arbitrary pytest args)
        devtools test -k <expr>
        devtools verify                     (full baseline)
        devtools why                        (explain the last run)

Running pytest directly opts out of the checkout guard, containment, and the
receipts the merge gate and `devtools why` read -- and it is SLOWER: measured on
the same 1342 tests, 124s through devtools versus 393s bare.

If devtools is what is blocking you, fix devtools. That is the supported move,
and it is cheaper than the debugging that a bypassed guard eventually causes.

Genuinely need it once: prefix the command with POLYLOGUE_ALLOW_BARE_PYTEST=1
MSG
  exit 2
fi
exit 0
