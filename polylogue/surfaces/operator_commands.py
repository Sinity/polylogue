"""Shell-safe rendering of archive-derived text in operator-facing commands.

Printed guidance is copy-pasteable by design: a refusal names the command
that resolves it, and the operator runs that command verbatim. Archive refs
are untrusted import data -- ``session_id = origin || ':' || native_id`` and
the native id is taken from a provider export with no charset restriction --
so anything interpolated into such a command is provider-controlled text
reaching the operator's shell.

Two distinct hazards, and they need different answers:

* **Shell syntax.** An unquoted ``;`` or ``$(...)`` in a ref turns the
  recommended command into an injection. :func:`quote_ref_argument` quotes the
  complete argument as one word.
* **Terminal control.** An ESC or CR in a ref can repaint the line it is
  printed on, letting a hostile export forge the surrounding text. Quoting
  does not help -- the bytes still reach the terminal inside the quotes.
  :func:`display_ref` escapes them to a visible ``\\xNN`` form.

Escaping runs *before* quoting, deliberately. A ref carrying control
characters cannot be displayed honestly, so it must not be presented as
pasteable-and-intact either; the shown command is then visibly not the
literal ref, which is the correct signal. Dropping the characters silently --
which is what ``shlex`` alone does -- would print a plausible-looking command
that does not name the session it claims to.

This module is shared by the CLI and the API surfaces so the two renderings
cannot drift apart.
"""

from __future__ import annotations

import shlex

__all__ = ["display_ref", "is_shell_quote_canonical", "quote_ref_argument"]


def display_ref(ref: str) -> str:
    """Return *ref* with control characters escaped to a visible form."""

    return "".join(
        character if character.isprintable() or character == " " else f"\\x{ord(character):02x}" for character in ref
    )


def quote_ref_argument(ref: str, *, id_prefixed: bool = True) -> str:
    """Return *ref* as one shell word, control characters made visible first."""

    shown = display_ref(ref)
    return shlex.quote(f"id:{shown}" if id_prefixed else shown)


def is_shell_quote_canonical(command: str) -> bool:
    """Report whether *command* survives a shlex split/join round trip.

    A command built from :func:`quote_ref_argument` satisfies this by
    construction. One that interpolated a ref raw does not: the metacharacter
    splits into its own token and the rejoin differs. Callers use this as the
    guard that makes forgetting to quote loud rather than exploitable.
    """

    try:
        tokens = shlex.split(command)
    except ValueError:  # unbalanced quoting is never safe to print
        return False
    return shlex.join(tokens) == command
