"""Completion classes that can render a message the shell must not insert.

Click's shipped completion types (``plain``, ``dir``, ``file``) can only offer
values. A completer sometimes has to say *why* it is offering nothing -- the
archive-backed sources answer from the resident daemon, and with no daemon
running an empty list is indistinguishable from "no matching tags". zsh can
render that with ``_message``, which displays text without inserting it, so a
``message`` item is a diagnosis on the completion surface rather than a
candidate the shell might complete to.

bash and fish have no equivalent display primitive, and their shipped templates
already ignore every type they do not recognise, so a ``message`` item degrades
to no completion there rather than to a bogus one.
"""

from __future__ import annotations

from typing import Final

from click.shell_completion import CompletionItem, ZshComplete, add_completion_class

#: Item type meaning "display this, never insert it".
MESSAGE_COMPLETION_TYPE: Final = "message"

_ZSH_DIR_BRANCH: Final = '        elif [[ "$type" == "dir" ]]; then'
_ZSH_MESSAGE_BRANCH: Final = """        elif [[ "$type" == "message" ]]; then
            _message -r "$key"
"""


def _zsh_source_template_with_messages() -> str:
    """Splice a ``message`` branch into Click's own zsh template.

    Derived from the upstream template rather than copied from it, so an
    upstream fix to the completion protocol is inherited. If the anchor ever
    stops matching this raises at import rather than silently shipping a
    template whose message branch is missing.
    """
    template = ZshComplete.source_template
    if _ZSH_DIR_BRANCH not in template:
        raise RuntimeError("Click's zsh completion template no longer has the expected dir branch")
    return template.replace(_ZSH_DIR_BRANCH, _ZSH_MESSAGE_BRANCH + _ZSH_DIR_BRANCH, 1)


class MessageAwareZshComplete(ZshComplete):
    """zsh completion that also renders non-insertable ``message`` items."""

    name = "zsh"
    source_template = _zsh_source_template_with_messages()


def completion_message(text: str) -> CompletionItem:
    """One non-insertable diagnostic line for the completion surface."""
    return CompletionItem(text, type=MESSAGE_COMPLETION_TYPE, help=text)


def register_completion_classes() -> None:
    """Install the message-aware classes over Click's defaults."""
    add_completion_class(MessageAwareZshComplete)


__all__ = [
    "MESSAGE_COMPLETION_TYPE",
    "MessageAwareZshComplete",
    "completion_message",
    "register_completion_classes",
]
