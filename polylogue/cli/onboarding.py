"""The single first-run guided path shared by every human entry point.

Bare ``polylogue`` on an absent archive, ``polylogue tutorial``, and the
strict-command-floor error hint all render the *same* ``GUIDED_PATH_STEPS``
sequence instead of maintaining their own divergent copy of "what should a
cold human type first" (polylogue-jnj.8). Every command listed here is
either executed end-to-end or parsed through the real Click parser by
``tests/unit/cli/test_onboarding_guided_path.py`` — a renamed or removed
subcommand fails that test instead of rotting silently in printed guidance.

The walkthrough is deliberately non-destructive: nothing here writes to a
real chat-source directory, and the demo archive is private-data-free and
disposable.
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import DEMO_CODEX_RECEIPTS_SESSION_ID

_DEMO_READ_QUERY = f"id:{DEMO_CODEX_RECEIPTS_SESSION_ID}"


@dataclass(frozen=True, slots=True)
class GuidedStep:
    """One numbered step in the guided path.

    ``program`` distinguishes the two console-script entry points
    (``polylogue`` vs. the long-lived ``polylogued`` daemon) so tests can
    route each step's ``argv`` through the correct Click group.
    """

    number: int
    title: str
    program: str
    argv: tuple[str, ...]
    note: str

    @property
    def command_text(self) -> str:
        return " ".join((self.program, *self.argv))


GUIDED_PATH_STEPS: tuple[GuidedStep, ...] = (
    GuidedStep(
        number=1,
        title="See where your archive will live",
        program="polylogue",
        argv=("config", "paths"),
        note="Read-only — resolves the configured archive root and tier paths.",
    ),
    GuidedStep(
        number=2,
        title="Write a starter config from detected chat sources",
        program="polylogue",
        argv=("init",),
        note="Idempotent — safe to rerun; never overwrites without --force.",
    ),
    GuidedStep(
        number=3,
        title="Seed the public demo archive",
        program="polylogue",
        argv=("demo", "seed"),
        note="Deterministic and private-data-free — works before any real source is configured.",
    ),
    GuidedStep(
        number=4,
        title="Search and read one demo session",
        program="polylogue",
        argv=("find", _DEMO_READ_QUERY, "then", "read"),
        note="One `find`, one `read`, chained with `then`.",
    ),
    GuidedStep(
        number=5,
        title="Enable real ingestion from your own chat history",
        program="polylogued",
        argv=("run",),
        note="Starts the background daemon against the sources `init` detected; rerun `polylogue tutorial` after.",
    ),
)

GUIDED_PATH_MANUAL_POINTER = "Full manual: polylogue manual   ·   Setup checklist: polylogue tutorial"


def render_guided_path(*, heading: str = "Guided path — no archive found yet") -> str:
    """Render the numbered, non-destructive guided path as plain text.

    Every printed command is a real, exact invocation — no placeholders — so
    it can be copy-pasted verbatim, and so the anti-vacuity test that parses
    each step through the live CLI parser is meaningful.
    """

    lines = [heading, ""]
    for step in GUIDED_PATH_STEPS:
        lines.append(f"  {step.number}. {step.title}")
        lines.append(f"     $ {step.command_text}")
        if step.note:
            lines.append(f"       {step.note}")
    lines.append("")
    lines.append(GUIDED_PATH_MANUAL_POINTER)
    return "\n".join(lines)


__all__ = [
    "GUIDED_PATH_MANUAL_POINTER",
    "GUIDED_PATH_STEPS",
    "GuidedStep",
    "render_guided_path",
]
