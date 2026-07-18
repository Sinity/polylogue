"""Guard against campaign vocabulary leaking into the browser-action conduit (polylogue-ptx AC6).

``polylogue-ptx`` replaced the Sol-specific ``BrowserLaunchJob`` queue and the
text-only ``BrowserPostCommand`` with one provider-neutral
``BrowserActionIntent`` conduit (PR #2928). Per that bead's design note,
mission/run/iteration/deliverable/package/cadence campaign identity is a
concern of the external orchestrator (``polylogue-yyvg.6``), never of the
extension or receiver product code. This static check walks the receiver
(``polylogue/browser_capture/``, ``polylogue/daemon/browser_capture.py``) and
the extension (``browser-extension/src/``) and asserts neither the retired
class names nor campaign-identity field names have been reintroduced.

External campaign tooling under ``.agent/handoffs/`` and devtools campaign
receipt/report modules are explicitly out of scope — they are the legitimate
orchestrator client this bead's design note describes, not the conduit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

RECEIVER_ROOTS = (
    REPO_ROOT / "polylogue" / "browser_capture",
    REPO_ROOT / "polylogue" / "daemon" / "browser_capture.py",
)
EXTENSION_ROOT = REPO_ROOT / "browser-extension" / "src"

# Word-boundary tokens. "mission" alone is deliberately excluded: the
# extension's ambient/popup status surface is legitimately named "mission
# control" (a UI metaphor for cross-conversation archive intelligence, see
# polylogue-yyvg.7) and is unrelated to campaign MissionId identity.
FORBIDDEN_TOKENS = (
    "BrowserPostCommand",
    "BrowserLaunchJob",
    "LaunchJob",
    "mission_id",
    "MissionId",
    "deliverable_id",
    "DeliverableId",
    "package_revision",
    "PackageRevisionId",
    "cadence_strategy",
    "campaign_id",
    "CampaignId",
    "handoff_id",
    "HandoffId",
)
TOKEN_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t in FORBIDDEN_TOKENS) + r")\b")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root in RECEIVER_ROOTS:
        if root.is_dir():
            files.extend(sorted(p for p in root.rglob("*.py") if "test" not in p.parts))
        elif root.is_file():
            files.append(root)
    files.extend(sorted(EXTENSION_ROOT.rglob("*.js")))
    return files


def test_conduit_source_rejects_retired_campaign_vocabulary() -> None:
    violations: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = TOKEN_PATTERN.search(line)
            if match:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {match.group(1)!r} — {line.strip()}")
    assert not violations, (
        "Campaign vocabulary reintroduced into the provider-neutral browser-action "
        "conduit (polylogue-ptx AC6). Campaign/mission/deliverable/package identity "
        "belongs to the external orchestrator (polylogue-yyvg.6), not extension or "
        "receiver product code:\n" + "\n".join(violations)
    )
