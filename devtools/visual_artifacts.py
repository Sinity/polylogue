"""Committed reader visual artifact inventory for docs and lab-smoke payloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisualArtifact:
    """One browserless reader artifact emitted by tests/visual."""

    artifact_id: str
    owner: str
    fixture_id: str
    routes: tuple[str, ...]
    evidence_kind: str = "browserless-dom"

    def as_payload(self) -> dict[str, object]:
        """Return the machine-readable inventory shape for lab-smoke reports."""
        return {
            "artifact_id": self.artifact_id,
            "owner": self.owner,
            "fixture_id": self.fixture_id,
            "routes": list(self.routes),
            "evidence_kind": self.evidence_kind,
        }


READER_VISUAL_SMOKE_PYTEST_COMMAND: tuple[str, ...] = ("python", "-m", "pytest", "-q", "tests/visual")
READER_VISUAL_SMOKE_DEVTOOLS_COMMAND: tuple[str, ...] = ("uv", "run", "devtools", "test", "tests/visual")
READER_VISUAL_SMOKE_REPORT: str = ".local/visual/reader-smoke/reader-visual-smoke.json"

READER_VISUAL_ARTIFACTS: tuple[VisualArtifact, ...] = (
    VisualArtifact(
        artifact_id="polylogue.local_reader.search",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/", "/api/sessions", "/api/facets", "/api/facets?origin=...", "/api/facets?query=..."),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.workspace.stack",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-workspace-v1",
        routes=("/w/stack?ids=...&focus=...", "/api/stack?ids=..."),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.workspace.compare",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-workspace-v1",
        routes=("/w/compare?left=...&right=...&align=prompt", "/api/compare?left=...&right=...&align=prompt"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.session",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/s/{id}", "/api/sessions/{id}", "/api/sessions/{id}/messages", "/api/sessions/{id}/raw"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.search.query",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/api/sessions?query=...",),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.cost_panel",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/api/sessions/{id}/cost",),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.evidence_panel",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/s/{id}", "/api/sessions/{id}/artifacts", "/api/sessions/{id}/neighbors"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.overlay_mutations",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/s/{id}", "/api/overlays/*"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.operator_flow",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/s/{id}", "/api/sessions/{id}/context", "/api/overlays/*"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.insights_browser",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/api/insights/sessions/{id}",),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.degraded",
        owner="tests/visual/test_reader_dom_smoke.py",
        fixture_id="reader-visual-synthetic-empty-and-degraded-v1",
        routes=("/api/sessions?query=...",),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.paste_spans",
        owner="tests/visual/test_reader_paste_spans.py",
        fixture_id="reader-visual-synthetic-v1+diff",
        routes=("/p", "/api/paste-browser"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.paste_browser_empty",
        owner="tests/visual/test_reader_paste_spans.py",
        fixture_id="reader-visual-empty-archive",
        routes=("/api/paste-browser",),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.attachment_surface",
        owner="tests/visual/test_reader_attachments.py",
        fixture_id="reader-visual-attachments-v1",
        routes=("/a", "/api/attachments", "/api/sessions/{id}/attachments"),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.attachment_library_empty",
        owner="tests/visual/test_reader_attachments.py",
        fixture_id="reader-visual-attachments-empty",
        routes=("/api/attachments",),
    ),
    VisualArtifact(
        artifact_id="polylogue.local_reader.message_card",
        owner="tests/visual/test_reader_action_rail.py",
        fixture_id="reader-visual-synthetic-v1",
        routes=("/", "/api/sessions", "/api/messages/{id}/actions"),
    ),
)


def reader_visual_artifact_payloads() -> list[dict[str, object]]:
    """Return the inventory in stable JSON order."""
    return [artifact.as_payload() for artifact in READER_VISUAL_ARTIFACTS]


__all__ = [
    "READER_VISUAL_ARTIFACTS",
    "READER_VISUAL_SMOKE_DEVTOOLS_COMMAND",
    "READER_VISUAL_SMOKE_PYTEST_COMMAND",
    "READER_VISUAL_SMOKE_REPORT",
    "VisualArtifact",
    "reader_visual_artifact_payloads",
]
