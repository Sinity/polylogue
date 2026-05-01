from __future__ import annotations

from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import (
    BreakerMetadata,
    Claim,
    EnvironmentContract,
    Kind,
    RunnerBinding,
    SourceSpan,
    SubjectRef,
    TrustMetadata,
)
from polylogue.proof.rendering import build_catalog_markdown


def _trust() -> TrustMetadata:
    return TrustMetadata(
        producer="tests",
        reviewed_at="2026-04-23T00:00:00+00:00",
        code_revision="test",
        dirty_state=False,
        schema_version=1,
        environment_fingerprint="test",
        runner_version="test",
        freshness="fresh",
        origin="tests",
    )


def test_build_catalog_markdown_renders_schema_provider_and_generated_scenario_sections() -> None:
    subjects = (
        SubjectRef(
            kind="cli.command",
            id="polylogue",
            attrs={},
            source_span=SourceSpan("polylogue/cli/click_app.py", line=10, symbol="cli"),
        ),
        SubjectRef(
            kind="schema.annotation",
            id="schema.annotation.claude-code.message.semantic_role",
            attrs={
                "provider": "claude-code",
                "element_kind": "message",
                "annotation": "x-polylogue-semantic-role",
            },
            source_span=SourceSpan("polylogue/schemas/providers/claude-code/messages.json", line=7),
        ),
        SubjectRef(
            kind="provider.capability",
            id="provider.capability.claude-code",
            attrs={
                "provider": "claude-code",
                "parser_identity": "claude",
                "reasoning_capability": "yes",
                "streaming_capability": "partial",
                "coverage_facets": {"sidecars": "supported"},
                "partial_coverage": ["tool-results"],
            },
            source_span=SourceSpan("polylogue/archive/provider/capabilities.py", line=12),
        ),
        SubjectRef(
            kind="generated.scenario_family",
            id="generated.scenario.audit",
            attrs={
                "status": "active",
                "generated_world": "synthetic",
                "workload_family": "audit",
                "semantic_claims": [{"family": "proof"}, {"family": "performance"}, {"family": ""}],
            },
            source_span=SourceSpan("polylogue/showcase/catalog.py", line=3),
        ),
        SubjectRef(
            kind="cli.command",
            id="missing-source",
            attrs={},
            source_span=None,
        ),
    )
    claims = (
        Claim(
            id="cli.command.help",
            description="help works",
            subject_query=Kind("cli.command"),
            evidence_schema={"type": "object"},
            bug_classes=("cli.help.regression",),
            breaker=BreakerMetadata("help broken"),
        ),
        Claim(
            id="provider.capability.identity_bridge",
            description="provider identity stays coherent",
            subject_query=Kind("provider.capability"),
            evidence_schema={"type": "object"},
            tracked_exception="none",
        ),
    )
    runners = (
        RunnerBinding(
            id="runner:help",
            claim_id="cli.command.help",
            runner="static",
            evidence_class="smoke",
            cost_tier="static",
            freshness_policy="test",
            environment=EnvironmentContract(required_commands=("polylogue",), network="optional"),
            trust=_trust(),
        ),
    )
    catalog = build_verification_catalog(
        subjects=subjects,
        claims=claims,
        runner_bindings=runners,
    )

    rendered = build_catalog_markdown(catalog)

    assert "| `claude-code` | `message` | `x-polylogue-semantic-role` | 1 |" in rendered
    assert "| `claude-code` | claude | yes | partial | `supported` | `tool-results` |" in rendered
    assert "| `generated.scenario.audit` | `active` | synthetic | audit | `proof`<br>`performance` |" in rendered
    assert "| `provider.capability.identity_bridge` | `serious` | — | tracked: none |" in rendered
    assert "| `missing-source` | `<missing>` |" in rendered
    assert "commands=`polylogue`" in rendered


def test_build_catalog_markdown_handles_empty_lists_without_crashing() -> None:
    catalog = build_verification_catalog(subjects=(), claims=(), runner_bindings=())

    rendered = build_catalog_markdown(catalog)

    assert "## Command Subjects" in rendered
    assert "| Command | Source |" in rendered
    assert "## Proof Obligations" in rendered
