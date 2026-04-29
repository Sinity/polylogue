"""Effect-driven claim compiler.

Walks the declared operation catalog and mints one proof subject per
effect-implication pair. Each subject carries the operation name, the
declared effect, and the specific implication that must hold.

Claims match on the implication attribute — one claim template covers
every operation that shares an implication, keeping the claim count
proportional to the implication vocabulary (~17) rather than to
operations × effects.
"""

from __future__ import annotations

from typing import Literal

from polylogue.operations.specs import Effect, build_declared_operation_catalog
from polylogue.proof.models import Oracle, SourceSpan, SubjectRef

# ─── effect → implication mapping ──────────────────────────────────────────

EffectImplication = Literal[
    "deterministic",
    "no_side_effect",
    "snapshot_consistent",
    "preview",
    "idempotent",
    "rollback_safe",
    "atomic",
    "path_sanitized",
    "atomic_rename",
    "parent_exists",
    "timeout_bounded",
    "retry_bounded",
    "sampling_bounded",
    "privacy_safe_evidence",
    "explicit_dry_run_evidence",
    "confirmed_before_execute",
]

EFFECT_IMPLICATIONS: dict[Effect, tuple[EffectImplication, ...]] = {
    "Pure": ("deterministic", "no_side_effect"),
    "DbRead": ("snapshot_consistent",),
    "DbWrite": ("preview", "idempotent", "rollback_safe", "atomic"),
    "FileWrite": ("path_sanitized", "atomic_rename", "parent_exists"),
    "Network": ("timeout_bounded", "retry_bounded"),
    "LiveArchive": ("sampling_bounded", "privacy_safe_evidence"),
    "Destructive": ("explicit_dry_run_evidence", "confirmed_before_execute"),
}

# ─── implication → oracle ──────────────────────────────────────────────────

IMPLICATION_ORACLE: dict[EffectImplication, Oracle] = {
    "deterministic": "construction_sanity",
    "no_side_effect": "construction_sanity",
    "snapshot_consistent": "construction_sanity",
    "preview": "smoke",
    "idempotent": "proof",
    "rollback_safe": "proof",
    "atomic": "proof",
    "path_sanitized": "construction_sanity",
    "atomic_rename": "construction_sanity",
    "parent_exists": "construction_sanity",
    "timeout_bounded": "construction_sanity",
    "retry_bounded": "construction_sanity",
    "sampling_bounded": "construction_sanity",
    "privacy_safe_evidence": "manual_review",
    "explicit_dry_run_evidence": "proof",
    "confirmed_before_execute": "proof",
}

# ─── severity classification ───────────────────────────────────────────────

DESTRUCTIVE_EFFECTS: frozenset[Effect] = frozenset({"DbWrite", "FileWrite", "Destructive"})


class EffectClaimCompiler:
    """Mints proof subjects from OperationSpec.effects.

    For each operation in the operation catalog, walks its declared
    effects and emits one subject per effect-implication pair. The
    resulting subjects carry enough attribution (operation name, effect,
    implication) for claims to match by implication attr.
    """

    @staticmethod
    def compile_subjects() -> tuple[SubjectRef, ...]:
        """Return one subject per (operation, effect, implication) triple."""
        catalog = build_declared_operation_catalog()
        subjects: list[SubjectRef] = []

        for spec in catalog.specs:
            if not spec.effects:
                continue
            for effect in spec.effects:
                implications = EFFECT_IMPLICATIONS.get(effect, ())
                for implication in implications:
                    subjects.append(
                        SubjectRef(
                            kind="operation.spec.effect",
                            id=f"operation.{spec.name}.{implication}",
                            attrs={
                                "operation_name": spec.name,
                                "effect": effect,
                                "implication": implication,
                            },
                            source_span=SourceSpan(
                                path="polylogue/operations/specs.py",
                                symbol=spec.name,
                            ),
                        )
                    )

        return tuple(sorted(subjects, key=lambda s: s.id))

    @staticmethod
    def implication_names() -> tuple[EffectImplication, ...]:
        """All distinct implications across the effect vocabulary."""
        seen: dict[str, EffectImplication] = {}
        for imps in EFFECT_IMPLICATIONS.values():
            for imp in imps:
                seen.setdefault(imp, imp)
        return tuple(seen.values())

    @staticmethod
    def oracle_for(implication: EffectImplication) -> Oracle:
        return IMPLICATION_ORACLE.get(implication, "construction_sanity")

    @staticmethod
    def is_severe(effect: Effect) -> bool:
        return effect in DESTRUCTIVE_EFFECTS


def effect_implication_subjects() -> tuple[SubjectRef, ...]:
    """Convenience wrapper matching the subject-compiler convention in subjects.py."""
    return EffectClaimCompiler.compile_subjects()
