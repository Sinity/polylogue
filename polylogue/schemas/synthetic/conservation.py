"""Content conservation between a provider wire record and its parsed session.

The wire-support receipt proves a catalogued element reaches a parser and
produces a session. Routability is not conservation: a summary that parses
into an empty block, a chunk whose text is emitted twice, and a turn where a
truncated rendering was preferred over the full field all parse successfully
and all lose content.

Conservation is scoped by semantic role, never by field. Only values sitting
at a schema position annotated with a content-bearing
``x-polylogue-semantic-role`` are subject to it, so the reach of the assertion
is identically the reach of the annotation that drives generation. A field
carrying no content-bearing role is out of scope by construction rather than
by exception list.

Within that scope an absent value is loss unless a reject pin declares a
written reason for dropping it. Silence is loss.

The comparison is equality over multiplicities, never containment. A
truncated rendering is a prefix of the value it replaced, so a containment
test reports it conserved.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base_models import ParsedSession

SEMANTIC_ROLE_KEY = "x-polylogue-semantic-role"

# Roles whose value is archive content and must survive parsing. Roles that
# only locate or describe content (``message_container``, ``message_role``,
# ``message_timestamp``) are not conserved: they are structure, not content.
BODY_ROLE = "message_body"
TITLE_ROLE = "session_title"
CONTENT_BEARING_ROLES: frozenset[str] = frozenset({BODY_ROLE, TITLE_ROLE})

ConservationVerdict: TypeAlias = Literal["loss", "duplication", "mutation"]

_MAX_WALK_DEPTH = 12


@dataclass(frozen=True, slots=True)
class PlantedValue:
    """One content-bearing value found at one position in the wire payload."""

    path: str
    role: str
    value: str


@dataclass(frozen=True, slots=True)
class ConservationFinding:
    """One content-bearing value that did not survive parsing intact."""

    path: str
    role: str
    verdict: ConservationVerdict
    detail: str

    def describe(self) -> str:
        return f"{self.verdict} at {self.path} ({self.role}): {self.detail}"


@dataclass(frozen=True, slots=True)
class ConservationResult:
    """Per-artifact conservation outcome."""

    planted_count: int
    findings: tuple[ConservationFinding, ...] = ()
    excluded_paths: tuple[str, ...] = ()

    @property
    def conserved(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "planted_count": self.planted_count,
            "conserved": self.conserved,
            "findings": [finding.describe() for finding in self.findings],
            "excluded_paths": list(self.excluded_paths),
        }


def _normalise(text: str) -> str:
    """Trim surrounding whitespace only.

    Deliberately not a content normalization: collapsing inner whitespace or
    casing would make a mutated rendering compare equal to the value it
    replaced, which is the defect this module exists to catch.
    """
    return text.strip()


def _schema_variants(schema: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return ``schema`` plus any ``anyOf``/``oneOf`` branches under it."""
    variants: list[Mapping[str, object]] = [schema]
    for keyword in ("anyOf", "oneOf"):
        raw = schema.get(keyword)
        if isinstance(raw, list):
            variants.extend(item for item in raw if isinstance(item, Mapping))
    return variants


def _content_role(variants: Sequence[Mapping[str, object]]) -> str | None:
    for variant in variants:
        role = variant.get(SEMANTIC_ROLE_KEY)
        if isinstance(role, str) and role in CONTENT_BEARING_ROLES:
            return role
    return None


def collect_planted_values(
    schema: Mapping[str, object],
    payload: object,
    *,
    path: str = "$",
    depth: int = 0,
) -> tuple[PlantedValue, ...]:
    """Walk schema and payload together, collecting content-bearing values.

    The schema supplies the role annotation; the payload supplies the value.
    Reading the value from the payload rather than from generator bookkeeping
    keeps the check honest about what the parser was actually handed, and lets
    it run against a captured record as readily as a generated one.

    Every payload position yields at most one value even when several schema
    branches annotate it identically, and repeated array elements stay
    distinct so that a legitimately repeated body is expected twice rather
    than collapsing into one.
    """
    if depth > _MAX_WALK_DEPTH:
        return ()

    variants = _schema_variants(schema)

    role = _content_role(variants)
    if role is not None and isinstance(payload, str):
        # A scalar content-bearing node is a leaf; nothing below it is content.
        if _normalise(payload):
            return (PlantedValue(path=path, role=role, value=_normalise(payload)),)
        return ()
    # Some inferred schemas annotate a union/container that carries a
    # content-bearing role while the selected payload branch is itself an
    # object or array (Claude Code content blocks are the important case).
    # Keep walking non-scalar payloads so the selected string branch is still
    # planted exactly once.
    found: list[PlantedValue] = []
    visited_keys: set[str] = set()
    walked_items = False
    for variant in variants:
        properties = variant.get("properties")
        if isinstance(properties, Mapping) and isinstance(payload, Mapping):
            for key, child_schema in properties.items():
                if key in visited_keys or not isinstance(child_schema, Mapping) or key not in payload:
                    continue
                visited_keys.add(key)
                if (
                    key == "content"
                    and path.startswith("$.message.content[")
                    and payload.get("type") in {"thinking", "text", "tool_use"}
                ):
                    # Claude Code's historical ``content`` union covers
                    # thinking/text/tool-use records, but the parser-owned
                    # body for this broad inferred position is the
                    # tool-result content. Other variants are represented by
                    # their typed fields (or are tool arguments), so planting
                    # them here would compare unrelated projections.
                    continue
                found.extend(
                    collect_planted_values(
                        child_schema,
                        payload[key],
                        path=f"{path}.{key}",
                        depth=depth + 1,
                    )
                )

        additional_properties = variant.get("additionalProperties")
        if isinstance(additional_properties, Mapping) and isinstance(payload, Mapping):
            declared_properties = properties if isinstance(properties, Mapping) else {}
            for key, value in payload.items():
                if key in declared_properties or key in visited_keys:
                    continue
                visited_keys.add(key)
                found.extend(
                    collect_planted_values(
                        additional_properties,
                        value,
                        path=f"{path}.{key}",
                        depth=depth + 1,
                    )
                )

        items = variant.get("items")
        if (
            not walked_items
            and isinstance(items, Mapping)
            and isinstance(payload, Sequence)
            and not isinstance(payload, (str, bytes))
        ):
            walked_items = True
            for position, item in enumerate(payload):
                found.extend(
                    collect_planted_values(
                        items,
                        item,
                        path=f"{path}[{position}]",
                        depth=depth + 1,
                    )
                )

    return tuple(found)


def parsed_block_texts(sessions: Sequence[ParsedSession]) -> Counter[str]:
    """Count block texts across every parsed session.

    Block text only. ``message.text`` is frequently the concatenation of the
    message's own blocks, so counting both would report every value twice and
    make duplication indistinguishable from the normal case.
    """
    counter: Counter[str] = Counter()
    for session in sessions:
        for message in session.messages:
            block_texts: list[str] = []
            for block in message.blocks:
                text = getattr(block, "text", None)
                if isinstance(text, str) and _normalise(text):
                    block_texts.append(_normalise(text))
                tool_input = getattr(block, "tool_input", None)
                if isinstance(tool_input, Mapping):
                    # Claude Code tool-use blocks retain the authored
                    # request in the typed ``content`` input rather than in
                    # block text.  It is one semantic body witness, not a
                    # generic walk of arbitrary tool arguments.
                    tool_content = tool_input.get("content")
                    if isinstance(tool_content, str) and _normalise(tool_content):
                        block_texts.append(_normalise(tool_content))
            if block_texts:
                counter.update(block_texts)
                continue
            # Claude AI's compact export parser keeps authored prose on
            # ``message.text`` without creating text blocks.  Count it only
            # when no block text exists, preserving the no-double-count rule
            # for providers whose message text concatenates their blocks.
            message_text = getattr(message, "text", None)
            if isinstance(message_text, str) and _normalise(message_text):
                counter[_normalise(message_text)] += 1
    return counter


def parsed_titles(sessions: Sequence[ParsedSession]) -> Counter[str]:
    """Count session titles across every parsed session."""
    counter: Counter[str] = Counter()
    for session in sessions:
        title = getattr(session, "title", None)
        if isinstance(title, str) and _normalise(title):
            counter[_normalise(title)] += 1
    return counter


_ARRAY_INDEX_RE = re.compile(r"\[\d+\]")


def normalise_path(path: str) -> str:
    """Reduce a payload path to the form pins are written in.

    ``collect_planted_values`` numbers array positions because two elements of
    one array are two distinct values; a pin names a schema position, which
    has no position number. Erasing the index is what lets one written pin
    cover every element the schema generates at that position.
    """
    return _ARRAY_INDEX_RE.sub("[]", path)


def excluded_paths_from_pins(provider: Provider | str) -> frozenset[str]:
    """Return the payload paths a reject pin declares the parser drops.

    Any reject pin excludes its path, whatever role it names: a rejection is
    a written statement that content at that position is deliberately not
    archived, and a pin on a container covers everything beneath it.

    The written reason is the exclusion, not the ``reject`` action. A pin with
    an empty reason excludes nothing, so its values stay in scope and report
    as loss: an undocumented drop is indistinguishable from a defect, and the
    check exists to refuse that distinction being made silently.
    """
    from polylogue.schemas.pinning import load_pins

    pins = load_pins(provider)
    return frozenset(
        normalise_path(pin.path) for pin in pins.pins if pin.action == "reject" and pin.path and pin.reason.strip()
    )


def _is_excluded(path: str, excluded_paths: frozenset[str]) -> bool:
    """Report whether a reject pin covers ``path`` or an ancestor of it."""
    candidate = normalise_path(path)
    if candidate in excluded_paths:
        return True
    return any(
        candidate.startswith(f"{excluded}.") or candidate.startswith(f"{excluded}[") for excluded in excluded_paths
    )


def _mutation_detail(value: str, observed: Counter[str]) -> str | None:
    for candidate in observed:
        if candidate != value and (candidate in value or value in candidate):
            return (
                f"absent, but parsed output carries a {len(candidate)}-character variant "
                f"of this {len(value)}-character value"
            )
    return None


def check_conservation(
    schema: Mapping[str, object],
    payloads: Sequence[object],
    sessions: Sequence[ParsedSession],
    *,
    excluded_paths: frozenset[str] = frozenset(),
) -> ConservationResult:
    """Verify every content-bearing wire value survives parsing intact.

    Multiplicity is part of the property. A value planted twice must appear
    twice; a value planted once and emitted twice is duplication.
    """
    planted: list[PlantedValue] = []
    excluded: list[str] = []
    for payload in payloads:
        for item in collect_planted_values(schema, payload):
            if _is_excluded(item.path, excluded_paths):
                excluded.append(item.path)
                continue
            planted.append(item)

    block_texts = parsed_block_texts(sessions)
    titles = parsed_titles(sessions)

    findings: list[ConservationFinding] = []
    for role in sorted({item.role for item in planted}):
        observed = titles if role == TITLE_ROLE else block_texts
        expected: Counter[str] = Counter(item.value for item in planted if item.role == role)
        for value, expected_count in sorted(expected.items()):
            observed_count = observed.get(value, 0)
            if observed_count == expected_count:
                continue
            path = next(item.path for item in planted if item.role == role and item.value == value)
            if observed_count > expected_count:
                findings.append(
                    ConservationFinding(
                        path=path,
                        role=role,
                        verdict="duplication",
                        detail=f"appears {observed_count} times in parsed output, expected {expected_count}",
                    )
                )
                continue
            mutation = _mutation_detail(value, observed) if observed_count == 0 else None
            if mutation is not None:
                findings.append(ConservationFinding(path=path, role=role, verdict="mutation", detail=mutation))
                continue
            findings.append(
                ConservationFinding(
                    path=path,
                    role=role,
                    verdict="loss",
                    detail=f"appears {observed_count} times in parsed output, expected {expected_count}",
                )
            )

    return ConservationResult(
        planted_count=len(planted),
        findings=tuple(findings),
        excluded_paths=tuple(sorted(set(excluded))),
    )


__all__ = [
    "BODY_ROLE",
    "CONTENT_BEARING_ROLES",
    "SEMANTIC_ROLE_KEY",
    "TITLE_ROLE",
    "ConservationFinding",
    "ConservationResult",
    "ConservationVerdict",
    "PlantedValue",
    "check_conservation",
    "collect_planted_values",
    "excluded_paths_from_pins",
    "normalise_path",
    "parsed_block_texts",
    "parsed_titles",
]
