"""Measured public record and content grammar on the original sampler owner."""

from __future__ import annotations

import math
import random
import sqlite3
from collections.abc import Callable, Generator

from polylogue.core.tool_result_pointer import pointer_width_relation
from polylogue.schemas.field_stats.distributions import _bucket_index
from polylogue.schemas.field_stats.models import FilenameWidthStats, PointerShapeStats
from polylogue.schemas.record_structure import (
    CODEX_PUBLIC_MESSAGE_ROLES,
    PUBLIC_CONTENT_BODY_FIELDS,
    content_prefix_kind,
    initial_content_prefix,
    published_kind_tokens,
    tool_identity_operand,
)
from polylogue.schemas.synthetic.models import GeneratedString, SchemaRecord
from polylogue.schemas.synthetic.observed_values import (
    existing_value_is_supported,
    measured_string_formats,
    observed_length_support,
)
from polylogue.schemas.synthetic.relations import ForeignKeyGraph
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText

ContentState = tuple[str, str]


def _count(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("public grammar counts must be positive exact integers")
    return value


def _tool_operand_coordinates(row: SchemaRecord, kind: str) -> tuple[str, str, str, str, str, int]:
    """One declared event/history predicate for path proof and actual binding."""
    event, field, presence, relation = (row.get(key) for key in ("event", "field", "presence", "relation"))
    if (
        not all(isinstance(value, str) for value in (event, field, presence, relation))
        or event not in {"call", "result"}
        or field not in {"id", "call_id", "tool_use_id", "tool_call_id"}
        or presence not in {"absent", "null", "string", "other"}
        or relation not in {"new", "call", "result", "both", "missing"}
    ):
        raise ValueError("tool relation contains undeclared coordinates")
    declared = tool_identity_operand({"type": kind})
    if declared is None or declared[1] != event:
        raise ValueError("tool relation contradicts its public event kind")
    if presence != "string" and relation != "missing":
        raise ValueError("tool relation contradicts its identity presence")
    prior_mode = row.get("prior_result_mode")
    prior_count = row.get("prior_result_count_bucket")
    if type(prior_count) is not int or prior_count < 0:
        raise ValueError("tool history count evidence is missing; re-infer the package")
    if not isinstance(prior_mode, str):
        raise ValueError("tool history evidence is missing; re-infer the package")
    return event, field, presence, relation, prior_mode, prior_count


def _state(kind: object, seen: object) -> ContentState:
    if not isinstance(kind, str) or not kind.startswith("block:"):
        raise ValueError("public content kind is malformed")
    if kind.removeprefix("block:") not in published_kind_tokens() and kind != "block:other":
        raise ValueError("content grammar contains an unpublished discriminator")
    if seen not in {
        "content",
        "user_text",
        "user_tool_result",
        "assistant_text",
        "assistant_thinking",
        "assistant_tool_use",
    }:
        raise ValueError("public content prefix state is malformed")
    assert isinstance(seen, str)
    return kind, seen


def _state_rows(profile: SchemaRecord, field: str) -> dict[ContentState, int]:
    rows = profile.get(field)
    if not isinstance(rows, list):
        raise ValueError("conditional content grammar is missing; re-infer the package")
    result: dict[ContentState, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("public content state is malformed")
        state = _state(row.get("kind"), row.get("seen"))
        if state in result:
            raise ValueError("public content state occurs twice")
        result[state] = _count(row.get("count"))
    return result


def _key(state: ContentState) -> str:
    return f"{state[0]}~{state[1]}"


def _logsum(values: list[float]) -> float | None:
    if not values:
        return None
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


class ConditionalContentKinds:
    """Sample measured transitions conditioned on count and terminal class.

    This reconstructs a Markov model from the published aggregate evidence;
    it does not recover an observed private sequence. The length's owner is
    still the existing count sketch sampler. Backward path weights remain on
    the same native relation owner, with only one finite vocabulary row in
    Python. No block position is fixed to make a record classify correctly.
    """

    def __init__(
        self, provider: str, record_kind: str, profile: SchemaRecord, count: int, owner: ForeignKeyGraph
    ) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("content count must be a nonnegative exact integer")
        self.count = count
        self.owner = owner
        self.starts = _state_rows(profile, "initial_states")
        endings = _state_rows(profile, "terminal_states")
        self.edges: dict[ContentState, dict[ContentState, int]] = {}
        rows = profile.get("transitions")
        if not isinstance(rows, list):
            raise ValueError("conditional content transitions are missing; re-infer the package")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("conditional content transition is malformed")
            before = _state(row.get("from"), row.get("from_seen"))
            after = _state(row.get("to"), row.get("to_seen"))
            if after[1] != content_prefix_kind(provider, record_kind, before[1], after[0]):
                raise ValueError("content transition contradicts its public prefix class")
            destinations = self.edges.setdefault(before, {})
            if after in destinations:
                raise ValueError("conditional content transition occurs twice")
            destinations[after] = _count(row.get("count"))
        initial = initial_content_prefix(provider, record_kind)
        for state in self.starts:
            if state[1] != content_prefix_kind(provider, record_kind, initial, state[0]):
                raise ValueError("content start contradicts its public prefix class")
        expected_end = record_kind if initial != "content" else "content"
        if any(state[1] != expected_end for state in endings):
            raise ValueError("content ending contradicts the measured record kind")
        if not count and expected_end != initial:
            raise ValueError("empty content contradicts the measured record kind")
        self.path = _MeasuredPath(self.starts, endings, self.edges, count, owner)

    def sample(self, rng: random.Random) -> Generator[str, None, None]:
        for kind in self.path.sample(rng):
            if kind == "block:other":
                raise ValueError("content has no published discriminator; re-infer the package")
            yield kind.removeprefix("block:")


class OpaqueReferenceKinds:
    """Borrow the existing native path sampler for measured reference files.

    The caller draws file count from its existing distribution owner. Reference
    classes describe public filename/pointer relationships, never private names.
    """

    def __init__(
        self, mode: str, profile: SchemaRecord, count: int, owner: ForeignKeyGraph, *, presence: str | None = None
    ) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("opaque file count must be a nonnegative exact integer")
        shape, separator, disposition = mode.partition(":")
        if shape not in {"absent", "null", "string", "array", "object", "other"} or not separator:
            raise ValueError("opaque reference content shape is malformed")
        if disposition not in {"inline", "truncated"} or (disposition == "truncated" and shape != "string"):
            raise ValueError("opaque reference disposition contradicts its content shape")
        suffix = "_truncated" if disposition == "truncated" else "_inline"
        vocabulary = {"direct" + suffix, "persisted" + suffix}

        def state(value: object) -> ContentState:
            if not isinstance(value, str) or value not in vocabulary:
                raise ValueError("opaque reference class contradicts its original content shape")
            return value, "opaque"

        def weights(field: str) -> dict[ContentState, int]:
            value = profile.get(field)
            if not isinstance(value, dict):
                raise ValueError("opaque reference grammar is missing; re-infer the package")
            return {state(key): _count(weight) for key, weight in value.items()}

        starts = weights("initial_classes")
        endings = weights("terminal_classes")
        edges: dict[ContentState, dict[ContentState, int]] = {}
        rows = profile.get("transitions")
        if not isinstance(rows, list):
            raise ValueError("opaque reference transitions are missing; re-infer the package")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("opaque reference transition is malformed")
            before, after = state(row.get("from")), state(row.get("to"))
            destinations = edges.setdefault(before, {})
            if after in destinations:
                raise ValueError("opaque reference transition occurs twice")
            destinations[after] = _count(row.get("count"))
        if presence is not None:
            if presence not in {"none", "direct", "persisted", "direct+persisted"}:
                raise ValueError("opaque reference presence is malformed")
            if (presence == "none") != (count == 0):
                raise ValueError("opaque reference presence contradicts file count")
            if count:

                def accumulated(before: str, after: str) -> str:
                    family = after.split("_", 1)[0]
                    return family if before == family else "direct+persisted"

                expanded: dict[ContentState, dict[ContentState, int]] = {}
                for (before_kind, _unused), destinations in edges.items():
                    for seen in (before_kind.split("_", 1)[0], "direct+persisted"):
                        expanded[(before_kind, seen)] = {
                            (after[0], accumulated(seen, after[0])): weight for after, weight in destinations.items()
                        }
                starts = {(kind, kind.split("_", 1)[0]): weight for (kind, _unused), weight in starts.items()}
                endings = {
                    (kind, presence): weight
                    for (kind, _unused), weight in endings.items()
                    if presence == "direct+persisted" or kind.startswith(presence + "_")
                }
                edges = expanded
        self.path = _MeasuredPath(starts, endings, edges, count, owner)

    def sample(self, rng: random.Random) -> Generator[str, None, None]:
        yield from self.path.sample(rng)


class OpaqueResultHistoryKinds:
    """Condition prior result shapes on the actual final reference mode.

    This uses the same measured path and original native owner as the other
    public grammars. A persisted-inline file must not be realized by inventing
    a pointer in its final null/inline field; its preceding records own it.
    """

    def __init__(self, terminal_mode: str, profile: SchemaRecord, count: int, owner: ForeignKeyGraph) -> None:
        if type(count) is not int or count < 1:
            raise ValueError("result history requires its original positive record count")

        def state(value: object) -> ContentState:
            if not isinstance(value, str):
                raise ValueError("opaque result history mode is malformed")
            shape, separator, disposition = value.partition(":")
            if shape not in {"absent", "null", "string", "array", "object", "other"} or not separator:
                raise ValueError("opaque result history content shape is malformed")
            if disposition not in {"inline", "truncated"} or (disposition == "truncated" and shape != "string"):
                raise ValueError("opaque result history contradicts its content shape")
            return value, "opaque-history"

        final = state(terminal_mode)

        def weights(field: str) -> dict[ContentState, int]:
            rows = profile.get(field)
            if not isinstance(rows, dict):
                raise ValueError("opaque result history is missing; re-infer the package")
            return {state(key): _count(value) for key, value in rows.items()}

        starts, endings = weights("initial_modes"), weights("terminal_modes")
        if set(endings) != {final}:
            raise ValueError("opaque result history contradicts the selected terminal shape")
        edges: dict[ContentState, dict[ContentState, int]] = {}
        rows = profile.get("transitions")
        if not isinstance(rows, list):
            raise ValueError("opaque result history transitions are missing; re-infer the package")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("opaque result history transition is malformed")
            before, after = state(row.get("from")), state(row.get("to"))
            destinations = edges.setdefault(before, {})
            if after in destinations:
                raise ValueError("opaque result history transition occurs twice")
            destinations[after] = _count(row.get("count"))
        self.path = _MeasuredPath(starts, endings, edges, count, owner)

    def sample(self, rng: random.Random) -> Generator[str, None, None]:
        yield from self.path.sample(rng)


class _MeasuredPath:
    """Native backward weights and exact suffix feasibility on one owner.

    Count feasibility filters the original log-mass draws. It does not claim
    to recompute probabilities conditioned on the complete count joint.
    """

    def __init__(
        self,
        starts: dict[ContentState, int],
        endings: dict[ContentState, int],
        edges: dict[ContentState, dict[ContentState, int]],
        count: int,
        owner: ForeignKeyGraph,
        *,
        resource_effects: dict[ContentState, tuple[int, int]] | None = None,
        available_resource: Callable[[], int] | None = None,
        count_effects: dict[ContentState, int] | None = None,
        selected_count: int | None = None,
    ) -> None:
        self.count = count
        self.owner = owner
        self.starts = starts
        self.resource_effects = resource_effects
        self.available_resource = available_resource
        self.count_effects = count_effects
        self.selected_count = selected_count
        if (resource_effects is None) != (available_resource is None):
            raise ValueError("path resource effects require their original inventory")
        if (count_effects is None) != (selected_count is None):
            raise ValueError("path count effects require their original selected joint")
        if selected_count is not None and (type(selected_count) is not int or not 0 <= selected_count <= count):
            raise ValueError("selected message count contradicts its physical record count")
        if count_effects is not None and resource_effects is not None:
            raise ValueError("combined count/resource path requires exact joint feasibility")
        self.edges = edges
        self.states = sorted(
            set(starts)
            | set(endings)
            | set(edges)
            | {destination for destinations in edges.values() for destination in destinations}
        )
        if count_effects is not None and (
            set(count_effects) != set(self.states)
            or any(type(effect) is not int or effect not in {0, 1} for effect in count_effects.values())
        ):
            raise ValueError("path count effects must classify every original state")
        self.denominators = {state: sum(edges.get(state, {}).values()) + endings.get(state, 0) for state in self.states}
        if any(not weight for weight in self.denominators.values()):
            raise ValueError("public content state has neither a measured successor nor an ending")
        self.layers = owner.new_array()
        mask = (1 << (selected_count + 1)) - 1 if selected_count is not None else 0
        previous: dict[ContentState, tuple[float, int, int]] = {}
        for state, weight in endings.items():
            required = resource_effects.get(state, (0, 0))[0] if resource_effects is not None else 0
            support = (1 << count_effects[state]) & mask if count_effects is not None else 0
            previous[state] = math.log(weight) - math.log(self.denominators[state]), required, support
        self._retain_layer(0, previous)
        for remaining in range(1, count):
            current: dict[ContentState, tuple[float, int, int]] = {}
            for state in self.states:
                successors = [
                    (destination, weight)
                    for destination, weight in edges.get(state, {}).items()
                    if destination in previous
                ]
                mass = _logsum(
                    [
                        math.log(weight) - math.log(self.denominators[state]) + previous[destination][0]
                        for destination, weight in successors
                    ]
                )
                if mass is not None:
                    required = 0
                    if resource_effects is not None:
                        own_required, change = resource_effects.get(state, (0, 0))
                        required = min(
                            max(own_required, previous[destination][1] - change) for destination, _ in successors
                        )
                    support = 0
                    if count_effects is not None:
                        for destination, _ in successors:
                            support |= previous[destination][2]
                        support = (support << count_effects[state]) & mask
                    current[state] = mass, required, support
            self._retain_layer(remaining, current)
            previous = current
        if (
            selected_count is not None
            and count
            and not any(state in previous and (previous[state][2] >> selected_count) & 1 for state in starts)
        ):
            raise ValueError("selected message/record joint has no complete original grammar path")

    def _connection(self) -> sqlite3.Connection:
        if self.owner._generation != self.layers.generation:
            raise RuntimeError("public content path lost its original native generation")
        return self.owner._live_tool_owner()

    def _retain_layer(self, remaining: int, values: dict[ContentState, tuple[float, int, int]]) -> None:
        # Only the finite grammar's current layer is held in Python. The tape
        # grows with exact length and quota; SQLite owns its physical lifetime.
        self._connection().executemany(
            "INSERT INTO generated_path_layers VALUES (?,?,?,?,?,?)",
            (
                (
                    self.layers.array_id,
                    remaining,
                    _key(state),
                    mass,
                    required,
                    support.to_bytes((support.bit_length() + 7) // 8, "little"),
                )
                for state, (mass, required, support) in values.items()
            ),
        ).close()

    def _layer(self, remaining: int) -> dict[str, tuple[float, int, int]]:
        cursor = self._connection().execute(
            "SELECT state,log_mass,required_resource,count_support FROM generated_path_layers WHERE path_id=? AND remaining=?",
            (self.layers.array_id, remaining),
        )
        try:
            return {
                state: (mass, required, int.from_bytes(support, "little")) for state, mass, required, support in cursor
            }
        finally:
            cursor.close()

    def sample(self, rng: random.Random) -> Generator[str, None, None]:
        if not self.count:
            return
        previous: ContentState | None = None
        quota = self.selected_count
        for ordinal in range(self.count):
            remaining = self.count - ordinal - 1
            paths = self._layer(remaining)
            candidates = self.starts if previous is None else self.edges.get(previous, {})
            weighted = [
                (state, math.log(weight) + paths[_key(state)][0])
                for state, weight in candidates.items()
                if _key(state) in paths
                and (self.available_resource is None or paths[_key(state)][1] <= self.available_resource())
                and (quota is None or (paths[_key(state)][2] >> quota) & 1)
            ]
            if not weighted:
                raise ValueError("measured count and content transitions have no complete path; re-infer the package")
            maximum = max(weight for _, weight in weighted)
            chosen = rng.choices(
                [state for state, _ in weighted],
                weights=[math.exp(weight - maximum) for _, weight in weighted],
                k=1,
            )[0]
            previous = chosen
            if quota is not None:
                assert self.count_effects is not None
                quota -= self.count_effects[chosen]
            yield chosen[0]


def _weights(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("conditional public record evidence is missing; re-infer the package")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("public grammar vocabulary is malformed")
    return {key: _count(count) for key, count in value.items()}


def _choose(value: object, rng: random.Random) -> str:
    weights = _weights(value)
    if not weights:
        raise ValueError("conditional public record evidence is malformed")
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


class PublicRecordPlan:
    """One measured record kind/form, bound to declared provider coordinates.

    Discriminators are public wire vocabulary. Counts, shape, and transitions
    come from the package's joint aggregate evidence, not observed private
    token lists or a provider template.
    """

    def __init__(
        self,
        provider: str,
        structure: SchemaRecord,
        kind: str,
        rng: random.Random,
        owner: ForeignKeyGraph,
        schema: SchemaRecord,
        *,
        header_class: SchemaRecord | None = None,
        metadata_class: SchemaRecord | None = None,
        agent_id: str | None = None,
        agent_stem: str | None = None,
    ) -> None:
        self.owner = owner
        self.generation = owner._generation
        self.schema = schema
        self.agent_children: tuple[str, tuple[int, ...]] | None = None
        self.agent_metadata: SchemaRecord | None = None
        self.agent_identity_prefix: bool | None = None
        self.header_types: dict[str, frozenset[str]] = {}
        self.graph_values: dict[str, str | None] = {}
        self.string_exclusions: dict[str, frozenset[str]] = {}
        self.tool_content_types: dict[str, frozenset[str]] = {}
        self.tool_pointer_fields: dict[str, bool] = {}
        self.tool_widths: dict[str, int] = {}
        self.tool_pointer_widths: dict[str, tuple[str, int, str]] = {}
        self.tool_pointer_relations: dict[str, str] = {}
        self.tool_fields: dict[str, tuple[str, str, str, GeneratedString | None, SchemaRecord | None, str]] = {}
        self.provider = provider
        self.kind = kind
        self.scalars: dict[str, str | None] = {}
        self.absent: set[str] = set()
        self.required_body_fields: set[str] = set()
        self.content_type: str | None = None
        if metadata_class is not None:
            if provider != "claude-code" or kind != "agent_sidecar_meta" or agent_id is None or agent_stem is None:
                raise ValueError("metadata binding requires its original dispatcher")
            self.form, self.mode = "claude-agent-meta", "absent"
            self.content = {}
            self.content_path = ""
            self.content_type = None
            self.bind_agent_metadata(metadata_class, agent_id=agent_id, agent_stem=agent_stem)
            return
        forms = structure.get("record_forms_by_kind")
        form_weights = forms.get(self.kind) if isinstance(forms, dict) else None
        self.form = str(header_class["form"]) if header_class is not None else _choose(form_weights, rng)
        if self.form not in _weights(form_weights):
            raise ValueError("first-header form has no original conditional record branch")
        contents = structure.get("content_blocks_by_record_kind")
        by_kind = contents.get(self.kind) if isinstance(contents, dict) else None
        by_form = by_kind.get(self.form) if isinstance(by_kind, dict) else None
        if not isinstance(by_form, dict) or not by_form:
            raise ValueError("conditional public content form is missing; re-infer the package")
        representations: dict[str, int] = {}
        for mode, profile in by_form.items():
            if not isinstance(profile, dict):
                raise ValueError("conditional public content form is malformed")
            counts = _weights(profile.get("representations"))
            if set(counts) != {mode}:
                raise ValueError("conditional content representation contradicts its form")
            representations[mode] = counts[mode]
        self.mode = _choose(representations, rng)
        profile = by_form[self.mode]
        assert isinstance(profile, dict)
        self.content = profile
        if provider == "claude-code":
            if self.form == "claude-message":
                self.scalars["$.type"] = "user" if self.kind.startswith("user_") else "assistant"
                self.content_path = "$.message.content"
            else:
                parts = self.kind.split(":")
                if len(parts) not in {2, 3} or parts[0] != "record" or parts[1] == "other":
                    raise ValueError("record has no published discriminator; re-infer the package")
                self.scalars["$.type"] = parts[1]
                # The selected physical form owns its envelope even when a
                # private subtype was folded out of the public kind. Do not
                # let unrelated optional fields exclude that original parent.
                if self.form in {"claude-record-data", "claude-record-attachment"}:
                    parent = "$.data" if self.form == "claude-record-data" else "$.attachment"
                    self.header_types[parent] = frozenset({"object"})
                if len(parts) == 3:
                    carriers = {
                        "claude-record-data": "$.data.type",
                        "claude-record-attachment": "$.attachment.type",
                        "claude-record-subtype": "$.subtype",
                    }
                    if self.form not in carriers:
                        raise ValueError("record subtype has no measured public carrier")
                    self.scalars[carriers[self.form]] = parts[2]
                # Non-message dispatcher fields remain original metadata;
                # no semantic content coordinate is declared for this form.
                self.content_path = ""
                if self.mode != "absent":
                    raise ValueError(
                        "non-message record contradicts its declared content envelope; re-infer the package"
                    )
        elif provider == "codex":
            # The selected public wire carrier excludes its competing root
            # fields before scalar draws; package-wide marginals can include
            # both forms because they describe multiple original streams.
            if self.form == "codex-envelope":
                self.absent.add("$.role")
            else:
                self.absent.add("$.payload")
            self.content_path = "$.payload.content" if self.form == "codex-envelope" else "$.content"
            role = self.kind.removesuffix("_message")
            if role in CODEX_PUBLIC_MESSAGE_ROLES:
                if self.form == "codex-envelope":
                    self.scalars.update(
                        {"$.type": "response_item", "$.payload.type": "message", "$.payload.role": role}
                    )
                elif self.form == "codex-flat":
                    self.scalars.update({"$.type": "message", "$.role": role})
                elif self.form in {"codex-role-message", "codex-role-message-null"}:
                    self.scalars["$.role"] = role
                    if self.form == "codex-role-message-null":
                        self.scalars["$.type"] = None
                    else:
                        self.absent.add("$.type")
                else:
                    raise ValueError("Codex message has an undeclared wire form")
            elif self.kind == "session_meta" and self.form in {
                "codex-flat-header",
                "codex-flat-header-null",
                "codex-flat-header-empty",
            }:
                self.tool_content_types["$.id"] = frozenset({"string"})
                self.tool_content_types["$.timestamp"] = frozenset({"string", "integer", "number"})
                if self.form == "codex-flat-header":
                    self.absent.add("$.type")
                else:
                    self.scalars["$.type"] = None if self.form.endswith("-null") else ""
            elif self.form == "codex-envelope":
                if self.kind in {"session_meta", "turn_context"}:
                    self.scalars["$.type"] = self.kind
                elif self.kind == "event_token_count":
                    self.scalars.update({"$.type": "event_msg", "$.payload.type": "token_count"})
                elif self.kind in {
                    "reasoning",
                    "function_call",
                    "function_call_output",
                    "custom_tool_call",
                    "custom_tool_call_output",
                }:
                    self.scalars.update({"$.type": "response_item", "$.payload.type": self.kind})
                else:
                    parts = self.kind.split(":")
                    if len(parts) not in {2, 3} or parts[0] != "record" or parts[1] == "other":
                        raise ValueError("Codex record has no published discriminator; re-infer the package")
                    self.scalars["$.type"] = parts[1]
                    if len(parts) == 3:
                        self.scalars["$.payload.type"] = parts[2]
            else:
                raise ValueError("Codex record has no measured public wire form")
        elif provider == "chatgpt" and self.kind == "record:empty":
            self.content_path = "$.message.content"
            if self.form == "chatgpt-message-null":
                self.scalars["$.message"] = None
            elif self.form == "chatgpt-message-absent":
                self.absent.add("$.message")
            else:
                raise ValueError("empty document node has no measured public form; re-infer the package")
            if self.mode != "absent":
                raise ValueError("empty document node contradicts its measured content")
        else:
            role = self.kind.removesuffix("_message")
            if self.form != "document-record" or role not in {"user", "assistant", "system", "developer", "tool"}:
                raise ValueError("document record has no published role/form; re-infer the package")
            paths = {"chatgpt": "$.message.author.role", "claude-ai": "$.sender", "gemini": "$.role"}
            if provider not in paths:
                raise ValueError("provider has no declared public record binding")
            if provider == "claude-ai" and role == "user":
                role = "human"
            elif provider == "gemini" and role == "assistant":
                role = "model"
            self.scalars[paths[provider]] = role
            self.content_path = "$.message.content" if provider == "chatgpt" else "$.content"
        for value in self.scalars.values():
            if value is not None and value not in published_kind_tokens():
                if value == "" and self.form == "codex-flat-header-empty":
                    continue
                raise ValueError("record plan contains an unpublished wire discriminator")
        modes = {"string": "string", "array": "array", "object": "object", "absent": None}
        if self.mode not in modes:
            raise ValueError("content has no declared representation; re-infer the package")
        self.content_type = modes[self.mode]
        if provider == "claude-code":
            self._bind_dispatch_fields(rng)
        if provider == "codex" and self.kind in {
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        }:
            self.bind_tool_operand("$.payload" if self.form == "codex-envelope" else "$", self.kind, rng)
        if self.mode == "object":
            kinds = ConditionalContentKinds(provider, self.kind, self.content, 1, owner).sample(rng)
            try:
                self.bind_content_kind(next(kinds), self.content_path, rng)
            finally:
                kinds.close()

    def bind_session_header(self, header_class: SchemaRecord, *, native_id: str | None, parent_id: str | None) -> None:
        """Bind original graph operands before walking the selected header schema.

        The profile owns presence and wire form only. Its fields must already
        exist in the chosen schema; the schema retains every value constraint.
        """
        if self.provider != "codex" or self.kind != "session_meta":
            raise ValueError("session operands require the original Codex header")
        if header_class.get("form") != self.form:
            raise ValueError("selected header class contradicts its original wire form")
        fields = header_class.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("first-header field classes are missing; re-infer the package")
        prefix = "$.payload." if self.form == "codex-envelope" else "$."
        pending: list[tuple[str, ...]] = [("forked_from_id", "parent_thread_id", "source")]
        while pending:
            for field in pending.pop():
                value = fields.get(field)
                if value not in {
                    "absent",
                    "null",
                    "empty",
                    "string",
                    "object",
                    "boolean",
                    "integer",
                    "number",
                    "array",
                }:
                    raise ValueError("first-header field class is unsupported")
                path = prefix + field
                if value == "absent":
                    self.absent.add(path)
                elif value in {"null", "empty"}:
                    self.graph_values[path] = None if value == "null" else ""
                else:
                    self.header_types[path] = frozenset({str(value)})
                if value == "object":
                    child = {
                        "source": "subagent",
                        "source.subagent": "thread_spawn",
                        "source.subagent.thread_spawn": "parent_thread_id",
                    }.get(field)
                    if child is not None:
                        pending.append((field + "." + child,))
        expected_fields = {
            path.removeprefix(prefix)
            for path in (*self.absent, *self.graph_values, *self.header_types)
            if path.startswith(prefix) and path.removeprefix(prefix) in fields
        }
        if expected_fields != set(fields):
            raise ValueError("first-header field class contains an undeclared coordinate")
        if native_id is not None:
            if not native_id:
                raise ValueError("generated session identity must be nonempty")
            self.graph_values[prefix + "id"] = native_id
        carrier = header_class.get("carrier")
        carriers = {
            "forked_from_id": "forked_from_id",
            "parent_thread_id": "parent_thread_id",
            "thread_spawn.parent_thread_id": "source.subagent.thread_spawn.parent_thread_id",
        }
        if parent_id is not None:
            if not isinstance(carrier, str) or carrier.partition(":")[2] not in carriers or not parent_id:
                raise ValueError("selected parent has no original header carrier")
            field = carriers[carrier.partition(":")[2]]
            if fields.get(field) != "string":
                raise ValueError("selected parent conflicts with original field presence")
            self.graph_values[prefix + field] = parent_id
        if not self.compatible(self.schema, "$"):
            raise ValueError("original header class has no declared schema branch; re-infer the package")

    def _bind_dispatch_fields(self, rng: random.Random) -> None:
        rows = self.content.get("dispatch_forms")
        if not isinstance(rows, list) or not rows:
            raise ValueError("original dispatcher field forms are missing; re-infer the package")
        weights: list[int] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("fields"), dict):
                raise ValueError("original dispatcher field form is malformed")
            weights.append(_count(row.get("count")))
        selected = rng.choices(rows, weights=weights, k=1)[0]
        assert isinstance(selected, dict)
        from polylogue.schemas.generation.workload_profiles import AgentChildCounts

        counts = AgentChildCounts.from_payload(selected.get("child_count_joint"))
        if not counts.bins:
            raise ValueError("original dispatcher child count joint is missing; re-infer the package")
        key = rng.choices(list(counts.bins), weights=[row[1] for row in counts.bins.values()], k=1)[0]
        point = rng.choice(counts.extrema[key])
        status = selected.get("children")
        if status not in {"no_dispatch", "matched", "missing_transcript", "ambiguous"}:
            raise ValueError("original dispatcher child association is malformed")
        if status == "ambiguous":
            raise ValueError("original dispatcher child association is ambiguous")
        if (
            status == "no_dispatch"
            and any(point)
            or status == "matched"
            and point[0] != 1
            or status == "missing_transcript"
            and point[0] != 0
            or point[0] > 1
            or point[2] > 1
            or status != "no_dispatch"
            and (not point[3] or point[4] not in {point[3], point[3] + 6})
        ):
            raise ValueError("original dispatcher child count contradicts its association")
        self.agent_children = str(status), point
        metadata = selected.get("metadata_class")
        if (point[2] and not isinstance(metadata, dict)) or (not point[2] and metadata is not None):
            raise ValueError("original metadata class contradicts selected physical count")
        self.agent_metadata = metadata if isinstance(metadata, dict) else None
        if status != "no_dispatch":
            self.tool_widths["$.toolUseResult.agentId"] = point[3]
            if point[0] or point[2]:
                self.agent_identity_prefix = point[4] == point[3]
        fields = selected["fields"]
        assert isinstance(fields, dict)
        roles = selected.get("roles")
        if not isinstance(roles, list) or len(roles) != 2 or any(not isinstance(role, str) for role in roles):
            raise ValueError("dispatcher class lacks its original role carriers")
        self._bind_role_carriers((str(roles[0]), str(roles[1])))
        required = {"toolUseResult"}
        if fields.get("toolUseResult") == "object":
            required.add("toolUseResult.agentId")
        if set(fields) != required:
            raise ValueError("dispatcher form contains an undeclared coordinate")
        for field, value in fields.items():
            path = "$." + field
            if value == "absent":
                self.absent.add(path)
            elif value in {"null", "empty"}:
                self.graph_values[path] = None if value == "null" else ""
            elif value in {"string", "object", "array", "boolean", "integer", "number"}:
                self.header_types[path] = frozenset({str(value)})
            else:
                raise ValueError("original dispatcher field form is unsupported")
        if not self.compatible(self.schema, "$"):
            raise ValueError(
                "original dispatcher form has no declared schema branch "
                f"at $ (kind={self.kind}, form={self.form}); re-infer the package"
            )

    def bind_agent_metadata(self, metadata: SchemaRecord, *, agent_id: str, agent_stem: str) -> None:
        from polylogue.core.claude_code_children import CLAUDE_AGENT_ID_FIELDS

        fields = metadata.get("fields")
        if not isinstance(fields, dict) or set(fields) != set(CLAUDE_AGENT_ID_FIELDS):
            raise ValueError("original metadata alias field classes are missing")
        for field, value in fields.items():
            path = "$." + field
            if value == "absent":
                self.absent.add(path)
            elif value in {"null", "empty"}:
                self.graph_values[path] = None if value == "null" else ""
            elif value in {"string", "boolean", "integer", "number", "object", "array"}:
                self.header_types[path] = frozenset({str(value)})
            else:
                raise ValueError("original metadata field class is unsupported")
        relation, carrier = metadata.get("relation"), metadata.get("carrier")
        if relation == "filename_only":
            if carrier is not None or any(value == "string" for value in fields.values()):
                raise ValueError("original filename-only metadata contradicts its field classes")
        elif relation in {"raw_agent", "stem"}:
            if carrier not in CLAUDE_AGENT_ID_FIELDS or fields.get(str(carrier)) != "string":
                raise ValueError("original metadata identity carrier contradicts its field classes")
            self.graph_values["$." + str(carrier)] = agent_id if relation == "raw_agent" else agent_stem
        elif relation == "independent":
            if carrier not in CLAUDE_AGENT_ID_FIELDS or fields.get(str(carrier)) != "string":
                raise ValueError("original independent metadata contradicts its identity carrier")
            self.string_exclusions["$." + str(carrier)] = frozenset({"", agent_id, agent_stem})
        else:
            raise ValueError("original metadata identity relation is unsupported")
        if not self.compatible(self.schema, "$"):
            raise ValueError("original metadata binding conflicts with selected schema support")

    def _bind_role_carriers(self, selected: tuple[str, str]) -> None:
        from polylogue.core.enums import ROLE_SYNONYMS

        rows = self.content.get("role_carriers")
        if not isinstance(rows, list) or not rows:
            raise ValueError("conditional public role carriers are missing; re-infer the package")
        vocabulary = frozenset(value for aliases in ROLE_SYNONYMS.values() for value in aliases)
        weights: dict[tuple[str, str], int] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("public role carrier is malformed")
            root, message = row.get("root"), row.get("message")
            for carrier in (root, message):
                if not isinstance(carrier, str) or (
                    carrier not in {"absent", "null", "unpublished"}
                    and not (carrier.startswith("role:") and carrier[5:] in vocabulary)
                ):
                    raise ValueError("public role carrier contains an undeclared value")
            assert isinstance(root, str) and isinstance(message, str)
            identity = root, message
            if identity in weights:
                raise ValueError("public role carrier occurs twice")
            weights[identity] = _count(row.get("count"))
        if selected not in weights:
            raise ValueError("dispatcher class contradicts its original role carriers")
        for path, carrier in zip(("$.role", "$.message.role"), selected, strict=True):
            if carrier == "absent":
                self.absent.add(path)
            elif carrier == "null":
                self.scalars[path] = None
            elif carrier == "unpublished":
                raise ValueError("role has no published wire value; re-infer the package")
            else:
                self.scalars[path] = carrier[5:]

    def bind_content_kind(self, kind: str, path: str, rng: random.Random) -> str:
        fields = self.content.get("block_discriminator_fields")
        field_name = _choose(fields.get("block:" + kind) if isinstance(fields, dict) else None, rng)
        if field_name not in {"type", "content_type"}:
            raise ValueError("content discriminator has no declared wire field")
        coordinate = path + "." + field_name
        forms_by_kind = self.content.get("block_body_field_forms")
        if isinstance(forms_by_kind, dict):
            forms = forms_by_kind.get("block:" + kind)
            if not isinstance(forms, list) or not forms:
                raise ValueError("original block body-field forms are missing; re-infer the package")
            weights: list[int] = []
            for row in forms:
                if not isinstance(row, dict) or not isinstance(row.get("fields"), list):
                    raise ValueError("original block body-field form is malformed")
                fields = row["fields"]
                if any(
                    not isinstance(field, str) or field not in PUBLIC_CONTENT_BODY_FIELDS for field in fields
                ) or len(fields) != len(set(fields)):
                    raise ValueError("original block body-field form is malformed")
                weights.append(_count(row.get("count")))
            selected = rng.choices(forms, weights=weights, k=1)[0]
            for field in PUBLIC_CONTENT_BODY_FIELDS:
                target = path + "." + field
                if field in selected["fields"]:
                    self.required_body_fields.add(target)
                else:
                    self.absent.add(target)
        self.scalars[coordinate] = kind
        if tool_identity_operand({"type": kind}) is not None:
            self.bind_tool_operand(path, kind, rng)
        return coordinate

    def _tool_field_schemas(self, coordinate: str) -> Generator[SchemaRecord, None, None]:
        pending = [(self.schema, "$")]
        while pending:
            schema, path = pending.pop()
            variants = schema.get("oneOf", schema.get("anyOf"))
            if isinstance(variants, list):
                pending.extend(
                    (variant, path)
                    for variant in variants
                    if isinstance(variant, dict) and self.compatible(variant, path)
                )
                continue
            if path == coordinate:
                yield schema
                continue
            if not coordinate.startswith(path + "."):
                continue
            field = coordinate[len(path) + 1 :].split(".", 1)[0]
            properties = schema.get("properties")
            child = properties.get(field) if isinstance(properties, dict) else None
            if field.endswith("[*]"):
                field = field[:-3]
                child = properties.get(field) if isinstance(properties, dict) else None
                arrays = [child] if isinstance(child, dict) else []
                array_path = path + "." + field
                while arrays:
                    array = arrays.pop()
                    variants = array.get("oneOf", array.get("anyOf"))
                    if isinstance(variants, list):
                        arrays.extend(
                            variant
                            for variant in variants
                            if isinstance(variant, dict) and self.compatible(variant, array_path)
                        )
                        continue
                    prefix = array.get("prefixItems")
                    if isinstance(prefix, list):
                        pending.extend(
                            (item, array_path + "[*]")
                            for item in prefix
                            if isinstance(item, dict) and self.compatible(item, array_path + "[*]")
                        )
                    item = array.get("items")
                    if isinstance(item, dict):
                        pending.append((item, array_path + "[*]"))
            elif isinstance(child, dict):
                pending.append((child, path + "." + field))

    def _new_tool_field_is_supported(self, coordinate: str, *, width: int | None = None) -> bool:
        for schema in self._tool_field_schemas(coordinate):
            labels = schema.get("x-polylogue-values")
            if not isinstance(labels, list) or not labels:
                if width is None or self._schema_accepts_width(schema, width):
                    return True
                continue
            if any(
                isinstance(value, str)
                and value
                and (width is None or len(value) == width)
                and self.owner.tool_identity_is_new(value)
                for value in labels
            ):
                return True
        return False

    def _pointer_width_candidates(self, path: str, row: SchemaRecord) -> dict[int, float] | None:
        joint = row.get("pointer_widths")
        if joint is None:
            if row.get("result_mode") == "string:truncated" and row.get("presence") == "string":
                raise ValueError("original pointer/identity width joint is missing; re-infer the package")
            widths = row.get("identity_width")
            if not isinstance(widths, dict):
                return None
            return {
                width: count
                for width, count in observed_length_support(widths).items()
                if width > 0
                and (
                    row.get("relation") != "new"
                    or self._new_tool_field_is_supported(path + "." + str(row["field"]), width=width)
                )
                and any(
                    self._schema_accepts_width(schema, width)
                    for schema in self._tool_field_schemas(path + "." + str(row["field"]))
                )
            }
        support: dict[int, float] = {}
        for (identity, _basename), weight in self._pointer_width_pairs(path, row).items():
            support[identity] = support.get(identity, 0.0) + weight
        return support

    def _pointer_width_pairs(self, path: str, row: SchemaRecord) -> dict[tuple[int, int], float]:
        joint = row.get("pointer_widths")
        if not isinstance(joint, dict) or set(joint) != {
            "identity_bucket",
            "basename_bucket",
            "relation",
            "basename_relation",
            "points",
        }:
            raise ValueError("original pointer/identity width joint is missing or malformed; re-infer the package")
        identity_bucket, basename_bucket, relation = (
            joint[key] for key in ("identity_bucket", "basename_bucket", "relation")
        )
        if (
            type(identity_bucket) is not int
            or type(basename_bucket) is not int
            or relation not in {"shorter", "equal", "one-more", "two-more", "longer"}
            or joint.get("basename_relation") not in {"own-direct", "other"}
        ):
            raise ValueError("pointer/identity width coordinates are malformed")
        measured_widths = FilenameWidthStats.from_payload(joint["points"])
        if not measured_widths.bins:
            raise ValueError("original pointer/identity width witnesses are missing; re-infer the package")
        field_widths: set[int] = set()
        for schema in self._tool_field_schemas(path + ".content"):
            for fmt, shape in measured_string_formats(schema).items():
                if fmt == "persisted-tool-output":
                    categories = shape.get("categories")
                    partitions = categories.values() if isinstance(categories, dict) else (shape,)
                    for partition in partitions:
                        payload = partition.get("pointer_shapes") if isinstance(partition, dict) else None
                        if isinstance(payload, list):
                            measured = PointerShapeStats.from_payload(payload)
                            field_widths.update(point[5] for group in measured.extrema.values() for point in group)
        pairs: dict[tuple[int, int], float] = {}
        for key, (_point, count) in measured_widths.bins.items():
            witnesses = measured_widths.extrema[key]
            for identity, basename in witnesses:
                if (
                    _bucket_index(float(identity)) != identity_bucket
                    or _bucket_index(float(basename)) != basename_bucket
                    or pointer_width_relation(identity, basename) != relation
                ):
                    raise ValueError("original width witness contradicts its declared relationship")
                if (
                    basename in field_widths
                    and (
                        row.get("relation") != "new"
                        or self._new_tool_field_is_supported(path + "." + str(row["field"]), width=identity)
                    )
                    and any(
                        self._schema_accepts_width(schema, identity)
                        for schema in self._tool_field_schemas(path + "." + str(row["field"]))
                    )
                ):
                    pairs[identity, basename] = count / len(witnesses)
        return pairs

    @staticmethod
    def _schema_accepts_width(schema: SchemaRecord, width: int) -> bool:
        labels = schema.get("x-polylogue-values")
        if isinstance(labels, list) and labels:
            return any(isinstance(value, str) and len(value) == width for value in labels)
        observed = schema.get("x-polylogue-observed-distribution")
        shapes = measured_string_formats(schema)
        partitions = list(shapes.values()) if shapes else [observed]
        return any(
            isinstance(shape, dict)
            and isinstance(shape.get("string_length"), dict)
            and any(
                _bucket_index(float(length)) == _bucket_index(float(width))
                for lengths in [shape.get("string_length")]
                if isinstance(lengths, dict)
                for length in observed_length_support(lengths)
            )
            for shape in partitions
        )

    def bind_tool_operand(self, path: str, kind: str, rng: random.Random) -> None:
        self.owner._array_owner(self.generation)
        operands = self.content.get("tool_operands")
        rows = operands.get(kind) if isinstance(operands, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("tool relation evidence is missing; re-infer the package")
        eligible: list[SchemaRecord] = []
        weights: list[int] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("tool relation evidence is malformed")
            event, field, presence, relation, prior_mode, prior_count = _tool_operand_coordinates(row, kind)
            weight = _count(row.get("count"))
            width_support = self._pointer_width_candidates(path, row)
            if width_support == {}:
                continue
            if relation in {"call", "result", "both"} and not self.owner.tool_identity_count(
                str(relation),
                prior_mode if self.provider == "claude-code" else None,
                prior_count if self.provider == "claude-code" else None,
                None if width_support is None else tuple(width_support),
            ):
                continue
            if relation == "new" and not self._new_tool_field_is_supported(path + "." + str(field)):
                continue
            eligible.append(row)
            weights.append(weight)
        if not eligible:
            raise ValueError("tool record has no realizable measured relation in its original scope")
        row = rng.choices(eligible, weights=weights, k=1)[0]
        selected_count = row["prior_result_count_bucket"]
        assert isinstance(selected_count, int) and not isinstance(selected_count, bool)
        event, field, presence, relation = (str(row[key]) for key in ("event", "field", "presence", "relation"))
        width_support = self._pointer_width_candidates(path, row)
        mode = row.get("result_mode")
        if not isinstance(mode, str):
            raise ValueError("tool result shape is missing; re-infer the package")
        if self.provider == "claude-code" and kind == "tool_result":
            shape, separator, disposition = mode.partition(":")
            if (
                not separator
                or shape not in {"absent", "null", "string", "array", "object", "other"}
                or disposition not in {"inline", "truncated"}
                or (disposition == "truncated" and shape != "string")
            ):
                raise ValueError("tool result shape contradicts its original content")
            content = path + ".content"
            if shape == "absent":
                self.absent.add(content)
            elif shape == "null":
                self.scalars[content] = None
            else:
                self.tool_content_types[content] = frozenset(
                    {"boolean", "integer", "number"} if shape == "other" else {shape}
                )
                if shape == "string":
                    self.tool_pointer_fields[content] = disposition == "truncated"
        elif mode:
            raise ValueError("opaque content shape belongs to its declared tool-result wire")
        coordinate = path + "." + field
        if relation == "missing":
            self.owner.observe_missing_tool_identity(event)
        if presence == "absent":
            self.absent.add(coordinate)
        elif presence == "null":
            self.scalars[coordinate] = None
        elif presence == "string" and relation == "missing":
            self.scalars[coordinate] = ""
        elif presence == "other":
            raise ValueError("tool identity has no declared string/null representation")
        else:
            value, schema = (
                self.owner.select_tool_identity(
                    relation,
                    rng,
                    str(row["prior_result_mode"]) if self.provider == "claude-code" else None,
                    selected_count if self.provider == "claude-code" else None,
                    None if width_support is None else tuple(width_support),
                )
                if relation != "new"
                else (None, None)
            )
            self.tool_fields[coordinate] = event, relation, presence, value, schema, mode
            if width_support is not None:
                self.tool_widths[coordinate] = (
                    len(value)
                    if value is not None
                    else rng.choices(list(width_support), weights=list(width_support.values()), k=1)[0]
                )
                widths = row["pointer_widths"]
                if widths is None:
                    return
                assert (
                    isinstance(widths, dict)
                    and isinstance(widths["basename_bucket"], int)
                    and isinstance(widths["relation"], str)
                )
                pairs = self._pointer_width_pairs(path, row)
                basenames = {
                    basename: weight
                    for (identity, basename), weight in pairs.items()
                    if identity == self.tool_widths[coordinate]
                }
                if not basenames:
                    raise ValueError("selected original identity has no measured filename width")
                basename = rng.choices(list(basenames), weights=list(basenames.values()), k=1)[0]
                self.tool_pointer_widths[path + ".content"] = coordinate, basename, widths["relation"]
                self.tool_pointer_relations[path + ".content"] = str(widths["basename_relation"])

    def clear_tool_operand(self, path: str) -> None:
        self.required_body_fields.difference_update(
            coordinate for coordinate in tuple(self.required_body_fields) if coordinate.startswith(path + ".")
        )
        for mapping in (
            self.tool_content_types,
            self.tool_pointer_fields,
            self.tool_widths,
            self.tool_pointer_widths,
            self.tool_pointer_relations,
        ):
            for coordinate in tuple(mapping):
                if coordinate.startswith(path + "."):
                    del mapping[coordinate]
        for coordinate in tuple(self.tool_fields):
            if coordinate.startswith(path + "."):
                del self.tool_fields[coordinate]
        for coordinate in tuple(self.absent):
            if coordinate.startswith(path + "."):
                self.absent.remove(coordinate)
        for coordinate in tuple(self.scalars):
            if coordinate.startswith(path + ".") and coordinate.rsplit(".", 1)[1] in {
                "id",
                "call_id",
                "tool_use_id",
                "tool_call_id",
            }:
                del self.scalars[coordinate]

    def tool_value(self, path: str, schema: SchemaRecord) -> GeneratedString | None:
        self.owner._array_owner(self.generation)
        selected = self.tool_fields.get(path)
        if selected is None or selected[3] is None:
            return None

        value, original_schema = selected[3], selected[4]
        # A borrowed spelling recipe is reusable under its exact selected
        # original field schema without hydrating its potentially huge text.
        supported = existing_value_is_supported(schema, value) if isinstance(value, str) else schema == original_schema
        if not supported:
            raise ValueError("tool reference conflicts with its selected field support")
        return value

    def observe_tool_field(self, path: str, value: object, schema: SchemaRecord) -> None:
        self.owner._array_owner(self.generation)
        selected = self.tool_fields.get(path)
        if selected is None:
            return
        if not isinstance(value, str | SampledText | JoinedText) or not len(value):
            raise ValueError("tool relation requires its measured nonempty identity")
        self.owner.observe_tool_identity(value, selected[0], selected[1], schema, selected[5])
        self.tool_fields[path] = selected[0], selected[1], selected[2], value, schema, selected[5]

    def requires_field(self, path: str) -> bool:
        return (
            any(
                key == path or key.startswith(path + ".")
                for key in (
                    *self.tool_fields,
                    *self.tool_content_types,
                    *self.header_types,
                    *self.graph_values,
                    *self.required_body_fields,
                )
            )
            or any(key == path or key.startswith(path + ".") for key in self.scalars)
            or (
                self.content_type is not None
                and (self.content_path == path or self.content_path.startswith(path + "."))
            )
        )

    def child_fields(self, path: str) -> set[str]:
        keys = {
            key[len(path) + 1 :].split(".", 1)[0]
            for key in (
                *self.scalars,
                *self.tool_fields,
                *self.tool_content_types,
                *self.header_types,
                *self.graph_values,
                *self.required_body_fields,
            )
            if key.startswith(path + ".")
        }
        if self.content_type is not None and self.content_path.startswith(path + "."):
            keys.add(self.content_path[len(path) + 1 :].split(".", 1)[0])
        return keys

    def compatible(self, schema: SchemaRecord, path: str) -> bool:
        """Filter the existing branch sampler by the selected public operand."""
        from polylogue.schemas.synthetic.classification import _schema_types

        constrained = self.requires_field(path)
        if not constrained:
            return True
        for keyword in ("anyOf", "oneOf"):
            variants = schema.get(keyword)
            if isinstance(variants, list) and variants:
                return any(isinstance(item, dict) and self.compatible(item, path) for item in variants)
        if path in self.graph_values:
            return existing_value_is_supported(schema, self.graph_values[path])
        if path in self.string_exclusions:
            labels = schema.get("x-polylogue-values")
            if (
                isinstance(labels, list)
                and labels
                and not any(isinstance(label, str) and label not in self.string_exclusions[path] for label in labels)
            ):
                return False
        if path in self.header_types:
            types = _schema_types(schema)
            if not types or not types.intersection(self.header_types[path]):
                return False
            if "object" not in self.header_types[path]:
                return True
        if path in self.tool_content_types:
            types = _schema_types(schema)
            if types and not types.intersection(self.tool_content_types[path]):
                return False
            if path in self.tool_pointer_fields:
                from polylogue.schemas.synthetic.observed_values import measured_string_formats

                formats = measured_string_formats(schema)
                pointer = self.tool_pointer_fields[path]
                labels = schema.get("x-polylogue-values")
                if isinstance(labels, list) and labels:
                    from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_POINTER_PATTERN

                    return any(
                        isinstance(value, str)
                        and (PERSISTED_OUTPUT_POINTER_PATTERN.search(value) is not None) == pointer
                        for value in labels
                    )
                if formats and not any((fmt == "persisted-tool-output") == pointer for fmt in formats):
                    return False
                if not formats and (schema.get("x-polylogue-format") == "persisted-tool-output") != pointer:
                    return False
            return True
        if path in self.tool_fields:
            types = _schema_types(schema)
            if types and "string" not in types:
                return False
            try:
                self.tool_value(path, schema)
            except ValueError:
                return False
            return True
        expected = self.scalars.get(path)
        if path in self.scalars:
            return existing_value_is_supported(schema, expected)
        types = _schema_types(schema)
        if path == self.content_path:
            if self.content_type is None or (types and self.content_type not in types):
                return False
            if self.content_type != "object":
                return True
        if types and "object" not in types:
            return False
        properties = schema.get("properties")
        declared = properties if isinstance(properties, dict) else {}
        if path == "$" and self.form.startswith("codex-flat-header") and not {"id", "timestamp"}.issubset(declared):
            return False
        if path == "$" and self.form.startswith("codex-role-message") and "role" not in declared:
            return False
        additional = schema.get("additionalProperties", True)
        required = schema.get("required")
        if isinstance(required, list) and any(path + "." + str(key) in self.absent for key in required):
            return False
        for key in self.child_fields(path):
            coordinate = path + "." + key
            if (
                any(
                    item == coordinate or item.startswith(coordinate + ".")
                    for item in (*self.header_types, *self.graph_values)
                )
                and key not in declared
            ):
                return False
            child = declared.get(key, additional)
            if child is False or (not isinstance(child, dict) and child is not True):
                return False
            if not self.compatible(child if isinstance(child, dict) else {}, path + "." + key):
                return False
        return True


class RecordScopeKinds:
    """Measured public record path conditioned on the existing scope count."""

    def __init__(
        self,
        structure: SchemaRecord,
        count: int,
        owner: ForeignKeyGraph,
        *,
        provider: str | None = None,
        schema: SchemaRecord | None = None,
        message_count: int | None = None,
    ) -> None:
        if message_count is not None and provider != "codex":
            raise ValueError("selected message count requires the original Codex record grammar")
        if type(count) is not int or count < 0:
            raise ValueError("scope count must be a nonnegative exact integer")
        if not count:
            self.path = _MeasuredPath(
                {},
                {},
                {},
                0,
                owner,
                count_effects={} if message_count is not None else None,
                selected_count=message_count,
            )
            return
        starts = {(kind, "record"): weight for kind, weight in _weights(structure.get("initial_kinds")).items()}
        endings = {(kind, "record"): weight for kind, weight in _weights(structure.get("terminal_kinds")).items()}
        rows = structure.get("transitions")
        if not isinstance(rows, list):
            raise ValueError("public record transitions are missing; re-infer the package")
        edges: dict[ContentState, dict[ContentState, int]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("from"), str) or not isinstance(row.get("to"), str):
                raise ValueError("public record transition is malformed")
            first, second = row["from"], row["to"]
            assert isinstance(first, str) and isinstance(second, str)
            destinations = edges.setdefault((first, "record"), {})
            target = second, "record"
            if target in destinations:
                raise ValueError("public record transition occurs twice")
            destinations[target] = _count(row.get("count"))
        resource = _single_call_resource(structure, provider, schema)
        self.path = _MeasuredPath(
            starts,
            endings,
            edges,
            count,
            owner,
            resource_effects=None if resource is None else resource[0],
            count_effects=None
            if message_count is None
            else {
                state: int(
                    state[0].endswith("_message") and state[0].removesuffix("_message") in CODEX_PUBLIC_MESSAGE_ROLES
                )
                for state in set(starts)
                | set(endings)
                | set(edges)
                | {destination for destinations in edges.values() for destination in destinations}
            },
            selected_count=message_count,
            available_resource=None
            if resource is None
            else lambda: owner.tool_identity_count("call", "", 0, (resource[1],)),
        )

    def sample(self, rng: random.Random) -> Generator[str, None, None]:
        yield from self.path.sample(rng)


def _single_call_resource(
    structure: SchemaRecord, provider: str | None, schema: SchemaRecord | None
) -> tuple[dict[ContentState, tuple[int, int]], int] | None:
    """Project a proved fungible call resource, never approximate joint history.

    One new call creates one call-only identity; its first result consumes it.
    This projection applies only when every original form/operand of a record
    has that same effect and width. Repeated results, competing widths, finite
    literal ID pools and multi-block records retain their original joint owner.
    """
    if provider != "claude-code" or schema is None:
        return None
    pending = [schema]
    while pending:
        node = pending.pop()
        properties = node.get("properties")
        if isinstance(properties, dict):
            for field, child in properties.items():
                if isinstance(child, dict):
                    if field in {"id", "call_id", "tool_use_id", "tool_call_id"} and child.get("x-polylogue-values"):
                        return None
                    pending.append(child)
        for key in ("oneOf", "anyOf", "prefixItems"):
            rows = node.get(key)
            if isinstance(rows, list):
                pending.extend(row for row in rows if isinstance(row, dict))
        item = node.get("items")
        if isinstance(item, dict):
            pending.append(item)
    contents = structure.get("content_blocks_by_record_kind")
    if not isinstance(contents, dict):
        return None
    effects: dict[ContentState, tuple[int, int]] = {}
    widths: set[int] = set()
    for record_kind, forms in contents.items():
        if not isinstance(forms, dict):
            return None
        record_effects: set[tuple[int, int]] = set()
        for modes in forms.values():
            if not isinstance(modes, dict):
                return None
            for mode, profile in modes.items():
                if not isinstance(profile, dict):
                    return None
                operands = profile.get("tool_operands")
                if not isinstance(operands, dict) or not operands:
                    record_effects.add((0, 0))
                    continue
                if mode not in {"array", "object"}:
                    return None
                if mode == "array":
                    count = profile.get("array_block_count")
                    if not isinstance(count, dict) or set(observed_length_support(count)) != {1}:
                        return None
                rows_by_kind = list(operands.items())
                if len(rows_by_kind) != 1:
                    return None
                tool_kind, rows = rows_by_kind[0]
                if not isinstance(rows, list) or not rows:
                    return None
                expected = (0, 1) if tool_kind == "tool_use" else (1, -1) if tool_kind == "tool_result" else None
                if expected is None:
                    return None
                endings = _state_rows(profile, "terminal_states")
                if any(state[0] != "block:" + tool_kind for state in endings):
                    return None
                for row in rows:
                    if not isinstance(row, dict):
                        return None
                    event, _field, presence, relation, prior_mode, prior_count = _tool_operand_coordinates(
                        row, tool_kind
                    )
                    if (
                        relation != ("new" if expected[1] == 1 else "call")
                        or event != ("call" if expected[1] == 1 else "result")
                        or presence != "string"
                        or prior_mode != ""
                        or prior_count != 0
                    ):
                        return None
                    width = row.get("identity_width")
                    if not isinstance(width, dict):
                        return None
                    support = observed_length_support(width)
                    if len(support) != 1 or min(support) <= 0:
                        return None
                    widths.update(support)
                record_effects.add(expected)
        if len(record_effects) != 1:
            return None
        effects[(record_kind, "record")] = next(iter(record_effects))
    if len(widths) != 1:
        return None
    return effects, next(iter(widths))
