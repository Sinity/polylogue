"""Declared neutral fixture operands, inferred before the shared field draws.

These are curated demo populations, never measurements of an operator archive.
The ordinary schema/profile inference and lazy driver consume them unchanged.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

from polylogue.core.json import JSONDocument, JSONValue, json_document, require_json_value
from polylogue.schemas.generation.models import _PackageAccumulator
from polylogue.schemas.generation.observation_journal import ObservationJournal
from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
from polylogue.schemas.generation.workload_profiles import build_package_workload_profile, workload_profile_identity
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.record_structure import record_kind
from polylogue.schemas.synthetic.demo_themes import _DEMO_THEMES
from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS


def _tool_blocks(*, failed: bool, read: bool) -> list[JSONDocument]:
    identity = "neutral-read" if read else "neutral-bash"
    return [
        {
            "type": "tool_use",
            "id": identity,
            "name": "Read" if read else "Bash",
            "input": {"file_path": "fixtures/generated.jsonl"}
            if read
            else {"command": "python -m pytest tests/generated/test_fixture.py -q"},
        },
        {
            "type": "tool_result",
            "tool_use_id": identity,
            "content": "FileNotFoundError: neutral fixture not found"
            if read and failed
            else "read 4096 bytes"
            if read
            else "AssertionError: generated fixture mismatch"
            if failed
            else "1 passed",
            **({"is_error": failed} if not read or failed else {}),
        },
    ]


def _records(provider: str, *, failed: bool) -> Iterator[JSONDocument]:
    if provider == "codex":
        yield {
            "type": "session_meta",
            "payload": {"id": str(uuid.UUID(int=100, version=4)), "timestamp": "2025-06-01T00:00:00Z"},
        }
    for index in range(4):
        role = "assistant" if index % 2 else "user"
        blocks: list[JSONDocument] = (
            _tool_blocks(failed=failed and (provider == "codex" or index != 1), read=index == 1)
            if role == "assistant"
            else [{"type": "text", "text": "Please inspect the fixture."}]
        )
        if provider == "claude-code":
            yield {
                "type": role,
                "uuid": str(uuid.UUID(int=index + 1, version=4)),
                "parentUuid": str(uuid.UUID(int=index, version=4)) if index else None,
                "sessionId": str(uuid.UUID(int=100, version=4)),
                "timestamp": "2025-06-01T00:00:00Z",
                "message": {"role": role, "content": require_json_value(blocks)},
            }
        else:
            for block in blocks:
                if block["type"] == "text":
                    block["type"] = "output_text" if role == "assistant" else "input_text"
            yield {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": str(uuid.UUID(int=index + 1, version=4)),
                    "role": role,
                    "content": require_json_value(blocks),
                },
            }
    if failed:
        text = "The pytest command failed with exit code 1; I will inspect the assertion before retrying."
        if provider == "claude-code":
            yield {
                "type": "assistant",
                "uuid": str(uuid.UUID(int=5, version=4)),
                "parentUuid": str(uuid.UUID(int=4, version=4)),
                "sessionId": str(uuid.UUID(int=100, version=4)),
                "timestamp": "2025-06-01T00:00:01Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            }
        else:
            yield {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": str(uuid.UUID(int=5, version=4)),
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            }


def _literal_fixture_fields(schema: JSONDocument, samples: list[JSONDocument]) -> None:
    """Only explicitly authored prose/tool operands receive literal labels.

    Native identities remain the existing relation/semantic owner's draws.
    This is declaration of fixture values, not recovery of private vocabulary.
    """
    pending: list[tuple[JSONDocument, list[object], str | None]] = [(schema, list(samples), None)]
    while pending:
        node, values, field = pending.pop()
        if node.get("type") == "string" and field in {
            "text",
            "content",
            "name",
            "command",
            "file_path",
            "mimeType",
            "data",
            "parts",
        }:
            labels = sorted({value for value in values if isinstance(value, str)})
            if labels:
                node["x-polylogue-values"] = require_json_value(labels)
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                if isinstance(child, dict):
                    pending.append(
                        (child, [value[name] for value in values if isinstance(value, dict) and name in value], name)
                    )
        items = node.get("items")
        if isinstance(items, dict):
            pending.append((items, [item for value in values if isinstance(value, list) for item in value], field))


@cache
def authored_schema_selection(
    provider: str, style: str, *, session_native_ids: tuple[str, ...] = (), theme_index: int = 0
) -> SyntheticSchemaSelection:
    """Infer the finite declared population once; select it before field draws."""
    attachment_operands: tuple[JSONDocument, ...] | None = None
    document_records: list[JSONDocument] | None = None
    samples: list[JSONDocument]
    scopes: list[list[JSONDocument]]
    if style == "demo" and provider == "chatgpt":
        theme = _DEMO_THEMES[theme_index]
        document_records = []
        for index in range(2 * len(theme.user_turns)):
            identity = str(uuid.UUID(int=index + 1, version=4))
            role = "assistant" if index % 2 else "user"
            text = (theme.assistant_turns if index % 2 else theme.user_turns)[index // 2]
            document_records.append(
                {
                    "id": identity,
                    "parent": str(uuid.UUID(int=index, version=4)) if index else None,
                    "children": [str(uuid.UUID(int=index + 2, version=4))]
                    if index + 1 < 2 * len(theme.user_turns)
                    else [],
                    "message": {
                        "id": identity,
                        "author": {"role": role},
                        "content": {"content_type": "text", "parts": [text]},
                        "create_time": 1748736000.0 + index * 60,
                    },
                }
            )
        samples = [
            {
                "id": native_id,
                "title": theme.title,
                "mapping": {str(record["id"]): record for record in document_records[:size]},
                "current_node": document_records[size - 1]["id"],
            }
            for native_id in (session_native_ids or (str(uuid.UUID(int=100, version=4)),))
            for size in range(1, len(document_records) + 1)
        ]
        scopes = [[sample] for sample in samples]
        element = "session_document"
    elif style in {"tool-heavy", "demo-tool-heavy"}:
        if provider not in {"claude-code", "codex"}:
            raise ValueError("authored tool fixtures require a declared tool wire route")
        samples = list(_records(provider, failed=style == "demo-tool-heavy"))
        scopes = [samples[:size] for size in range(2 if provider == "codex" else 1, len(samples) + 1)]
        element = "session_record_stream"
    elif style == "demo-attachments" and provider == "gemini":
        payload = b"demo attachment bytes for Polylogue fixture coverage\n"
        user: JSONDocument = {
            "role": "user",
            "text": "Please inspect the attached fixture note.",
            "inlineFile": {"mimeType": "text/plain", "data": base64.b64encode(payload).decode("ascii")},
        }
        model: JSONDocument = {"role": "model", "text": "I inspected the fixture note."}
        followup: JSONDocument = {"role": "user", "text": "Please summarize the fixture note."}
        chunks = [user, model, followup, model]
        attachment_operands = tuple(chunks)
        samples = [
            {"id": str(uuid.UUID(int=100, version=4)), "chunkedPrompt": {"chunks": require_json_value(chunks[:size])}}
            for size in range(1, len(chunks) + 1)
        ]
        scopes = [[sample] for sample in samples]
        element = "session_document"
    else:
        raise ValueError("style has no declared authored fixture population")
    schema = generate_schema_from_samples(samples, max_stats_samples=0, provider=provider)
    _literal_fixture_fields(schema, samples)
    record_population: list[JSONDocument] = (
        document_records
        if document_records is not None
        else samples
        if provider != "gemini"
        else list(attachment_operands or ())
    )
    branches: list[JSONValue] = []
    for index, record in enumerate(record_population):
        branch_population: list[JSONDocument] = [record]
        if provider == "claude-code":
            for native_id in session_native_ids:
                if not native_id:
                    raise ValueError("authored native identity must be a nonempty string")
                branch_population.append({**record, "sessionId": native_id})
        if provider == "chatgpt" and index == 0:
            branch_population.append({**record, "parent": record_population[-1]["id"]})
        if provider == "claude-code" and index == 0:
            # The authored cycle declares this user record both at the root
            # and after a preceding cycle. Infer both original link operands.
            continuation = {**record, "parentUuid": record_population[-1]["uuid"]}
            branch_population.append(continuation)
        if provider == "codex" and index == 0:
            for native_id in session_native_ids:
                if not native_id:
                    raise ValueError("authored native identity must be a nonempty string")
                branch_population.append(
                    {"type": "session_meta", "payload": {**json_document(record["payload"]), "id": native_id}}
                )
                branch_population.append(
                    {
                        "type": "session_meta",
                        "payload": {**json_document(record["payload"]), "id": native_id, "forked_from_id": native_id},
                    }
                )
            continuation = {
                "type": "session_meta",
                "payload": {**json_document(record["payload"]), "forked_from_id": str(uuid.UUID(int=101, version=4))},
            }
            branch_population.append(continuation)
        branch = generate_schema_from_samples(branch_population, max_stats_samples=0, provider=provider)
        _literal_fixture_fields(branch, branch_population)
        branch["x-polylogue-authored-record-branch"] = index
        if provider in {"claude-code", "codex"} and record_kind(provider, record) != "session_meta":
            envelope = json_document(record["message" if provider == "claude-code" else "payload"])
            content = envelope["content"]
            assert isinstance(content, list)
            block_branches: list[JSONValue] = []
            for block_value in content:
                block = json_document(block_value)
                block_schema = generate_schema_from_samples([block], max_stats_samples=0, provider=provider)
                _literal_fixture_fields(block_schema, [block])
                json_document(json_document(block_schema["properties"])["type"])["x-polylogue-values"] = [block["type"]]
                block_branches.append(block_schema)
            carrier = json_document(branch["properties"])
            carrier = json_document(
                json_document(carrier["message" if provider == "claude-code" else "payload"])["properties"]
            )
            # The curated pair's original public ordering is declared in the
            # schema, not appended after unrelated operator field draws.
            json_document(carrier["content"])["prefixItems"] = block_branches
            json_document(carrier["content"])["items"] = False
        elif provider == "gemini":
            json_document(json_document(branch["properties"])["role"])["x-polylogue-values"] = [record["role"]]
        branches.append(branch)
    if provider == "chatgpt":
        mapping = json_document(json_document(schema["properties"])["mapping"])
        mapping.pop("properties", None)
        mapping["additionalProperties"] = {"oneOf": branches}
    elif provider == "gemini":
        prompt = json_document(json_document(schema["properties"])["chunkedPrompt"])
        json_document(json_document(prompt["properties"])["chunks"])["items"] = {"oneOf": branches}
    else:
        schema = {"oneOf": branches}
    schema["x-polylogue-authored-fixture"] = style
    with (
        TemporaryDirectory(prefix="authored-schema-") as temporary,
        ObservationJournal.create(root=Path(temporary) / "journal") as journal,
    ):
        for index, scope in enumerate(scopes):
            unit = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=scope,
                    artifact_kind=element,
                    bundle_scope=f"authored-{index}",
                )
            )
            journal.assign_profile_family(unit, "authored-neutral")
        profile = build_package_workload_profile(
            provider=provider,
            version="authored-fixture-v1",
            package=_PackageAccumulator(
                provider=provider,
                anchor_family_id="authored-neutral",
                anchor_kind=element,
                memberships=journal.memberships(profile_family_id="authored-neutral"),
            ),
            element_schemas={element: schema},
            privacy_policy="standard",
        )
        branch_structures: list[JSONValue] = []
        for index, record in enumerate(record_population):
            family = f"authored-branch-{index}"
            branch_sample: JSONDocument = (
                {"mapping": {str(record["id"]): record}}
                if provider == "chatgpt"
                else {"chunkedPrompt": {"chunks": [record]}}
                if provider == "gemini"
                else record
            )
            unit = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=[branch_sample],
                    artifact_kind="session_record_stream",
                    bundle_scope=family,
                )
            )
            journal.assign_profile_family(unit, family)
            branch_profile = build_package_workload_profile(
                provider=provider,
                version="authored-fixture-v1",
                package=_PackageAccumulator(
                    provider=provider,
                    anchor_family_id=family,
                    anchor_kind="session_record_stream",
                    memberships=journal.memberships(profile_family_id=family),
                ),
                element_schemas={"session_record_stream": json_document(branches[index])},
                privacy_policy="standard",
            )
            branch_structures.append(
                json_document(json_document(branch_profile["elements"])["session_record_stream"])["record_structure"]
            )
    profile["authored_record_structures"] = branch_structures
    profile["profile_kind"] = "authored-fixture"
    profile["privacy_classification"] = "declared-neutral-fixture-operands"
    profile["authored_style"] = style
    if provider == "chatgpt":
        profile["authored_theme_index"] = theme_index
    profile["authored_session_header"] = provider == "codex"
    profile["authored_record_cycle"] = [
        {"kind": record_kind(provider, record), "branch": index} for index, record in enumerate(record_population)
    ]
    profile["profile_id"] = workload_profile_identity(profile)
    return SyntheticSchemaSelection(
        provider=provider,
        package_version="authored-fixture-v1",
        element_kind=element,
        schema=schema,
        wire_format=PROVIDER_WIRE_FORMATS[provider],
        workload_profile=profile,
        element_schemas={element: schema},
    )


def authored_record_schema(profile: Mapping[str, object] | None, schema: JSONDocument, index: int) -> JSONDocument:
    """Select an explicit authored branch before the ordinary field walker."""
    if profile is None or profile.get("profile_kind") != "authored-fixture":
        return schema
    cycle = profile.get("authored_record_cycle")
    if not isinstance(cycle, list) or not cycle:
        raise ValueError("authored fixture has no declared original record order")
    ordinal = (
        (0 if index == 0 else 1 + (index - 1) % (len(cycle) - 1))
        if profile.get("authored_session_header") is True
        else index % len(cycle)
    )
    row = json_document(cycle[ordinal])
    branch = row.get("branch")
    variants = schema.get("oneOf")
    if type(branch) is not int or not isinstance(variants, list) or not 0 <= branch < len(variants):
        raise ValueError("authored record has no original schema branch")
    return json_document(variants[branch])
