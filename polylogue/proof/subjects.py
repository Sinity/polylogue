"""Subject discovery for the verification catalog."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

import click

from polylogue.cli.command_inventory import CommandPath, iter_command_paths
from polylogue.lib.json import JSONDocument, json_document, json_document_list, require_json_value
from polylogue.lib.provider_capabilities import iter_provider_capabilities
from polylogue.operations import build_declared_operation_catalog
from polylogue.proof.models import SourceSpan, SubjectRef
from polylogue.schemas.packages import SchemaVersionPackage
from polylogue.schemas.runtime_registry import SCHEMA_DIR, SchemaRegistry

SELECTED_SCHEMA_ANNOTATIONS: tuple[str, ...] = (
    "x-polylogue-values",
    "x-polylogue-foreign-keys",
    "x-polylogue-mutually-exclusive",
)
SELECTED_JSON_COMMANDS: tuple[tuple[str, ...], ...] = (("doctor",), ("tags",))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_COMPOSITE_KEYWORDS = ("anyOf", "oneOf", "allOf")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _command_callback(command: click.Command) -> Callable[..., object] | None:
    callback = getattr(command, "callback", None)
    return callback if callable(callback) else None


def _command_source_span(command: click.Command, *, fallback_symbol: str) -> SourceSpan:
    callback = _command_callback(command)
    if callback is None:
        return SourceSpan(path="polylogue/cli/click_app.py", symbol=fallback_symbol)

    unwrapped = inspect.unwrap(callback)
    source_file = inspect.getsourcefile(unwrapped)
    line: int | None = None
    if source_file is not None:
        try:
            _, line = inspect.getsourcelines(unwrapped)
        except (OSError, TypeError):
            line = None
    symbol = f"{getattr(unwrapped, '__module__', '')}.{getattr(unwrapped, '__qualname__', fallback_symbol)}".strip(".")
    return SourceSpan(
        path=_repo_relative(Path(source_file)) if source_file is not None else "polylogue/cli/click_app.py",
        line=line,
        symbol=symbol or fallback_symbol,
    )


def command_subjects(root_command: click.Command | None = None) -> tuple[SubjectRef, ...]:
    """Compile visible Click commands into proof subjects."""
    if root_command is None:
        from polylogue.cli.click_app import cli

        root_command = cli

    subjects: list[SubjectRef] = []
    for command_path in iter_command_paths(root_command, include_root=True):
        subjects.append(_command_subject(command_path))
    return tuple(sorted(subjects, key=lambda subject: subject.id))


def json_command_subjects(
    root_command: click.Command | None = None,
    *,
    selected_paths: Iterable[tuple[str, ...]] = SELECTED_JSON_COMMANDS,
) -> tuple[SubjectRef, ...]:
    """Compile the selected JSON-capable commands into proof subjects."""
    if root_command is None:
        from polylogue.cli.click_app import cli

        root_command = cli

    selected = set(selected_paths)
    subjects: list[SubjectRef] = []
    for command_path in iter_command_paths(root_command, include_root=False):
        if command_path.path not in selected:
            continue
        command_id = f"polylogue {' '.join(command_path.path)} --json"
        attrs = _json_document(
            {
                "command_path": list(command_path.path),
                "display_name": f"{command_path.display_name} --json",
                "json_args": ["--plain", *command_path.path, "--json"],
            }
        )
        subjects.append(
            SubjectRef(
                kind="cli.json_command",
                id=command_id,
                attrs=attrs,
                source_span=_command_source_span(command_path.command, fallback_symbol=command_id),
            )
        )
    return tuple(sorted(subjects, key=lambda subject: subject.id))


def query_law_subjects() -> tuple[SubjectRef, ...]:
    """Compile stable archive-query laws used by semantic evidence runners."""
    return (
        SubjectRef(
            kind="archive.query_law",
            id="archive.query_law.provider_filter.codex",
            attrs=_json_document(
                {
                    "provider": "codex",
                    "laws": [
                        "provider_subset",
                        "provider_count_matches_list",
                        "equivalent_provider_filter_constructions",
                    ],
                }
            ),
            source_span=SourceSpan(path="polylogue/proof/subjects.py", symbol="query_law_subjects"),
        ),
    )


def provider_capability_subjects() -> tuple[SubjectRef, ...]:
    """Compile typed provider capability metadata into proof subjects."""
    subjects = [
        SubjectRef(
            kind="provider.capability",
            id=f"provider.capability.{capability.provider.value}",
            attrs=capability.to_payload(),
            source_span=SourceSpan(
                path="polylogue/lib/provider_capabilities.py",
                symbol=capability.source_symbol,
            ),
        )
        for capability in iter_provider_capabilities()
    ]
    return tuple(sorted(subjects, key=lambda subject: subject.id))


def operation_spec_subjects() -> tuple[SubjectRef, ...]:
    """Compile declared operation specifications into proof subjects."""
    subjects = [
        SubjectRef(
            kind="operation.spec",
            id=f"operation.spec.{operation.name}",
            attrs=operation.to_dict(),
            source_span=SourceSpan(path="polylogue/operations/specs.py", symbol=operation.name),
        )
        for operation in build_declared_operation_catalog().specs
    ]
    return tuple(sorted(subjects, key=lambda subject: subject.id))


def workflow_claim_subjects() -> tuple[SubjectRef, ...]:
    """Compile durable workflow claims that are not coupled to GitHub runtime state."""
    return (
        SubjectRef(
            kind="workflow.claim",
            id="workflow.claim.generated_surfaces_current",
            attrs=_json_document(
                {
                    "claim_family": "generated-surfaces",
                    "required_command": "devtools render-all --check",
                    "source_changes": [
                        "CLI help",
                        "devtools command catalog",
                        "proof catalog",
                        "quality registry",
                        "agent memory",
                    ],
                }
            ),
            source_span=SourceSpan(path="devtools/generated_surfaces.py", symbol="GENERATED_SURFACES"),
        ),
        SubjectRef(
            kind="workflow.claim",
            id="workflow.claim.pr_verification_recorded",
            attrs=_json_document(
                {
                    "claim_family": "pr-body",
                    "required_sections": ["Summary", "Problem", "Solution", "Verification"],
                    "required_linking": ["Ref #<issue>", "Closes #<issue>"],
                }
            ),
            source_span=SourceSpan(path="CONTRIBUTING.md", symbol="Pull Requests"),
        ),
    )


def _command_subject(command_path: CommandPath) -> SubjectRef:
    is_root = not command_path.path
    command_id = "polylogue" if is_root else f"polylogue {' '.join(command_path.path)}"
    display_name = "polylogue" if is_root else command_path.display_name
    attrs = _json_document(
        {
            "command_path": list(command_path.path),
            "display_name": display_name,
            "help_exercise_name": "help-main" if is_root else command_path.help_exercise_name,
            "root": is_root,
        }
    )
    return SubjectRef(
        kind="cli.command",
        id=command_id,
        attrs=attrs,
        source_span=_command_source_span(command_path.command, fallback_symbol=command_id),
    )


def schema_annotation_subjects(
    registry: SchemaRegistry | None = None,
    *,
    annotation_keys: Iterable[str] = SELECTED_SCHEMA_ANNOTATIONS,
) -> tuple[SubjectRef, ...]:
    """Compile selected packaged-schema annotations into proof subjects."""
    schema_registry = registry or SchemaRegistry(storage_root=SCHEMA_DIR)
    selected = tuple(annotation_keys)
    subjects: list[SubjectRef] = []
    for provider in schema_registry.list_providers():
        for version in schema_registry.list_versions(provider):
            package = schema_registry.get_package(provider, version=version)
            if package is None:
                continue
            for element in package.elements:
                schema = schema_registry.get_element_schema(
                    provider, version=version, element_kind=element.element_kind
                )
                if schema is None:
                    continue
                schema_source = _schema_source_path(provider, package, element.schema_file)
                subjects.extend(
                    _annotation_subjects_for_schema(
                        provider=provider,
                        version=version,
                        element_kind=element.element_kind,
                        schema=schema,
                        schema_source=schema_source,
                        annotation_keys=selected,
                    )
                )
    return tuple(sorted(_dedupe(subjects), key=lambda subject: subject.id))


def build_catalog_subjects() -> tuple[SubjectRef, ...]:
    """Compile all subjects included in the first proof-catalog slice."""
    return (
        *command_subjects(),
        *json_command_subjects(),
        *query_law_subjects(),
        *provider_capability_subjects(),
        *operation_spec_subjects(),
        *schema_annotation_subjects(),
        *workflow_claim_subjects(),
    )


def _dedupe(subjects: Iterable[SubjectRef]) -> Iterator[SubjectRef]:
    seen: set[str] = set()
    for subject in subjects:
        if subject.id in seen:
            continue
        seen.add(subject.id)
        yield subject


def _annotation_subjects_for_schema(
    *,
    provider: str,
    version: str,
    element_kind: str,
    schema: Mapping[str, object],
    schema_source: str,
    annotation_keys: tuple[str, ...],
) -> Iterator[SubjectRef]:
    if "x-polylogue-values" in annotation_keys:
        yield from _value_annotation_subjects(
            provider=provider,
            version=version,
            element_kind=element_kind,
            schema=schema,
            schema_source=schema_source,
        )
    if "x-polylogue-foreign-keys" in annotation_keys:
        yield from _root_record_annotation_subjects(
            provider=provider,
            version=version,
            element_kind=element_kind,
            schema=schema,
            schema_source=schema_source,
            annotation="x-polylogue-foreign-keys",
            id_fields=("source", "target"),
        )
    if "x-polylogue-mutually-exclusive" in annotation_keys:
        yield from _root_record_annotation_subjects(
            provider=provider,
            version=version,
            element_kind=element_kind,
            schema=schema,
            schema_source=schema_source,
            annotation="x-polylogue-mutually-exclusive",
            id_fields=("parent", "fields"),
        )


def _value_annotation_subjects(
    *,
    provider: str,
    version: str,
    element_kind: str,
    schema: Mapping[str, object],
    schema_source: str,
) -> Iterator[SubjectRef]:
    for path, node in _walk_schema_nodes(json_document(schema)):
        values = node.get("x-polylogue-values")
        if not isinstance(values, list):
            continue
        value_list = [require_json_value(value, context="x-polylogue-values item") for value in values]
        if not value_list:
            continue
        attrs = _base_schema_attrs(
            provider=provider,
            version=version,
            element_kind=element_kind,
            annotation="x-polylogue-values",
            schema_path=path,
        )
        attrs["values"] = value_list
        attrs["value_count"] = len(value_list)
        yield _schema_subject(
            provider=provider,
            version=version,
            element_kind=element_kind,
            annotation="x-polylogue-values",
            suffix=path,
            attrs=attrs,
            source_span=SourceSpan(path=schema_source, line=1, symbol=f"{path}.x-polylogue-values"),
        )


def _root_record_annotation_subjects(
    *,
    provider: str,
    version: str,
    element_kind: str,
    schema: Mapping[str, object],
    schema_source: str,
    annotation: str,
    id_fields: tuple[str, ...],
) -> Iterator[SubjectRef]:
    for index, record in enumerate(json_document_list(schema.get(annotation))):
        attrs = _base_schema_attrs(
            provider=provider,
            version=version,
            element_kind=element_kind,
            annotation=annotation,
            schema_path="$",
        )
        attrs["record_index"] = index
        for key, value in record.items():
            attrs[key] = require_json_value(value, context=f"{annotation}.{key}")
        suffix = ":".join(_record_id_part(record.get(field)) for field in id_fields)
        yield _schema_subject(
            provider=provider,
            version=version,
            element_kind=element_kind,
            annotation=annotation,
            suffix=f"{index}:{suffix}",
            attrs=attrs,
            source_span=SourceSpan(path=schema_source, line=1, symbol=f"$.{annotation}[{index}]"),
        )


def _schema_subject(
    *,
    provider: str,
    version: str,
    element_kind: str,
    annotation: str,
    suffix: str,
    attrs: JSONDocument,
    source_span: SourceSpan,
) -> SubjectRef:
    normalized_suffix = suffix.replace(" ", "_")
    return SubjectRef(
        kind="schema.annotation",
        id=f"{provider}:{version}:{element_kind}:{annotation}:{normalized_suffix}",
        attrs=attrs,
        source_span=source_span,
    )


def _base_schema_attrs(
    *,
    provider: str,
    version: str,
    element_kind: str,
    annotation: str,
    schema_path: str,
) -> JSONDocument:
    return _json_document(
        {
            "provider": provider,
            "package_version": version,
            "element_kind": element_kind,
            "annotation": annotation,
            "schema_path": schema_path,
        }
    )


def _walk_schema_nodes(schema: JSONDocument, path: str = "$") -> Iterator[tuple[str, JSONDocument]]:
    yield path, schema

    properties = json_document(schema.get("properties"))
    for name, child in properties.items():
        child_node = json_document(child)
        if child_node:
            yield from _walk_schema_nodes(child_node, f"{path}.{name}")

    additional_properties = json_document(schema.get("additionalProperties"))
    if additional_properties:
        yield from _walk_schema_nodes(additional_properties, f"{path}.*")

    items = json_document(schema.get("items"))
    if items:
        yield from _walk_schema_nodes(items, f"{path}[*]")

    for keyword in _SCHEMA_COMPOSITE_KEYWORDS:
        for child in json_document_list(schema.get(keyword)):
            yield from _walk_schema_nodes(child, path)


def _record_id_part(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _schema_source_path(provider: str, package: SchemaVersionPackage, schema_file: str | None) -> str:
    if schema_file is None:
        return f"polylogue/schemas/providers/{provider}/versions/{package.version}/elements"
    return _repo_relative(SCHEMA_DIR / provider / "versions" / package.version / "elements" / schema_file)


def _json_document(items: dict[str, object]) -> JSONDocument:
    return {key: require_json_value(value, context=key) for key, value in items.items()}


__all__ = [
    "SELECTED_SCHEMA_ANNOTATIONS",
    "SELECTED_JSON_COMMANDS",
    "build_catalog_subjects",
    "command_subjects",
    "json_command_subjects",
    "operation_spec_subjects",
    "provider_capability_subjects",
    "query_law_subjects",
    "schema_annotation_subjects",
    "workflow_claim_subjects",
]
