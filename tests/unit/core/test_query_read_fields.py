"""Read/output descriptor contracts for archive surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import signature

import click

from polylogue.api.archive import PolylogueArchiveMixin
from polylogue.archive.query.read_fields import (
    READ_FIELD_DESCRIPTORS,
    read_field_descriptor_map,
    read_field_names_for_surface,
)
from polylogue.cli.query_verbs import (
    bulk_export_verb,
    delete_verb,
    list_verb,
    messages_verb,
    open_verb,
    raw_verb,
    show_verb,
    stats_verb,
)


def _click_parameter_names(command: click.Command) -> frozenset[str]:
    return frozenset(param.name for param in command.params if param.name)


def test_read_field_descriptors_cover_core_read_surfaces() -> None:
    descriptors = read_field_descriptor_map()

    assert descriptors["conversation_id"].kind == "identity"
    assert descriptors["message_role"].kind == "projection"
    assert descriptors["message_type"].kind == "projection"
    assert descriptors["limit"].kind == "pagination"
    assert descriptors["offset"].kind == "pagination"
    assert descriptors["output_format"].kind == "format"
    assert descriptors["content_projection"].kind == "projection"
    assert len(descriptors) == len(READ_FIELD_DESCRIPTORS)


def test_cli_query_verb_parameters_have_read_descriptors() -> None:
    covered = read_field_names_for_surface("cli")
    expected_query_only: Mapping[str, frozenset[str]] = {
        "list": frozenset(),
        "stats": frozenset(),
        "show": frozenset(),
        "open": frozenset(),
        "bulk-export": frozenset(),
        "delete": frozenset(),
        "messages": frozenset(),
        "raw": frozenset(),
    }

    for command in (
        list_verb,
        stats_verb,
        show_verb,
        open_verb,
        bulk_export_verb,
        delete_verb,
        messages_verb,
        raw_verb,
    ):
        missing = _click_parameter_names(command) - covered
        assert command.name is not None
        assert missing == expected_query_only[command.name]


def test_read_field_surface_name_sets_are_intentional() -> None:
    assert {"conversation_id", "message_role", "message_type", "limit", "offset"}.issubset(
        read_field_names_for_surface("storage")
    )
    assert {"conversation_id", "message_role", "message_type", "limit", "offset", "group_by"}.issubset(
        read_field_names_for_surface("mcp")
    )
    assert {"content_projection", "conversation_id", "limit", "offset", "group_by"}.issubset(
        read_field_names_for_surface("api")
    )


def test_api_point_read_method_parameters_have_read_descriptors() -> None:
    covered = read_field_names_for_surface("api")
    method_names = (
        "get_conversation",
        "get_conversations",
        "get_messages_paginated",
        "get_raw_artifacts_for_conversation",
        "resume_brief",
    )

    for method_name in method_names:
        params = set(signature(getattr(PolylogueArchiveMixin, method_name)).parameters)
        params.discard("self")
        assert params - covered == set()
