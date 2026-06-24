"""Session read-view profile registry tests (#1997)."""

from __future__ import annotations

from polylogue.archive.viewport.profiles import (
    READ_VIEW_HTTP_CAPABILITIES,
    READ_VIEW_PROFILE_BY_ID,
    READ_VIEW_PROFILES,
    read_view_choices,
    read_view_http_capability_payloads,
    read_view_http_choices,
    read_view_http_format_choices,
    read_view_http_query_params,
)


def test_read_view_choices_are_profile_backed_and_ordered() -> None:
    choices = read_view_choices()

    assert choices == tuple(profile.view_id for profile in READ_VIEW_PROFILES)
    assert set(choices) == set(READ_VIEW_PROFILE_BY_ID)


def test_every_read_view_profile_declares_contract_fields() -> None:
    for profile in READ_VIEW_PROFILES:
        assert profile.view_id
        assert profile.label
        assert profile.owner.startswith("polylogue.")
        assert profile.purpose
        assert profile.input_scope
        assert profile.included_kinds
        assert profile.formats
        assert profile.privacy_policy
        assert profile.degraded_states


def test_successor_handoff_profiles_are_evidence_or_caveat_bearing() -> None:
    handoff_profiles = [profile for profile in READ_VIEW_PROFILES if profile.successor_handoff]

    assert {profile.view_id for profile in handoff_profiles} == {"context", "context-pack", "recovery"}
    for profile in handoff_profiles:
        assert profile.lossiness in {"derived", "summarized"}
        assert profile.evidence_policy in {"required", "optional"}


def test_http_read_view_capabilities_are_profile_backed_and_payload_driving() -> None:
    choices = read_view_http_choices()

    assert choices == tuple(READ_VIEW_HTTP_CAPABILITIES)
    assert set(choices).issubset(READ_VIEW_PROFILE_BY_ID)
    assert "summary" not in choices
    assert read_view_http_format_choices() == ("json", "markdown")
    assert "report" in read_view_http_query_params()
    assert "max_messages" in read_view_http_query_params()

    payloads = read_view_http_capability_payloads()
    assert set(payloads) == set(choices)
    assert payloads["recovery"]["formats"] == ["json", "markdown"]
    assert payloads["context-pack"]["query_params"] == [
        "include_messages",
        "max_messages",
        "max_text",
        "no_redact",
    ]
