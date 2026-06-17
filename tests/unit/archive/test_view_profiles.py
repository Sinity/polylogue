"""Session read-view profile registry tests (#1997)."""

from __future__ import annotations

from polylogue.archive.viewport.profiles import READ_VIEW_PROFILE_BY_ID, READ_VIEW_PROFILES, read_view_choices


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
