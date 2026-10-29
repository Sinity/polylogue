use super::*;

#[test]
fn assertion_input_refuses_caller_authorship_and_seen_hash() {
    let value = serde_json::json!({"kind":"note","value":"A statement","links":[{"role":"about","reference":"message:codex:message"}]});
    let parsed: NewAssertion = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(parsed.links[0].role, LinkRole::About);
    for field in ["author", "via", "machine", "created_at"] {
        let mut with_actor = value.clone();
        with_actor[field] = "operator".into();
        assert!(
            serde_json::from_value::<NewAssertion>(with_actor).is_err(),
            "{field}"
        );
    }
    let mut with_hash = value;
    with_hash["links"][0]["seen_hash"] = "00".repeat(32).into();
    assert!(serde_json::from_value::<NewAssertion>(with_hash).is_err());
}
#[test]
fn assertion_revision_patch_distinguishes_inherit_clear_and_replace() {
    let inherit: AssertionPatch = serde_json::from_str("{}").unwrap();
    assert_eq!(inherit.key, Change::Keep);
    let change:AssertionPatch=serde_json::from_value(serde_json::json!({"key":{"change":"clear"},"value":{"change":"set","value":"Replacement"}})).unwrap();
    assert_eq!(change.key, Change::Clear);
    assert_eq!(change.value, Change::Set("Replacement".into()));
    assert_eq!(change.valid_until, Change::Keep);
}
