use super::*;

#[test]
fn session_id_splits_at_first_colon_and_keeps_colons_in_native() {
    let id = SessionId::parse("chatgpt:a:b:c").unwrap();
    assert_eq!(id.origin().as_str(), "chatgpt");
    assert_eq!(id.native().as_str(), "a:b:c");
    assert_eq!(id.to_string(), "chatgpt:a:b:c");
}

#[test]
fn ids_round_trip_through_json() {
    let id = MessageId::parse("claude-code:7f0c-1").unwrap();
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "\"claude-code:7f0c-1\"");
    assert_eq!(serde_json::from_str::<MessageId>(&json).unwrap(), id);
}

#[test]
fn bad_ids_are_refused() {
    assert!(Origin::new("Claude").is_err());
    assert!(Origin::new("").is_err());
    assert!(SessionId::parse("no-colon").is_err());
    assert!(SessionId::parse("codex:").is_err());
}

#[test]
fn blob_hash_round_trips_hex() {
    let h = BlobHash::of(b"abc");
    assert_eq!(BlobHash::parse(&h.to_string()).unwrap(), h);
    assert!(BlobHash::parse("zz").is_err());
}

#[test]
fn native_run_ids_preserve_one_uuid_spelling() {
    let text = "019a2f20-1234-7000-8000-0123456789ab";
    let id = RunId::parse(text).unwrap();
    assert_eq!(id.as_str(), text);
    assert_eq!(id.to_string(), text);
    assert_eq!(
        serde_json::from_value::<RunId>(serde_json::json!(text)).unwrap(),
        id
    );
    assert_eq!(serde_json::to_value(&id).unwrap(), serde_json::json!(text));
    for invalid in [
        "",
        "019a2f201234700080000123456789ab",
        "019A2F20-1234-7000-8000-0123456789AB",
        "urn:uuid:019a2f20-1234-7000-8000-0123456789ab",
    ] {
        assert!(matches!(RunId::parse(invalid), Err(IdError::BadRun(_))));
        assert!(serde_json::from_value::<RunId>(serde_json::json!(invalid)).is_err());
    }
}

#[test]
fn canonical_v4_run_ids_roundtrip_as_opaque_identities() {
    let run = RunId::parse("7f564f3c-0d2b-4d43-91ae-c07dd4e521d9").unwrap();
    assert_eq!(
        serde_json::from_value::<RunId>(serde_json::to_value(&run).unwrap()).unwrap(),
        run
    );
}
