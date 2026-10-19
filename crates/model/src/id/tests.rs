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
