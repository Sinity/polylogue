use super::*;

#[test]
fn targets_parse_and_text_is_labelled_by_hash() {
    assert!(matches!(
        Scope::parse("session:claude-code:abc").unwrap(),
        Scope::Session(_)
    ));
    assert!(matches!(
        Scope::parse("claude-code:abc").unwrap(),
        Scope::Session(_)
    ));
    assert!(matches!(
        Scope::parse("input:0193").unwrap(),
        Scope::Input(_)
    ));
    let t = Scope::parse("text:sk-secret").unwrap();
    assert!(!t.label().contains("sk-secret"));
    assert!(Scope::parse("nonsense").is_err());
}

#[test]
fn finds_written_and_json_escaped_forms() {
    let n = needles("a\"b");
    assert_eq!(n.len(), 2);
    let hay = br#"{"t":"a\"b"} a"b"#;
    assert_eq!(find_all(hay, &n).len(), 2);
}

mod deliveries;
mod retained;

#[test]
fn malformed_removed_message_ids_are_not_an_empty_refusal_set() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,message_ids_json,created_at,commit_seq) VALUES ('removed','session:claude-code:gone','invalid',1,1)",[]).unwrap();
    assert!(Refused::load(&conn).is_err());
    conn.execute(
        "UPDATE excision SET message_ids_json='[]' WHERE id='removed'",
        [],
    )
    .unwrap();
    let refused = Refused::load(&conn).unwrap();
    assert!(refused.sessions.contains("claude-code:gone"));
    assert!(refused.messages.is_empty());
}
