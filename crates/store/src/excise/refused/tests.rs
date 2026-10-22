use super::*;

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

#[test]
fn selected_sessions_are_canonical_and_malformed_policies_refuse_loading() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,message_ids_json,refused_json,created_at,commit_seq) VALUES ('remove','input:source','[]',?1,1,1)", [r#"{"sessions":["codex:opaque:session"],"publication":{"ignored":"for this read"}}"#]).unwrap();
    let policy = Refused::load(&conn).unwrap();
    assert!(policy.sessions.contains("codex:opaque:session"));
    for value in [
        r#"{"sessions":[1]}"#,
        r#"{"sessions":[""]}"#,
        r#"{"sessions":"codex:wrong"}"#,
        r#"{"sessions":null}"#,
        r#"[["codex:wrong"]]"#,
        "not JSON",
    ] {
        conn.execute("UPDATE excision SET refused_json=?1", [value])
            .unwrap();
        assert!(Refused::load(&conn).is_err(), "{value}");
        assert!(super::super::RefusedJson::decode(value).is_err(), "{value}");
    }
    // Old input-scoped records have no recoverable selected-session list.
    conn.execute("UPDATE excision SET refused_json='{}'", [])
        .unwrap();
    assert!(
        matches!(Refused::load(&conn), Err(StoreError::Invalid(message)) if message.contains("no retained session selection"))
    );
    conn.execute("UPDATE excision SET refused_json='{\"sessions\":[]}'", [])
        .unwrap();
    assert!(Refused::load(&conn).unwrap().sessions.is_empty());
}
