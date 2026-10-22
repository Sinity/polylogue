use super::*;

#[test]
fn a_large_policy_stays_on_disk_and_only_current_candidates_return() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('large','session:codex:removed',1,1)", []).unwrap();
    conn.execute_batch("WITH RECURSIVE n(value) AS (VALUES(0) UNION ALL SELECT value+1 FROM n WHERE value<99999)
        UPDATE excision SET message_ids_json=(SELECT json_group_array('codex:message:'||value) FROM n)").unwrap();
    let cache = Refused::load(&conn).unwrap();
    let conn = cache.0.as_ref().unwrap();
    let bytes: i64 = conn
        .query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(bytes > 1 << 20);
    assert_eq!(conn.path(), Some(""));
    let messages: Vec<_> = (0..1500)
        .map(|n| {
            if n % 3 == 0 {
                format!("codex:message:{}", n * 53)
            } else {
                format!("absent:{n}")
            }
        })
        .collect();
    let hits = cache
        .matching(&["codex:removed".into(), "codex:absent".into()], &messages)
        .unwrap();
    assert_eq!(hits.sessions, HashSet::from(["codex:removed".into()]));
    assert_eq!(hits.messages.len(), 500);
    for n in (0..1500).step_by(3) {
        assert!(hits.messages.contains(&format!("codex:message:{}", n * 53)));
    }
    drop(cache);
}

#[test]
fn empty_history_stays_lazy_and_opaque_ids_keep_separate_namespaces() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    assert!(Refused::load(&conn).unwrap().0.is_none());
    let id = "codex:opaque:\0☺'\"";
    conn.execute("INSERT INTO excision(id,scope,message_ids_json,refused_json,created_at,commit_seq) VALUES ('remove','input:source',?1,?2,1,1)", rusqlite::params![serde_json::to_string(&[id,id]).unwrap(),r#"{"sessions":["codex:session"]}"#]).unwrap();
    let cache = Refused::load(&conn).unwrap();
    let candidates = [id.into(), "codex:session".into(), id.into()];
    let hits = cache.matching(&candidates, &candidates).unwrap();
    assert_eq!(hits.sessions, HashSet::from(["codex:session".into()]));
    assert_eq!(hits.messages, HashSet::from([id.into()]));
    cache
        .0
        .as_ref()
        .unwrap()
        .execute("DROP TABLE messages", [])
        .unwrap();
    assert!(cache.matching(&[], &candidates).is_err());
}

#[test]
fn duplicate_fields_and_malformed_suffixes_do_not_publish_partial_lookups() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('remove','input:source','{}',1,1)", []).unwrap();
    for policy in [
        r#"{"sessions":["codex:one"],"sessions":[]}"#,
        r#"{"sessions":["codex:one"]} trailing"#,
        r#"{"sessions":["codex:one",null]}"#,
    ] {
        conn.execute("UPDATE excision SET refused_json=?1", [policy])
            .unwrap();
        assert!(Refused::load(&conn).is_err());
    }
}

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
    let hits = refused.matching(&["claude-code:gone".into()], &[]).unwrap();
    assert!(hits.sessions.contains("claude-code:gone"));
    assert!(hits.messages.is_empty());
}

#[test]
fn selected_sessions_are_canonical_and_malformed_policies_refuse_loading() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,message_ids_json,refused_json,created_at,commit_seq) VALUES ('remove','input:source','[]',?1,1,1)", [r#"{"sessions":["codex:opaque:session"],"publication":{"ignored":"for this read"}}"#]).unwrap();
    let policy = Refused::load(&conn).unwrap();
    assert!(
        policy
            .matching(&["codex:opaque:session".into()], &[])
            .unwrap()
            .sessions
            .contains("codex:opaque:session")
    );
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
    assert!(Refused::load(&conn).unwrap().is_empty());
}

#[cfg(feature = "faults")]
#[test]
fn an_id_insertion_failure_preserves_the_store_error_and_can_rebuild() {
    use crate::fault::{self, Action};
    fault::clear();
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,message_ids_json,refused_json,created_at,commit_seq) VALUES ('remove','input:source','[\"codex:seed-message\"]','{\"sessions\":[\"codex:seed-session\"]}',1,1)", []).unwrap();
    for id in ["codex:seed-session", "codex:seed-message"] {
        fault::arm("exclusion_cache.insert", id, Action::Fail, 0, 1);
        let error = Refused::load(&conn).unwrap_err();
        assert!(
            matches!(error, StoreError::Invalid(ref text) if text.contains("exclusion_cache.insert") && text.contains(id)),
            "{error}"
        );
        let cache = Refused::load(&conn).unwrap();
        let hits = cache
            .matching(
                &["codex:seed-session".into()],
                &["codex:seed-message".into()],
            )
            .unwrap();
        assert_eq!(hits.sessions.len(), 1);
        assert_eq!(hits.messages.len(), 1);
    }
    fault::clear();
}
