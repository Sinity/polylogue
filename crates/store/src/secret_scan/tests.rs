use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn coverage_survives_resume_and_invalidates_on_rules_or_content() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let planted = "AKIAABCDEFGHIJKLMNOP";
    for i in 0..5 {
        let id = format!("test:s{i}");
        let pk = fixture
            .session(SessionFacts {
                id: &id,
                ..Default::default()
            })
            .unwrap();
        fixture
            .message(
                pk,
                MessageFacts {
                    id: &format!("m{i}"),
                    role: "system",
                    text: Some(if i == 4 { "ordinary prose" } else { planted }),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let reads = fixture.reads().unwrap();
    let mut index =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(None)
            .unwrap(),
        5
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(Some("other"))
            .unwrap(),
        0
    );
    assert!(!scan(&mut index, "test:missing").unwrap().found);
    let first = scan(&mut index, "test:s0").unwrap();
    assert_eq!(first.blocks_scanned, 1);
    assert_eq!(first.findings.len(), 1);
    assert_eq!(first.findings[0].byte_length, planted.len());
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(None)
            .unwrap(),
        4
    );
    let mut findings = first.findings.len();
    while let Some(id) = reads
        .reader()
        .unwrap()
        .pending_secret_scan(None, None)
        .unwrap()
    {
        findings += scan(&mut index, &id).unwrap().findings.len();
    }
    assert_eq!(findings, 4);
    let stored: String = index
        .query_row("SELECT group_concat(output) FROM secret_scan", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(!stored.contains(planted));
    index.execute("UPDATE secret_scan SET version='different' WHERE session_pk=(SELECT pk FROM session WHERE id='test:s0')", []).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(None)
            .unwrap(),
        1
    );
    scan(&mut index, "test:s0").unwrap();
    let text_id = fixture.queue_search_text(planted).unwrap();
    index.execute("UPDATE block SET text_id=?1 WHERE session_pk=(SELECT pk FROM session WHERE id='test:s4')", [text_id]).unwrap();
    index
        .execute(
            "UPDATE session SET changed=changed+1 WHERE id='test:s4'",
            [],
        )
        .unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .pending_secret_scan(None, None)
            .unwrap()
            .as_deref(),
        Some("test:s4")
    );
    assert_eq!(scan(&mut index, "test:s4").unwrap().findings.len(), 1);
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(None)
            .unwrap(),
        0
    );
}

#[test]
fn modes_are_exclusive_and_limits_are_explicit() {
    for (session, all, status, origin, limit) in [
        (None, false, false, None, None),
        (Some("x"), true, false, None, None),
        (Some("x"), false, true, None, None),
        (None, true, true, None, None),
        (Some("x"), false, false, Some("test"), None),
        (None, false, true, None, Some(1)),
        (None, true, false, None, Some(0)),
    ] {
        assert!(validate(session, all, status, origin, limit).is_err());
    }
    assert!(validate(None, true, false, Some("test"), Some(2)).is_ok());
}

#[test]
fn structured_strings_and_keys_are_scanned_without_leaking_into_paths() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let pk = fixture
        .session(SessionFacts {
            id: "test:structured",
            ..Default::default()
        })
        .unwrap();
    let message = fixture
        .message(
            pk,
            MessageFacts {
                id: "m",
                role: "user",
                text: Some("normal"),
                ..Default::default()
            },
        )
        .unwrap();
    let secret = "AKIAABCDEFGHIJKLMNOP";
    let generic = "password=Ab3dEf7hIj9kLm2nOp4qRs6t";
    let body = crate::encoding::pack_json(&serde_json::json!({
        "type":"tool_call", "call_id":null, "tool":"test", "input":{
            "nested":[{(generic): format!("{} {secret}","ordinary ".repeat(4000))}]
        }
    }))
    .unwrap();
    let mut index =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    index
        .execute(
            "UPDATE block SET body=?1, text_id=NULL, kind='tool_call' WHERE message_pk=?2",
            params![body, message],
        )
        .unwrap();
    let result = scan(&mut index, "test:structured").unwrap();
    assert_eq!(result.findings.len(), 2);
    let encoded = serde_json::to_string(&result).unwrap();
    assert!(!encoded.contains(secret));
    assert!(!encoded.contains(generic));
    assert!(result.findings.iter().any(|f| f.start == 36001));
}

#[test]
fn session_events_cover_system_instructions_and_user_context() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let pk = fixture
        .session(SessionFacts {
            id: "test:context",
            ..Default::default()
        })
        .unwrap();
    fixture.event(pk, "turn_context", Some(1)).unwrap();
    let secret = "AKIAABCDEFGHIJKLMNOP";
    let data = crate::encoding::pack_json(&serde_json::json!({
        "system_prompt":secret,"user_context":{"text":secret}
    }))
    .unwrap();
    let mut index =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    index
        .execute(
            "UPDATE event SET data=?1 WHERE session_pk=?2",
            params![data, pk],
        )
        .unwrap();
    let result = scan(&mut index, "test:context").unwrap();
    assert_eq!(result.events_scanned, 1);
    assert_eq!(result.findings.len(), 2);
    assert!(
        result
            .findings
            .iter()
            .all(|f| f.target.starts_with("event:") && f.block.is_none())
    );
    assert!(!serde_json::to_string(&result).unwrap().contains(secret));
}

#[test]
fn unreadable_content_stays_pending_until_a_complete_scan() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let pk = fixture
        .session(SessionFacts {
            id: "test:unreadable",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            pk,
            MessageFacts {
                id: "m",
                role: "user",
                text: Some("AKIAABCDEFGHIJKLMNOP"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture.event(pk, "turn_context", Some(1)).unwrap();
    let reads = fixture.reads().unwrap();
    let mut index =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    index
        .execute("UPDATE event SET data=x'00' WHERE session_pk=?1", [pk])
        .unwrap();
    assert!(scan(&mut index, "test:unreadable").is_err());
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(None)
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM secret_scan", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let data = crate::encoding::pack_json(&serde_json::json!({})).unwrap();
    index
        .execute(
            "UPDATE event SET data=?1 WHERE session_pk=?2",
            params![data, pk],
        )
        .unwrap();
    assert_eq!(
        scan(&mut index, "test:unreadable").unwrap().findings.len(),
        1
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .secret_scan_pending_count(None)
            .unwrap(),
        0
    );
}
