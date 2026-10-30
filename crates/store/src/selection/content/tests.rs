use super::*;
use crate::fixture::writer::{WriterArchive, call, message, prose, sid};
use polylogue_model::{Block, Link, LinkKind, Message, Role};
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

fn options() -> Options {
    Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::time(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        command_words: vec![],
        page_size: Some(1),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    }
}
fn fingerprint(archive: &WriterArchive, target: &str, parents: bool) -> String {
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader.prepare_selection(target, &options()).unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| {
            if matches!(
                context.action,
                AuthAction::Read {
                    table_name: "block",
                    column_name: "body" | "data"
                } | AuthAction::Read {
                    table_name: "text" | "message_version",
                    column_name: "body"
                }
            ) {
                Authorization::Deny
            } else {
                Authorization::Allow
            }
        }))
        .unwrap();
    reader
        .selection_content_fingerprint(&selection, parents, None)
        .unwrap()
}
fn observe_snapshot(archive: &mut WriterArchive, session: &str, message: Message) {
    observe_snapshot_at(archive, session, 0, message);
}
fn observe_snapshot_at(archive: &mut WriterArchive, session: &str, from: u64, message: Message) {
    let id = message.id.clone();
    let expected = message.content_hash().unwrap();
    archive.write_snapshot_at(session, from, vec![message]);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .message_with_projection(&id)
            .unwrap()
            .unwrap()
            .1
            .source_version,
        expected,
        "the fixture must observe the actual shown snapshot version"
    );
}
fn prefix_rows(reader: &Reader<'_>, call: &Message, anchor: &Message) -> Vec<serde_json::Value> {
    reader
        .query(
            "SELECT m.*,shown.version_hash,p.layout FROM message m
        LEFT JOIN message_source shown ON shown.message_pk=m.pk AND shown.shown=1
        LEFT JOIN message_projection p ON p.version_hash=shown.version_hash
        WHERE m.id=?1 OR m.id=?2 ORDER BY m.seq,m.record_position,m.id",
            &[
                SqlValue::Text(call.id.to_string()),
                SqlValue::Text(anchor.id.to_string()),
            ],
        )
        .unwrap()
}
#[test]
fn off_page_pair_versions_bind_message_and_range_but_only_the_selected_block() {
    let mut archive = WriterArchive::new();
    let mut one = call(
        "s",
        1,
        1,
        "synthetic",
        None,
        serde_json::json!({"synthetic":true}),
        Ok("one"),
    );
    let two = call(
        "s",
        2,
        2,
        "synthetic",
        None,
        serde_json::json!({"synthetic":true}),
        Ok("two"),
    );
    let mut selected = one.remove(0);
    selected.blocks.push(two[0].blocks[0].clone());
    let mut changed = two[1].clone();
    archive.write("s", vec![selected.clone(), one.remove(0), changed.clone()]);
    let targets = [
        format!("message:{}", selected.id),
        format!("message:{}..{}", selected.id, selected.id),
        format!("block:{}#0", selected.id),
        format!("block:{}#1", selected.id),
    ];
    let before: Vec<_> = targets
        .iter()
        .map(|target| fingerprint(&archive, target, false))
        .collect();
    if let Block::ToolResult { output, .. } = &mut changed.blocks[0] {
        *output = Some("changed outside the selected message".into());
    }
    observe_snapshot(&mut archive, "s", changed);
    let after: Vec<_> = targets
        .iter()
        .map(|target| fingerprint(&archive, target, false))
        .collect();
    assert_ne!(after[0], before[0]);
    assert_ne!(after[1], before[1]);
    assert_eq!(after[2], before[2]);
    assert_ne!(after[3], before[3]);
}
#[test]
fn a_selected_parent_call_binds_its_result_beyond_the_inherited_prefix() {
    let mut archive = WriterArchive::new();
    let mut pair = call(
        "parent",
        1,
        1,
        "synthetic",
        None,
        serde_json::json!({"synthetic":true}),
        Ok("before"),
    );
    let call = pair.remove(0);
    let mut result = pair.remove(0);
    let anchor = message(
        "anchor",
        "parent",
        Role::Assistant,
        2,
        vec![Block::Text {
            text: "fork here".into(),
        }],
    );
    archive.write("parent", vec![call.clone(), anchor.clone(), result.clone()]);
    archive.links.push(Link {
        kind: LinkKind::Fork,
        parent: sid("parent"),
        child: sid("child"),
        at: Some(anchor.id.clone()),
        call_id: None,
        evidence: "explicit fork".into(),
    });
    archive.write("child", prose("child", 1, 3, "own available tail"));
    let (own_before, rows_before) = {
        let pool = crate::ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        let selected = reader.select_messages(&sid("child"), true).unwrap();
        let mut ids = Vec::new();
        selected
            .visit_available_tool_message_ids(|id| {
                ids.push(id.to_owned());
                Ok(())
            })
            .unwrap();
        assert_eq!(ids, [call.id.to_string()]);
        assert!(!ids.contains(&result.id.to_string()));
        (
            selected.fingerprint().unwrap(),
            prefix_rows(&reader, &call, &anchor),
        )
    };
    assert_eq!(rows_before.len(), 2);
    let before = fingerprint(&archive, "session:claude-code:child", true);
    let result_before = result.content_hash().unwrap();
    if let Block::ToolResult { output, .. } = &mut result.blocks[0] {
        *output = Some("after".into());
    }
    assert_ne!(result.content_hash().unwrap(), result_before);
    // The producer supplies the result's full-conversation position while its
    // new source observation appends. Call and anchor are not reobserved, so
    // their original canonical rows and raw provenance remain unchanged.
    let result_id = result.id.clone();
    observe_snapshot_at(&mut archive, "parent", 2, result);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(prefix_rows(&reader, &call, &anchor), rows_before);
    assert!(
        reader.message_order(&result_id).unwrap().unwrap()
            > reader.message_order(&anchor.id).unwrap().unwrap()
    );
    let selected = reader.select_messages(&sid("child"), true).unwrap();
    let mut ids = Vec::new();
    selected
        .visit_available_tool_message_ids(|id| {
            ids.push(id.to_owned());
            Ok(())
        })
        .unwrap();
    assert_eq!(ids, [call.id.to_string()]);
    assert!(!ids.contains(&result_id.to_string()));
    assert_eq!(selected.fingerprint().unwrap(), own_before);
    drop(selected);
    drop(reader);
    assert_ne!(
        fingerprint(&archive, "session:claude-code:child", true),
        before
    );
}
#[test]
fn incomplete_parent_issue_keeps_available_tail_pair_dependencies() {
    let mut archive = WriterArchive::new();
    archive.links.push(Link {
        kind: LinkKind::Copy,
        parent: sid("missing"),
        child: sid("child"),
        at: None,
        call_id: None,
        evidence: "recorded missing parent".into(),
    });
    let mut pair = call(
        "child",
        1,
        1,
        "synthetic",
        None,
        serde_json::json!({"synthetic":true}),
        Ok("before"),
    );
    let call = pair.remove(0);
    let mut result = pair.remove(0);
    archive.write("child", vec![call, result.clone()]);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert!(
        !reader
            .select_messages(&sid("child"), true)
            .unwrap()
            .parents()
            .complete
    );
    drop(reader);
    let before = fingerprint(&archive, "session:claude-code:child", true);
    if let Block::ToolResult { output, .. } = &mut result.blocks[0] {
        *output = Some("after".into());
    }
    observe_snapshot(&mut archive, "child", result);
    assert_ne!(
        fingerprint(&archive, "session:claude-code:child", true),
        before
    );
}
