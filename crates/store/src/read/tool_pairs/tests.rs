use super::*;
use crate::{
    ReadPool,
    cancellation::Cancellation,
    fixture::writer::{WriterArchive, call},
    read::limit::ReadLimit,
};
use polylogue_model::BlobHash;
use polylogue_model::Message;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

fn pair(session: &str, n: u32, output: &str) -> (Message, Message) {
    let mut messages = call(
        session,
        n,
        10,
        "synthetic",
        Some("run"),
        serde_json::json!({"synthetic":true}),
        Ok(output),
    );
    (messages.remove(0), messages.remove(0))
}
fn index(archive: &WriterArchive) -> rusqlite::Connection {
    crate::archive::open_write(&archive.archive.index_db(), crate::archive::cache::SETUP).unwrap()
}
#[test]
fn indexed_pairs_keep_original_ordinals_and_provider_payload_in_either_arrival_order() {
    for result_first in [false, true] {
        let mut archive = WriterArchive::new();
        let output = "💡".repeat(300_000);
        let (mut call, mut result) = pair("s", 1, &output);
        call.blocks.insert(
            0,
            Block::Text {
                text: "before".into(),
            },
        );
        result.blocks.splice(
            0..0,
            (0..7).map(|_| Block::Text {
                text: "before".into(),
            }),
        );
        let expected_data = serde_json::json!({"unknown_provider_field":{"flag":false,"n":9},"empty":[],"large":"x".repeat(128 * 1024)});
        if let Block::ToolResult {
            data,
            reported_duration_ms,
            ..
        } = &mut result.blocks[7]
        {
            *data = Some(expected_data);
            *reported_duration_ms = Some(u64::MAX);
        }
        let expected_call = call.blocks[1].clone();
        let expected_result = result.blocks[7].clone();
        let rows = if result_first {
            vec![result.clone(), call.clone()]
        } else {
            vec![call.clone(), result.clone()]
        };
        archive.write("s", rows);
        let pool = ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        let pair = reader.message_tool_pair(&call.id, 1).unwrap().unwrap();
        assert_eq!(pair.call.block, expected_call);
        assert_eq!(pair.result.as_ref().unwrap().block, expected_result);
        assert_eq!(pair.call.source.block, Some(1));
        assert_eq!(pair.result.as_ref().unwrap().source.block, Some(7));
        assert_eq!(pair.emit_at, format!("block:{}#1", call.id));
        assert_eq!(pair.duration_ms, ValueState::known(u64::MAX));
        assert_eq!(
            reader
                .message_tool_pair_header(&call.id, 1)
                .unwrap()
                .unwrap()
                .duration_ms,
            ValueState::known(u64::MAX)
        );
        assert_eq!(
            reader.message_tool_pair(&result.id, 7).unwrap().unwrap(),
            pair
        );
        assert_eq!(
            reader.message_tool_pair_sources(&result.id, 7).unwrap(),
            Some((
                Source::block(&call.id, &call.session, 1, false),
                Some(Source::block(&result.id, &result.session, 7, false))
            ))
        );
    }
}
#[test]
fn missing_and_orphan_results_never_pair_by_recency_or_cross_session_ids() {
    let mut archive = WriterArchive::new();
    let (call, _) = pair("s", 1, "missing");
    let (_, orphan) = pair("s", 2, "orphan");
    let (other_call, other_result) = pair("other", 1, "other session");
    archive.write("s", vec![orphan.clone(), call.clone()]);
    archive.write("other", vec![other_call, other_result]);
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let pair = reader.message_tool_pair(&call.id, 0).unwrap().unwrap();
    assert!(pair.result.is_none());
    assert_eq!(
        reader.message_tool_pair_result_shape(&call.id, 0).unwrap(),
        None
    );
    assert!(matches!(pair.duration_ms, ValueState::Unknown { .. }));
    assert!(!pair.running);
    assert!(reader.message_tool_pair(&orphan.id, 0).unwrap().is_none());
    assert!(
        reader
            .message_tool_pair_sources(&orphan.id, 0)
            .unwrap()
            .is_none()
    );
}
#[test]
fn result_shape_reports_whether_data_is_present_and_how_many_attachments() {
    for (attachments, expected) in [(0, (false, 0)), (1, (false, 1))] {
        let mut archive = WriterArchive::new();
        let (call, mut result) = pair("s", 1, "output");
        if let Block::ToolResult {
            attachments: stored,
            ..
        } = &mut result.blocks[0]
        {
            *stored = (0..attachments)
                .map(|_| polylogue_model::Attachment {
                    name: Some("file".into()),
                    ..Default::default()
                })
                .collect();
        }
        archive.write("s", vec![call.clone(), result.clone()]);
        let pool = ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader.message_tool_pair_result_shape(&call.id, 0).unwrap(),
            Some(expected)
        );
    }
}
#[test]
fn metadata_and_collapsed_view_skip_remote_output_and_data() {
    let mut archive = WriterArchive::new();
    let (call, mut result) = pair("s", 1, "remote output");
    if let Block::ToolResult {
        data, attachments, ..
    } = &mut result.blocks[0]
    {
        *data = Some(serde_json::json!({"provider":"structured payload"}));
        *attachments = vec![
            polylogue_model::Attachment {
                name: Some("one".into()),
                ..Default::default()
            },
            polylogue_model::Attachment {
                name: Some("two".into()),
                ..Default::default()
            },
        ];
    }
    archive.write("s", vec![call.clone(), result.clone()]);
    // Shape must not attempt to decode this deliberately poisoned data payload.
    let conn = index(&archive);
    crate::content::poison_data(
        &conn,
        "block",
        "message_pk=(SELECT pk FROM message WHERE id=?1) AND data_length IS NOT NULL",
        [result.id.to_string()],
    )
    .unwrap();
    drop(conn);
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| {
            if matches!(
                context.action,
                AuthAction::Read {
                    table_name: "text",
                    column_name: "body"
                }
            ) {
                Authorization::Deny
            } else {
                Authorization::Allow
            }
        }))
        .unwrap();
    assert!(
        reader
            .message_tool_pair_sources(&result.id, 0)
            .unwrap()
            .is_some()
    );
    assert_eq!(
        reader.message_tool_pair_result_shape(&call.id, 0).unwrap(),
        Some((true, 2))
    );
    assert_eq!(
        reader
            .message_tool_pair_result_shape(&result.id, 0)
            .unwrap(),
        Some((true, 2))
    );
    let header = reader
        .message_tool_pair_header(&call.id, 0)
        .unwrap()
        .unwrap();
    assert!(matches!(
        &header.call.block,
        Block::ToolCall { input, .. } if input.is_null()
    ));
    assert!(matches!(
        header.result.unwrap().block,
        Block::ToolResult {
            output: None,
            data: None,
            attachments,
            ..
        } if attachments.is_empty()
    ));
    assert!(reader.message_tool_pair(&call.id, 0).is_err());
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
                    table_name: "text",
                    column_name: "body"
                }
            ) {
                Authorization::Deny
            } else {
                Authorization::Allow
            }
        }))
        .unwrap();
    assert!(
        reader
            .message_tool_pair_sources(&call.id, 0)
            .unwrap()
            .is_some()
    );
}
#[test]
fn call_coordinate_never_hydrates_other_calls_or_their_results() {
    let mut archive = WriterArchive::new();
    let (mut first, result) = pair("s", 1, "selected");
    let (second, other_result) = pair("s", 2, "not selected");
    first.blocks.push(second.blocks[0].clone());
    archive.write("s", vec![first.clone(), result, other_result.clone()]);
    let conn = index(&archive);
    crate::content::poison_bodies(&conn, "block", "(message_pk=(SELECT pk FROM message WHERE id=?1) AND position=1) OR message_pk=(SELECT pk FROM message WHERE id=?2)", params![first.id.to_string(), other_result.id.to_string()]).unwrap();
    drop(conn);
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader.message_tool_pair(&first.id, 0).unwrap().unwrap();
    assert!(
        matches!(selected.result.unwrap().block, Block::ToolResult { output: Some(ref text), .. } if text == "selected")
    );
    assert!(reader.message_tool_pair(&first.id, 1).is_err());
}
#[test]
fn hidden_current_blocks_never_fall_back_to_old_bodies() {
    use polylogue_model::{MessageRow, Output, Session, SessionKind, bundle::ShownBlocks};
    let mut archive = WriterArchive::new();
    let (call, result) = pair("s", 1, "known fragment");
    let call_ordinal = u64::MAX - 1;
    let mut output = Output {
        sessions: vec![Session::new(call.session.clone(), SessionKind::Main)],
        messages: vec![
            MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 0,
                end: 1,
                message: call.clone(),
            },
            MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 1,
                end: 2,
                message: result.clone(),
            },
        ],
        ..Default::default()
    };
    output.block_selections.insert(
        0,
        ShownBlocks {
            source_version: BlobHash::of(b"partial original call"),
            block_count: u64::MAX,
            block_ordinals: vec![call_ordinal],
        },
    );
    output.block_selections.insert(
        1,
        ShownBlocks {
            source_version: BlobHash::of(b"partial original result"),
            block_count: 8,
            block_ordinals: vec![7],
        },
    );
    archive.write_output("sparse", "synthetic", "/synthetic/sparse", output, 2);
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let pair = reader
        .message_tool_pair(&call.id, call_ordinal)
        .unwrap()
        .unwrap();
    assert_eq!(pair.call.source.block, Some(call_ordinal));
    assert_eq!(pair.result.as_ref().unwrap().source.block, Some(7));
    drop(reader);
    let conn = index(&archive);
    conn.execute(
        "UPDATE message SET layout=?1 WHERE id=?2",
        params![
            serde_json::to_vec(&(8u64, vec![6u64])).unwrap(),
            result.id.to_string()
        ],
    )
    .unwrap();
    crate::content::poison_bodies(
        &conn,
        "block",
        "message_pk=(SELECT pk FROM message WHERE id=?1)",
        [result.id.to_string()],
    )
    .unwrap();
    drop(conn);
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .message_tool_pair(&call.id, call_ordinal)
            .unwrap()
            .unwrap()
            .result
            .is_none()
    );
    assert!(reader.message_tool_pair(&result.id, 7).unwrap().is_none());
}
#[test]
fn a_corrupt_cross_session_or_mismatched_id_relation_cannot_expose_a_result_body() {
    for cross_session in [false, true] {
        let mut archive = WriterArchive::new();
        let (call, _) = pair("s", 1, "not present");
        let (_, other_result) = pair(
            if cross_session { "other" } else { "s" },
            if cross_session { 1 } else { 2 },
            "must stay unread",
        );
        archive.write("s", vec![call.clone()]);
        archive.write(
            if cross_session { "other" } else { "s" },
            vec![other_result.clone()],
        );
        let conn = index(&archive);
        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        conn.execute("UPDATE tool_call SET result_message_id=?1,result_position=(SELECT b.position FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.id=?1 AND b.kind='tool_result') WHERE call_block_pk IN (SELECT b.pk FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.id=?2)", params![other_result.id.to_string(), call.id.to_string()]).unwrap();
        crate::content::poison_bodies(
            &conn,
            "block",
            "message_pk=(SELECT pk FROM message WHERE id=?1)",
            [other_result.id.to_string()],
        )
        .unwrap();
        drop(conn);
        let pool = ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert!(
            reader
                .message_tool_pair(&call.id, 0)
                .unwrap()
                .unwrap()
                .result
                .is_none()
        );
        assert!(
            reader
                .message_tool_pair_sources(&other_result.id, 0)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn cancellation_and_scope_reset_are_reported_by_the_pair_readers() {
    let mut archive = WriterArchive::new();
    let (call, result) = pair("s", 1, "payload");
    archive.write("s", vec![call.clone(), result]);
    let pool = ReadPool::open(&archive.archive).unwrap();
    let cancellation = Cancellation::default();
    let limit = ReadLimit::without_deadline(cancellation.clone());
    limit.scope(|| {
        let reader = pool.reader().unwrap();
        reader.message_tool_pair_sources(&call.id, 0).unwrap();
        cancellation.cancel();
        assert!(matches!(
            reader.message_tool_pair(&call.id, 0),
            Err(StoreError::ReadCancelled)
        ));
    });
    let reader = pool.reader().unwrap();
    archive.archive.read_scope.begin_reset().unwrap();
    assert!(matches!(
        reader.message_tool_pair(&call.id, 0),
        Err(StoreError::ScopeChanged)
    ));
    assert!(matches!(
        reader.message_tool_pair_sources(&call.id, 0),
        Err(StoreError::ScopeChanged)
    ));
    archive.archive.read_scope.finish_reset();
}

#[test]
fn a_failed_call_shows_its_first_output_line() {
    let failure = |session: &str, output: &str| {
        let mut messages = call(
            session,
            1,
            10,
            "synthetic",
            Some("cargo test"),
            serde_json::json!({}),
            Err(output),
        );
        (messages.remove(0), messages.remove(0))
    };
    let (call, result) = failure(
        "s",
        "running 2 tests\n\ntest parse_dates::tomorrow_at_noon ... FAILED\ntest other ... ok\n",
    );
    let mut archive = WriterArchive::new();
    archive.write("s", vec![call.clone(), result]);
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    // The collapsed read pairs without output and still names the failure.
    for pair in [
        reader
            .message_tool_pair_header(&call.id, 0)
            .unwrap()
            .unwrap(),
        reader.message_tool_pair(&call.id, 0).unwrap().unwrap(),
    ] {
        assert_eq!(
            pair.error_line.as_deref(),
            Some("test parse_dates::tomorrow_at_noon ... FAILED")
        );
    }
    // An output that never names a failure shows its first non-blank line.
    let (call, result) = failure("q", "\n\nsomething went sideways\nsecond");
    let mut quiet = WriterArchive::new();
    quiet.write("q", vec![call.clone(), result]);
    let pool = ReadPool::open(&quiet.archive).unwrap();
    let pair = pool
        .reader()
        .unwrap()
        .message_tool_pair_header(&call.id, 0)
        .unwrap()
        .unwrap();
    assert_eq!(pair.error_line.as_deref(), Some("something went sideways"));
}
