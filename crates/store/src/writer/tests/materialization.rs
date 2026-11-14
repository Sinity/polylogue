use super::*;
use polylogue_model::parser_context::{Delta, Entry, Key, Reference};
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
struct Work {
    message_reads: u64,
    text_queries: u64,
    text_rows: u64,
    text_key_queries: u64,
    version_key_queries: u64,
    vm_steps: u64,
}

thread_local! {
    static WORK: Cell<Work> = Cell::new(Work::default());
}

fn family(sql: &str) -> u8 {
    if sql.starts_with("SELECT b.body,p.layout,v.pk") {
        1
    } else if sql.starts_with("SELECT v.idx,t.body FROM message_version_text") {
        2
    } else if sql.starts_with("SELECT idx,text_id FROM message_version_text") {
        3
    } else if sql.starts_with("SELECT pk FROM message_version WHERE version_hash") {
        4
    } else {
        0
    }
}

fn trace(event: TraceEvent<'_>) {
    WORK.with(|work| {
        let mut value = work.get();
        match event {
            TraceEvent::Profile(statement, _) => {
                let kind = family(&statement.sql());
                match kind {
                    1 => value.message_reads += 1,
                    2 => value.text_queries += 1,
                    3 => value.text_key_queries += 1,
                    4 => value.version_key_queries += 1,
                    _ => {}
                }
                if kind != 0 {
                    // This fixture disables the statement cache, so each
                    // statement's VM counter belongs to this execution.
                    value.vm_steps +=
                        u64::try_from(statement.get_status(rusqlite::StatementStatus::VmStep))
                            .unwrap();
                }
            }
            TraceEvent::Row(statement) if family(&statement.sql()) == 2 => value.text_rows += 1,
            _ => {}
        }
        work.set(value);
    });
}

fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.set_prepared_statement_cache_capacity(0);
    crate::versions::track_unreferenced(&conn).unwrap();
    conn
}

fn reference(message: &Message, key: &str) -> Entry {
    Entry {
        key: Key::Response {
            namespace: String::new(),
            session: message.session.to_string(),
            key: key.into(),
        },
        value: Reference::Response {
            message: message.id.clone(),
            version: message.content_hash().unwrap(),
            offset: 0,
            end: 100,
        },
    }
}

fn reading(input: &str, messages: Vec<Message>, entries: Vec<Entry>) -> Batch {
    let mut b = batch(
        input,
        Route::Live,
        0,
        messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| row(i as u64, m))
            .collect(),
        &["s"],
    );
    b.parse[0].checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        "synthetic=1".into(),
        100,
        Some(serde_json::json!({"_polylogue_context": Delta {
            version: 1, base: 0, through: 100, entries,
        }})),
    ));
    b
}

fn apply(conn: &mut Connection, mut b: Batch) -> (Result<u64>, Work) {
    resolve::prepare_index(conn, &mut b, 1, false).unwrap();
    let tx = conn.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let texts = crate::text_rows::TextIds::new(&tx, false).unwrap();
    WORK.with(|w| w.set(Work::default()));
    tx.trace_v2(
        TraceEventCodes::SQLITE_TRACE_PROFILE | TraceEventCodes::SQLITE_TRACE_ROW,
        Some(trace),
    );
    let p = &b.parse[0];
    let result = write_rows(
        &texts,
        &tx,
        &b,
        p,
        p.resolution.as_ref().unwrap(),
        1,
        None,
        Some(&fixture_receipt()),
        &changes,
    );
    tx.trace_v2(TraceEventCodes::empty(), None);
    let work = WORK.with(Cell::get);
    if result.is_ok() {
        crate::items::refresh(&tx, &changes).unwrap();
        tx.commit().unwrap();
    } else {
        tx.rollback().unwrap();
    }
    (result, work)
}

#[test]
fn registered_responses_and_selected_blocks_consume_the_written_keys() {
    let mut conn = database();
    let body = "synthetic response text ".repeat(4096);
    let messages: Vec<_> = (0..16)
        .map(|i| {
            let mut message = text_message(
                &format!("response-{i}"),
                "s",
                Role::Assistant,
                &body,
                Some("end_turn"),
            );
            message.blocks.push(Block::Text { text: body.clone() });
            message
        })
        .collect();
    let entries: Vec<_> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| reference(m, &format!("response-{i}")))
        .collect();
    let (written, fresh) = apply(
        &mut conn,
        reading("original", messages.clone(), entries.clone()),
    );
    assert_eq!(written.unwrap(), 16);
    assert_eq!(
        (fresh.message_reads, fresh.text_queries, fresh.text_rows),
        (0, 0, 0)
    );
    assert_eq!(fresh.text_key_queries, 0);
    // Existing versions were not written by this registration. This real
    // duplicate-source path is the independent full-validation control.
    let (written, duplicate) = apply(&mut conn, reading("copy", messages.clone(), entries));
    assert_eq!(written.unwrap(), 0);
    assert_eq!(
        (
            duplicate.message_reads,
            duplicate.text_queries,
            duplicate.text_rows
        ),
        (16, 16, 32)
    );
    assert!(duplicate.version_key_queries >= 16);
    assert!(duplicate.vm_steps > fresh.vm_steps);
    for message in &messages {
        let (stored, _) = crate::versions::read_message(&conn, &message.content_hash().unwrap().0)
            .unwrap()
            .unwrap();
        assert_eq!(&stored, message);
    }
    assert_eq!(
        conn.query_row("SELECT count(*) FROM parser_context", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        32
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
    println!(
        "registered materialization: fresh={fresh:?} duplicate={duplicate:?} fallback_decoded_text_bytes={}",
        32 * body.len()
    );
}

#[test]
fn mismatched_context_references_use_exact_validation_and_rollback() {
    for wrong in ["message", "session", "version", "range"] {
        let mut conn = database();
        let message = text_message(
            "response",
            "s",
            Role::Assistant,
            "synthetic body",
            Some("end_turn"),
        );
        let mut entry = reference(&message, "response");
        match wrong {
            "message" => {
                if let Reference::Response { message, .. } = &mut entry.value {
                    *message = MessageId::parse("claude-code:other").unwrap();
                }
            }
            "session" => {
                if let Key::Response { session, .. } = &mut entry.key {
                    *session = "claude-code:other".into();
                }
            }
            "version" => {
                if let Reference::Response { version, .. } = &mut entry.value {
                    *version = polylogue_model::BlobHash::of(b"unregistered version");
                }
            }
            "range" => {
                if let Reference::Response { end, .. } = &mut entry.value {
                    *end = 101;
                }
            }
            _ => unreachable!(),
        }
        let (result, work) = apply(
            &mut conn,
            reading("bad", vec![message.clone()], vec![entry]),
        );
        assert!(result.is_err(), "{wrong}");
        if wrong != "range" {
            assert_eq!(work.message_reads, 1, "{wrong}");
        }
        assert_eq!(
            conn.query_row("SELECT count(*) FROM message_version", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        let (result, _) = apply(
            &mut conn,
            reading(
                "bad",
                vec![message.clone()],
                vec![reference(&message, "response")],
            ),
        );
        assert_eq!(result.unwrap(), 1, "retry after {wrong}");
    }
}

#[test]
fn old_context_versions_keep_payload_and_hash_corruption_checks() {
    for damage in ["missing", "text", "layout"] {
        let mut conn = database();
        let message = text_message(
            "response",
            "s",
            Role::Assistant,
            "original response",
            Some("end_turn"),
        );
        apply(
            &mut conn,
            reading(
                "original",
                vec![message.clone()],
                vec![reference(&message, "response")],
            ),
        )
        .0
        .unwrap();
        match damage {
            "missing" => conn
                .execute("DELETE FROM message_version_body", [])
                .unwrap(),
            "text" => conn
                .execute(
                    "UPDATE text SET body=?1",
                    [crate::encoding::pack_text("changed response").unwrap()],
                )
                .unwrap(),
            "layout" => conn
                .execute("UPDATE message_projection SET layout=x'00'", [])
                .unwrap(),
            _ => unreachable!(),
        };
        let (result, work) = apply(
            &mut conn,
            reading(
                "old-reference",
                vec![],
                vec![reference(&message, "response")],
            ),
        );
        assert!(result.is_err(), "{damage}");
        assert_eq!(work.message_reads, 1);
        assert_eq!(
            conn.query_row(
                "SELECT count(*) FROM input_ref WHERE id='old-reference'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
}

#[test]
fn context_keeps_old_and_fresh_versions_of_the_same_message_distinct() {
    let mut conn = database();
    let old = text_message(
        "response",
        "s",
        Role::Assistant,
        "old response",
        Some("end_turn"),
    );
    let new = text_message(
        "response",
        "s",
        Role::Assistant,
        "new longer response",
        Some("end_turn"),
    );
    apply(
        &mut conn,
        reading("old", vec![old.clone()], vec![reference(&old, "old")]),
    )
    .0
    .unwrap();
    let (result, work) = apply(
        &mut conn,
        reading(
            "new",
            vec![new.clone()],
            vec![reference(&old, "old"), reference(&new, "new")],
        ),
    );
    result.unwrap();
    assert_eq!((work.message_reads, work.text_queries), (1, 1));
    let input: i64 = conn
        .query_row("SELECT pk FROM input_ref WHERE id='new'", [], |r| r.get(0))
        .unwrap();
    let rows: Vec<Vec<u8>> = conn.prepare("SELECT v.version_hash FROM parser_context c JOIN message_version v ON v.pk=c.version_pk WHERE c.input_pk=?1 ORDER BY c.key")
        .unwrap().query_map([input], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect();
    let mut expected = vec![
        old.content_hash().unwrap().0.to_vec(),
        new.content_hash().unwrap().0.to_vec(),
    ];
    expected.sort();
    let mut actual = rows;
    actual.sort();
    assert_eq!(actual, expected);
}

#[test]
fn sparse_registered_context_keeps_full_hash_and_enriches_block_positions() {
    let mut conn = database();
    let mut full = text_message(
        "response",
        "s",
        Role::Assistant,
        "first block",
        Some("end_turn"),
    );
    full.blocks.push(Block::Text {
        text: "second block".into(),
    });
    let hash = full.content_hash().unwrap();
    for ordinal in [0, 1] {
        let mut part = full.clone();
        part.blocks = vec![full.blocks[ordinal].clone()];
        let mut b = reading(
            &format!("part-{ordinal}"),
            vec![part],
            vec![reference(&full, "response")],
        );
        b.parse[0].output.message_projections.insert(
            0,
            polylogue_model::bundle::MessageProjection {
                source_version: hash,
                block_count: 2,
                block_ordinals: vec![ordinal as u64],
            },
        );
        let (result, work) = apply(&mut conn, b);
        result.unwrap();
        assert_eq!((work.message_reads, work.text_key_queries), (0, 0));
    }
    let (stored, projection) = crate::versions::read_message(&conn, &hash.0)
        .unwrap()
        .unwrap();
    assert_eq!(stored, full);
    assert_eq!(projection.block_ordinals, [0, 1]);
    assert_eq!(projection.source_version, hash);
}
