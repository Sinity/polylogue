//! A value a parser leaves in the stored bytes is not copied into the index
//! and is read from the record when it is shown (docs/archive.md, content).
use super::*;

const SIGNATURE: &str = "gAAAAABsyntheticEncryptedReasoningThatNoSearchUses";

/// One input whose two records hold an encrypted reasoning item and a
/// compaction restating a message, as a Codex rollout writes them.
fn records() -> (Vec<u8>, [(u64, u64); 2]) {
    let reasoning = serde_json::json!({"timestamp": "2026-03-01T10:00:00.000Z", "type": "response_item",
        "payload": {"type": "reasoning", "summary": [], "encrypted_content": SIGNATURE}});
    let compaction = serde_json::json!({"timestamp": "2026-03-01T10:00:01.000Z", "type": "compacted",
        "payload": {"message": "", "replacement_history": [{"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "the restated prompt"}]}]}});
    let mut bytes = Vec::new();
    let mut ranges = [(0, 0); 2];
    for (n, record) in [reasoning, compaction].iter().enumerate() {
        let start = bytes.len() as u64;
        bytes.extend(serde_json::to_vec(record).unwrap());
        bytes.push(b'\n');
        ranges[n] = (start, bytes.len() as u64);
    }
    (bytes, ranges)
}

fn stored_batch(w: &TestArchive) -> crate::staging::PreparedBatch {
    let (bytes, [reasoning, compaction]) = records();
    let mut message = text_message("rs", "s", Role::Assistant, "", None);
    message.blocks = vec![Block::Thinking {
        text: None,
        signature: Some(SIGNATURE.into()),
        redacted: true,
    }];
    let mut row = row(reasoning.0, message);
    row.end = reasoning.1;
    let mut batch = batch("input-codex", Route::Live, 0, vec![row], &["s"]);
    let step = &mut batch.parse[0];
    step.output.stored_fields.insert(
        0,
        vec![polylogue_model::StoredField {
            block: Some(0),
            field: "/signature".into(),
            start: reasoning.0,
            end: reasoning.1,
            pointer: "/payload/encrypted_content".into(),
        }],
    );
    step.output.events.push(Event {
        session: Some(sid("s")),
        kind: "compaction".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some(compaction),
        data: serde_json::json!({"replacement_history": [{"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "the restated prompt"}]}]}),
    });
    step.output.stored_event_fields.insert(
        0,
        vec![polylogue_model::StoredField {
            block: None,
            field: "/replacement_history".into(),
            start: compaction.0,
            end: compaction.1,
            pointer: "/payload/replacement_history".into(),
        }],
    );
    step.counts = counts(2);
    step.parsed_upto = bytes.len() as u64;
    batch.length = bytes.len() as u64;
    let (blob, file) = w.archive.prepare_blob(&bytes).unwrap();
    batch.new_blob = Some(blob);
    crate::staging::PreparedBatch {
        batch,
        files: file.into_iter().collect(),
    }
}

#[test]
fn encrypted_reasoning_and_restated_history_are_read_from_the_record_when_shown() {
    let w = test_archive();
    for result in w.writer.store_group(vec![stored_batch(&w)]).unwrap() {
        result.unwrap();
    }
    w.writer.flush().unwrap();
    // The index holds neither value.
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let mut held = Vec::new();
    for body in conn
        .prepare("SELECT body FROM content_group")
        .unwrap()
        .query_map([], |r| r.get::<_, Vec<u8>>(0))
        .unwrap()
    {
        held.extend(crate::content::decode(&body.unwrap()).unwrap());
    }
    let held = String::from_utf8_lossy(&held);
    assert!(!held.contains(SIGNATURE), "{held}");
    assert!(!held.contains("the restated prompt"), "{held}");
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    // A read of whole values shows them from the record.
    let messages = reader.messages(&sid("s"), false).unwrap();
    assert_eq!(
        messages[0].blocks,
        [Block::Thinking {
            text: None,
            signature: Some(SIGNATURE.into()),
            redacted: true,
        }]
    );
    let record = reader.event_record(&event_id(&conn)).unwrap().unwrap();
    assert_eq!(
        record.event.data["replacement_history"][0]["content"][0]["text"],
        "the restated prompt"
    );
    // A text read reads no record: the reasoning still shows as encrypted.
    let mut text = Vec::new();
    reader
        .select_messages(&sid("s"), false)
        .unwrap()
        .for_text()
        .visit(
            crate::read::MessagePosition {
                message: 0,
                key: None,
            },
            |message| {
                text.push(message);
                std::ops::ControlFlow::Continue(())
            },
        )
        .unwrap();
    let Block::Thinking {
        signature: Some(signature),
        ..
    } = &text[0].blocks[0]
    else {
        panic!("{:?}", text[0].blocks);
    };
    assert_ne!(signature, SIGNATURE);
}

fn event_id(conn: &rusqlite::Connection) -> String {
    conn.query_row("SELECT id FROM event", [], |r| r.get(0))
        .unwrap()
}

/// The rows a primary producer takes from a prepared step are whole: what
/// the index leaves in the stored bytes is still in them.
#[test]
fn a_prepared_step_gives_complete_rows_with_the_values_the_index_leaves_out() {
    let w = test_archive();
    let mut batch = stored_batch(&w).batch;
    let step = &mut batch.parse[0];
    let history = step.output.events[0].data.clone();
    step.prepare();
    let output = step.complete_output().unwrap();
    assert_eq!(
        output.messages[0].message.blocks,
        [Block::Thinking {
            text: None,
            signature: Some(SIGNATURE.into()),
            redacted: true,
        }]
    );
    assert_eq!(output.events[0].data, history);
    // The prepared content holds the places instead.
    let stored = step.prepared.message_value(0).unwrap();
    let Block::Thinking {
        signature: Some(signature),
        ..
    } = &stored.blocks[0]
    else {
        panic!("{:?}", stored.blocks);
    };
    assert!(crate::content::stored_at(signature).is_some());
    assert!(
        crate::content::stored_at(
            step.prepared.event_data(0).unwrap()["replacement_history"]
                .as_str()
                .unwrap()
        )
        .is_some()
    );
}

/// A message whose value's record cannot be read still reads, with the
/// value's place where the value would be (docs/archive.md, content).
#[test]
fn a_message_reads_while_the_record_holding_its_value_is_not_at_hand() {
    let w = test_archive();
    for result in w.writer.store_group(vec![stored_batch(&w)]).unwrap() {
        result.unwrap();
    }
    w.writer.flush().unwrap();
    // The reading names a record beyond its chain's stored bytes.
    let far = polylogue_model::StoredField {
        block: Some(0),
        field: "/signature".into(),
        start: 9_999_999,
        end: 99_999_999,
        pointer: "/payload/encrypted_content".into(),
    };
    let gone = crate::content::stored_marker(&far);
    {
        let conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
        let mut message = text_message("rs", "s", Role::Assistant, "", None);
        message.blocks = vec![Block::Thinking {
            text: None,
            signature: Some(SIGNATURE.into()),
            redacted: true,
        }];
        let content = crate::encoding::message_content(&mut message, &[far]).unwrap();
        let piece = crate::content::store_piece(&conn, &content, None).unwrap();
        conn.execute(
            "UPDATE message_parse SET content_group=?1,content_start=?2,content_length=?3",
            rusqlite::params![piece.group, piece.start, piece.length],
        )
        .unwrap();
    }
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let (message, _) = reader
        .message_with_block_selection(&MessageId::parse("claude-code:rs").unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(
        message.blocks,
        [Block::Thinking {
            text: None,
            signature: Some(gone),
            redacted: true,
        }]
    );
}
