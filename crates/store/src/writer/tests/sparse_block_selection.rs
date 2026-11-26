use super::*;
use polylogue_model::bundle::ShownBlocks;
use polylogue_model::read::EntryKind::{Call, Header, Other, Result as Output, Text};
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
mod order;

fn original() -> Message {
    let mut message = text_message("sparse", "s", Role::Assistant, "source", Some("end_turn"));
    message.blocks = [
        serde_json::json!({"type":"text","text":"first source text"}),
        serde_json::json!({"type":"tool_call","call_id":"call","tool":"shell","input":{"opaque":[null,false,42]},"command":"true"}),
        serde_json::json!({"type":"tool_result","call_id":"call","outcome":{"kind":"ok"},"output":"source output","data":{"opaque":[{},null]}}),
        serde_json::json!({"type":"other","type_name":"synthetic","data":{"exact":null}}),
    ].into_iter().map(|value| serde_json::from_value(value).unwrap()).collect();
    message
}

fn selected(message: &Message, ordinals: &[u64]) -> (Message, ShownBlocks) {
    let mut part = message.clone();
    part.blocks = ordinals
        .iter()
        .map(|&ordinal| message.blocks[ordinal as usize].clone())
        .collect();
    (
        part,
        ShownBlocks {
            source_version: message.content_hash().unwrap(),
            block_count: message.blocks.len() as u64,
            block_ordinals: ordinals.to_vec(),
        },
    )
}

fn projected_batch(input: &str, message: Message, block_selection: ShownBlocks) -> Batch {
    let mut batch = batch(input, Route::Live, 0, vec![row(0, message)], &["s"]);
    batch.parse[0]
        .output
        .block_selections
        .insert(0, block_selection);
    batch.parse[0].prepare();
    batch
}

fn write(w: &TestArchive, mut batch: Batch) {
    if matches!(batch.change, Change::First) {
        let bytes = vec![b' '; batch.length as usize];
        let prepared = w.archive.prepare_bytes(bytes.as_slice()).unwrap();
        if batch.input.route.is_imported() {
            batch.input.id = prepared.hash().to_string();
        }
        let mut captured = batch.clone();
        captured.parse.clear();
        let stored = w.writer.store_bytes(prepared, captured).unwrap();
        batch.input.id = stored.input;
        batch.change = Change::Reparse {
            chain: stored.chain,
        };
    }
    batch.observation = None;
    w.writer.send(batch).unwrap();
    let report = w.writer.flush_rows().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
}

fn current(w: &TestArchive, id: &MessageId) -> (Message, ShownBlocks) {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .message_with_block_selection(id)
        .unwrap()
        .unwrap()
}

#[test]
fn sparse_original_positions_survive_reopen_and_streaming_reads() {
    let w = test_archive();
    let message = original();
    let part = selected(&message, &[1, 3]);
    let expected = part.clone();
    write(&w, projected_batch("partial", part.0, part.1));
    assert_eq!(current(&w, &message.id), expected);
    super::items::assert_entry_kinds(
        &w,
        "s",
        &[(None, Header), (Some(1), Call), (Some(3), Other)],
    );
    let TestArchive {
        writer,
        _lock,
        _dir,
        ..
    } = w;
    drop(writer);
    drop(_lock);
    let (archive, _lock) = Archive::open_or_create(&_dir.path().join("archive")).unwrap();
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.message_with_block_selection(&message.id).unwrap(),
        Some(expected.clone())
    );
    assert!(
        reader
            .message_commit(&message.id)
            .unwrap()
            .is_some_and(|commit| commit > 0)
    );
    let mut streamed = Vec::new();
    reader
        .select_messages(&sid("s"), false)
        .unwrap()
        .visit_with_sources(
            Default::default(),
            |message,
             crate::read::MessageParse {
                 blocks: block_selection,
                 committed: _,
                 ordinal: _,
             }| {
                streamed.push((message, block_selection));
                std::ops::ControlFlow::Continue(())
            },
        )
        .unwrap();
    assert_eq!(streamed, vec![expected]);
    assert_eq!(
        reader
            .query("SELECT idx,ordinal FROM block ORDER BY idx", &[])
            .unwrap(),
        [
            serde_json::json!({"idx":0,"ordinal":"1"}),
            serde_json::json!({"idx":1,"ordinal":"3"}),
        ]
    );
}

#[test]
fn compatible_partials_union_without_downgrading_full_or_pinned_readings() {
    let w = test_archive();
    let message = original();
    let first = selected(&message, &[2]);
    write(
        &w,
        projected_batch("first", first.0.clone(), first.1.clone()),
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let pinned = pool.reader().unwrap();
    assert_eq!(
        pinned.message_with_block_selection(&message.id).unwrap(),
        Some(first.clone())
    );
    super::items::assert_entry_kinds(&w, "s", &[(None, Header), (Some(2), Output)]);
    let all = [
        (None, Header),
        (Some(0), Text),
        (Some(1), Call),
        (Some(2), Output),
        (Some(3), Other),
    ];
    let second = selected(&message, &[0, 1, 3]);
    write(
        &w,
        projected_batch("second", second.0.clone(), second.1.clone()),
    );
    assert_eq!(
        pinned.message_with_block_selection(&message.id).unwrap(),
        Some(first)
    );
    drop(pinned);
    let full = selected(&message, &[0, 1, 2, 3]);
    assert_eq!(current(&w, &message.id), full);
    super::items::assert_entry_kinds(&w, "s", &all);
    let mut replay = projected_batch("second", second.0, second.1);
    replay.change = Change::Reparse { chain: 1 };
    write(&w, replay);
    assert_eq!(current(&w, &message.id), full);
    super::items::assert_entry_kinds(&w, "s", &all);
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(DISTINCT hash) AS n FROM message_parse", &[])
            .unwrap()[0]["n"],
        1
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM parse_text", &[])
            .unwrap()[0]["n"],
        2
    );
}

#[test]
fn sparse_observation_cannot_replace_a_complete_value() {
    let w = test_archive();
    let message = original();
    write(
        &w,
        batch(
            "native",
            Route::Live,
            0,
            vec![row(0, message.clone())],
            &["s"],
        ),
    );
    let part = selected(&message, &[1]);
    write(&w, projected_batch("import", part.0, part.1));
    assert_eq!(current(&w, &message.id), selected(&message, &[0, 1, 2, 3]));
}

#[test]
fn sparse_parts_at_later_records_of_one_input_still_union() {
    let w = test_archive();
    let message = original();
    let first = selected(&message, &[0]);
    let second = selected(&message, &[1, 2, 3]);
    let mut input = batch(
        "same-input",
        Route::Live,
        0,
        vec![row(0, first.0), row(50, second.0)],
        &["s"],
    );
    input.parse[0].output.block_selections.insert(0, first.1);
    input.parse[0].output.block_selections.insert(1, second.1);
    input.parse[0].prepare();
    write(&w, input);
    assert_eq!(current(&w, &message.id), selected(&message, &[0, 1, 2, 3]));
}

#[test]
fn sparse_union_keeps_the_later_required_bytes() {
    let w = test_archive();
    let message = original();
    for (input, ordinals) in [("a-first", &[0][..]), ("z-second", &[1, 2, 3][..])] {
        let part = selected(&message, ordinals);
        write(&w, projected_batch(input, part.0, part.1));
    }
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let first = crate::source::Source::committed_through(reader.conn().unwrap(), "a-first", 1, 100)
        .unwrap()
        .unwrap();
    let second =
        crate::source::Source::committed_through(reader.conn().unwrap(), "z-second", 1, 100)
            .unwrap()
            .unwrap();
    assert!(second > first);
    let shown: String = reader
        .conn()
        .unwrap()
        .query_row(
            "SELECT i.id FROM message m JOIN input_ref i ON i.pk=m.input_pk",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(shown, "a-first");
    assert_eq!(reader.message_commit(&message.id).unwrap(), Some(second));
    assert_eq!(current(&w, &message.id), selected(&message, &[0, 1, 2, 3]));
}

#[test]
fn sparse_distinct_source_versions_keep_the_existing_selection_rule() {
    let w = test_archive();
    let message = original();
    let first = selected(&message, &[0]);
    write(
        &w,
        projected_batch("preferred", first.0.clone(), first.1.clone()),
    );
    let mut different = message.clone();
    different.blocks[3] = Block::Text {
        text: "another source reading".into(),
    };
    let second = selected(&different, &[3]);
    let mut batch = projected_batch("other", second.0, second.1);
    batch.input.route = polylogue_model::Route::Export;
    write(&w, batch);
    assert_eq!(current(&w, &message.id), first);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(DISTINCT hash) AS n FROM message_parse", &[])
            .unwrap()[0]["n"],
        2
    );
}

#[test]
fn sparse_unsigned_ordinals_are_not_renumbered_by_attachment_or_tool_reads() {
    let w = test_archive();
    let mut message = original();
    message.blocks = vec![
        serde_json::from_value(serde_json::json!({"type":"image","direction":"input","url":"https://example.invalid/image.png"})).unwrap(),
        message.blocks[1].clone(),
    ];
    let ordinals = vec![1_u64 << 63, u64::MAX - 1];
    let block_selection = ShownBlocks {
        source_version: BlobHash::of(b"synthetic unknown complete source"),
        block_count: u64::MAX,
        block_ordinals: ordinals.clone(),
    };
    write(
        &w,
        projected_batch("large-positions", message.clone(), block_selection.clone()),
    );
    assert_eq!(current(&w, &message.id), (message.clone(), block_selection));
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let asset = reader
        .attachment(&message.id, ordinals[0] as usize, 0)
        .unwrap();
    assert_eq!(asset.block as u64, ordinals[0]);
    assert!(matches!(
        reader.attachment(&message.id, 0, 0),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(
        reader.tool_outcomes(&message.id).unwrap()[0].0 as u64,
        ordinals[1]
    );
}

#[test]
fn complete_repeated_readings_do_not_reload_historical_message_bodies() {
    let w = test_archive();
    let message = original();
    write(
        &w,
        batch(
            "native",
            Route::Live,
            0,
            vec![row(0, message.clone())],
            &["s"],
        ),
    );
    let mut index = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    // Immediate: the writer may still be applying search entries, and a
    // deferred transaction that read before its commit could not write.
    let tx = index
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    tx.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "message_parse",
            column_name: "body",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    let mut replay = batch("native", Route::Live, 0, vec![row(0, message)], &["s"]);
    replay.change = Change::Reparse { chain: 1 };
    assert_eq!(
        write_test_rows(&tx, &replay, &replay.parse[0], 1, 2).unwrap(),
        0
    );
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
}
