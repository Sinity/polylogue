use super::*;
use polylogue_model::{Block, IdKind, Json, MessageRow, Role};

fn response(revision: u64) -> MessageRow {
    let mut message = crate::fixture::writer::message(
        "response",
        "scratch",
        Role::Assistant,
        100,
        vec![Block::Text {
            text: format!(
                "revision {revision}: {}",
                "Łódź \"response\"\n".repeat(4096)
            ),
        }],
    );
    message.attrs = serde_json::from_str(
        r#"{"z":{"second":1.50,"first":[true,null,"\\literal"]},"a":"終わり"}"#,
    )
    .unwrap();
    MessageRow {
        offset: 3,
        end: revision + 4,
        id_kind: if revision.is_multiple_of(2) {
            IdKind::Native
        } else {
            IdKind::Content
        },
        page_text: !revision.is_multiple_of(2),
        message,
    }
}

fn key(row: &MessageRow) -> Key {
    Key::Response {
        namespace: "nested/agent".into(),
        session: row.message.session.to_string(),
        key: row.message.id.to_string(),
    }
}

fn assert_reference(delta: &Delta, row: &MessageRow) {
    let entry = delta.entries.iter().find(|e| e.key == key(row)).unwrap();
    let expected = Reference::Response {
        message: row.message.id.clone(),
        version: row.message.content_hash().unwrap(),
        offset: row.offset,
        end: row.end,
    };
    assert_eq!(
        serde_json::to_value(&entry.value).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
}

#[test]
fn checkpoints_hash_only_final_responses_without_decoding_lookup_payloads() {
    let mut context = Overlay::temporary().unwrap();
    super::super::scratch::take_work();
    for revision in 0..16 {
        let row = response(revision);
        context
            .put(key(&row), Value::Response { row: Box::new(row) })
            .unwrap();
    }
    let alias = Key::Alias {
        namespace: String::new(),
        uuid: "alias".into(),
    };
    context
        .put(alias.clone(), Value::Alias { target: None })
        .unwrap();
    let state = Key::State {
        namespace: "calls".into(),
        key: "finished".into(),
    };
    context
        .put(state.clone(), Value::State { value: Json::Null })
        .unwrap();
    let puts = super::super::scratch::take_work();
    assert_eq!((puts.response_encodes, puts.response_hashes), (16, 0));

    let delta = context.checkpoint_delta(100).unwrap();
    let checkpoint = super::super::scratch::take_work();
    assert_eq!(
        (delta.base, delta.through, delta.entries.len()),
        (0, 100, 3)
    );
    assert_reference(&delta, &response(15));
    assert_eq!(
        (
            checkpoint.response_encodes,
            checkpoint.response_decodes,
            checkpoint.response_hashes
        ),
        (0, 0, 1)
    );
    assert_eq!(checkpoint.response_views, 1);
    assert_eq!(
        checkpoint.hashed_response_bytes,
        canonical_json::canonical_json(&response(15).message)
            .unwrap()
            .len() as u64
    );
    assert!(puts.encoded_response_bytes > 15 * checkpoint.hashed_response_bytes);
    assert!(
        delta
            .entries
            .iter()
            .any(|e| e.key == alias && matches!(e.value, Reference::Alias { target: None }))
    );
    assert!(delta.entries.iter().any(
        |e| e.key == state && matches!(&e.value, Reference::State { value } if value.is_null())
    ));
    assert!(context.checkpoint_delta(100).unwrap().entries.is_empty());
    assert_eq!(super::super::scratch::take_work().response_hashes, 0);

    // Actual reopen still decodes the one complete lookup payload. It does
    // not lose fields to the smaller checkpoint view or mark a clean key dirty.
    let Some(Value::Response { row }) = context.get(&key(&response(15))).unwrap() else {
        panic!("response disappeared after checkpoint")
    };
    assert_eq!(*row, response(15));
    let lookup = super::super::scratch::take_work();
    assert_eq!((lookup.response_decodes, lookup.response_hashes), (1, 0));
    context
        .put(
            key(&row),
            Value::Response {
                row: Box::new(response(16)),
            },
        )
        .unwrap();
    assert!(context.checkpoint_delta(99).is_err());
    let delta = context.checkpoint_delta(110).unwrap();
    assert_eq!(
        (delta.base, delta.through, delta.entries.len()),
        (100, 110, 1)
    );
    assert_reference(&delta, &response(16));
    assert_eq!(super::super::scratch::take_work().response_hashes, 1);
    println!("parser scratch work: puts={puts:?} checkpoint={checkpoint:?} lookup={lookup:?}");
}

#[test]
fn a_fresh_checkpoint_hashes_each_distinct_dirty_response_once() {
    let mut context = Overlay::temporary().unwrap();
    super::super::scratch::take_work();
    let rows: Vec<_> = (0..8)
        .map(|revision| {
            let mut row = response(revision);
            row.message.id =
                polylogue_model::MessageId::parse(&format!("claude-code:response-{revision}"))
                    .unwrap();
            context
                .put(
                    key(&row),
                    Value::Response {
                        row: Box::new(row.clone()),
                    },
                )
                .unwrap();
            row
        })
        .collect();
    let puts = super::super::scratch::take_work();
    assert_eq!((puts.response_encodes, puts.response_hashes), (8, 0));
    let delta = context.checkpoint_delta(100).unwrap();
    let checkpoint = super::super::scratch::take_work();
    assert_eq!(delta.entries.len(), 8);
    assert_eq!(
        (checkpoint.response_decodes, checkpoint.response_hashes),
        (0, 8)
    );
    assert_eq!(checkpoint.response_views, 8);
    for row in rows {
        assert_reference(&delta, &row);
    }
}

#[test]
fn failed_checkpoint_keeps_the_cut_and_all_dirty_values_for_retry() {
    let mut context = Overlay::temporary().unwrap();
    let row = response(0);
    let key = key(&row);
    context
        .put(
            key.clone(),
            Value::Response {
                row: Box::new(row.clone()),
            },
        )
        .unwrap();
    context
        .put(
            Key::Alias {
                namespace: String::new(),
                uuid: "first".into(),
            },
            Value::Alias { target: None },
        )
        .unwrap();
    let encoded_key = serde_json::to_string(&key).unwrap();
    let packed = {
        let attempt = context.0.lock().unwrap();
        let packed: Vec<u8> = attempt
            .scratch
            .scalar("SELECT value FROM entry WHERE key=?1", [&encoded_key])
            .unwrap();
        attempt
            .scratch
            .execute("UPDATE entry SET value=x'00' WHERE key=?1", [&encoded_key])
            .unwrap();
        packed
    };
    assert!(context.checkpoint_delta(100).is_err());
    {
        let attempt = context.0.lock().unwrap();
        assert_eq!(attempt.base, 0);
        assert_eq!(
            attempt
                .scratch
                .scalar::<i64>("SELECT count(*) FROM entry WHERE dirty=1", [])
                .unwrap(),
            2
        );
        attempt
            .scratch
            .execute(
                "UPDATE entry SET value=?1 WHERE key=?2",
                params![packed, encoded_key],
            )
            .unwrap();
    }
    let delta = context.checkpoint_delta(100).unwrap();
    assert_eq!(
        (delta.base, delta.through, delta.entries.len()),
        (0, 100, 2)
    );
    assert_reference(&delta, &row);
}

#[test]
fn committed_response_reopens_and_an_uncommitted_rewrite_stays_private() {
    let mut fixture = crate::fixture::writer::WriterArchive::new();
    let mut row = response(0);
    row.offset = 0;
    row.end = 1;
    fixture.write("scratch", vec![row.message.clone()]);
    let mut writer = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let (input, chain, version, through): (i64, i64, String, i64) = writer
        .query_row(
            "SELECT i.pk,p.chain,p.reading_version,p.parsed_upto FROM input_ref i
         JOIN parse_state p ON p.input_pk=i.pk WHERE i.id='input-scratch'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    let through = u64::try_from(through).unwrap();
    let mut context =
        Overlay::open(&fixture.archive, "input-scratch", Some(chain), &version, 0).unwrap();
    context
        .put(
            key(&row),
            Value::Response {
                row: Box::new(row.clone()),
            },
        )
        .unwrap();
    let delta = context.checkpoint_delta(through).unwrap();
    assert_reference(&delta, &row);
    let mut step = crate::ParseStep::new(Default::default(), Default::default(), 0..through, false);
    step.reading_version = version.clone();
    step.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        version.clone(),
        through,
        Some(serde_json::json!({"_polylogue_context":delta})),
    ));
    let tx = writer.transaction().unwrap();
    apply(
        &tx,
        input,
        chain,
        &step,
        &Default::default(),
        &crate::writer::resolve::IndexUpkeep::PerTransaction,
    )
    .unwrap();
    tx.commit().unwrap();
    drop(context);

    let reopen = || {
        Overlay::open(
            &fixture.archive,
            "input-scratch",
            Some(chain),
            &version,
            through,
        )
        .unwrap()
    };
    let mut context = reopen();
    let Some(Value::Response { row: stored }) = context.get(&key(&row)).unwrap() else {
        panic!("committed response absent")
    };
    assert_eq!(*stored, row);
    let mut rewritten = row.clone();
    rewritten.message.blocks = vec![Block::Text {
        text: "uncommitted replacement".into(),
    }];
    context
        .put(
            key(&rewritten),
            Value::Response {
                row: Box::new(rewritten.clone()),
            },
        )
        .unwrap();
    assert_reference(&context.checkpoint_delta(through + 1).unwrap(), &rewritten);
    drop(context);
    let Some(Value::Response { row: stored }) = reopen().get(&key(&row)).unwrap() else {
        panic!("prior committed response absent")
    };
    assert_eq!(*stored, row);
}
