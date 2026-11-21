use super::*;
use crate::writer::resolve::membership::{Choices, SelectedMessage};

#[test]
fn fresh_message_selection_shares_one_value_through_preparation_and_spill() {
    let mut conn = database();
    let body = "unique synthetic version payload ".repeat(8192);
    let message = text_message("large", "s", Role::Assistant, &body, Some("end_turn"));
    let hash = message.content_hash().unwrap();
    let mut b = reading(
        "large-source",
        vec![message.clone()],
        vec![reference(&message, "large")],
    );
    b.parse[0].prepare();
    let (_, input_bytes) = crate::text_rows::PreparedTexts::count_input_bytes(|| {
        resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    });
    assert_eq!(input_bytes, 0, "resolver rehashed the prepared source text");
    let choices = &b.parse[0].resolution.as_ref().unwrap().choices;
    let choice = choices.messages.values().next().unwrap().as_ref().unwrap();
    let value = choices.selected_message(choice).unwrap().unwrap();
    let version = choices.message_versions.get(&hash).unwrap();
    assert!(std::ptr::eq(value, version.value.as_ref()));
    assert_eq!(value.message, message);
    let cloned = choices.clone();
    assert!(Arc::ptr_eq(
        &version.value,
        &cloned.message_versions[&hash].value
    ));

    let encoded = serde_json::to_vec(choices).unwrap();
    assert_eq!(
        std::str::from_utf8(&encoded)
            .unwrap()
            .matches(body.as_str())
            .count(),
        1
    );
    // The existing owned selection is an independent value-equivalent path:
    // keeping it would put a second complete payload into the spill record.
    let mut owned = choices.clone();
    owned
        .messages
        .values_mut()
        .next()
        .unwrap()
        .as_mut()
        .unwrap()
        .value = Some(SelectedMessage::Value(Box::new(value.clone())));
    let owned_encoded = serde_json::to_vec(&owned).unwrap();
    assert_eq!(
        std::str::from_utf8(&owned_encoded)
            .unwrap()
            .matches(body.as_str())
            .count(),
        2
    );
    assert!(owned_encoded.len() - encoded.len() >= body.len());
    println!(
        "fresh value body_bytes={} referenced_spill_bytes={} owned_spill_bytes={}",
        body.len(),
        encoded.len(),
        owned_encoded.len()
    );

    let decoded: Choices = crate::encoding::read_json(&encoded).unwrap();
    assert_eq!(&decoded, choices);
    b.parse[0].resolution.as_mut().unwrap().choices = decoded;
    let mut other = b.clone();
    other.parse[0].resolution.as_mut().unwrap().choices = owned;
    let mut other_conn = database();
    apply_prepared(&mut conn, b).0.unwrap();
    apply_prepared(&mut other_conn, other).0.unwrap();
    for database in [&conn, &other_conn] {
        assert_eq!(
            crate::versions::read_message(database, &hash.0)
                .unwrap()
                .unwrap()
                .0,
            message
        );
    }
    let entries = |database: &Connection| {
        let session: i64 = database
            .query_row(
                "SELECT pk FROM session WHERE id=?1",
                [sid("s").to_string()],
                |r| r.get(0),
            )
            .unwrap();
        crate::items::computed_entries(database, session).unwrap()
    };
    assert_eq!(entries(&conn), entries(&other_conn));
}

#[test]
fn selected_version_references_refuse_missing_mismatched_or_sparse_values_atomically() {
    for defect in [
        "missing",
        "different-message",
        "block_selection",
        "prepared-id",
    ] {
        let mut conn = database();
        let messages = ["first", "second"].map(|id| {
            text_message(
                id,
                "s",
                Role::Assistant,
                &format!("{id} payload"),
                Some("end_turn"),
            )
        });
        let mut b = reading("reference-source", messages.to_vec(), Vec::new());
        resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
        let retry = b.clone();
        let choices = &mut b.parse[0].resolution.as_mut().unwrap().choices;
        let pk = *choices.messages.iter().find(|(_, c)| c.as_ref().is_some_and(|c|
            matches!(&c.value, Some(SelectedMessage::Version { message }) if *message == messages[0].id)))
            .unwrap().0;
        let hash = messages[0].content_hash().unwrap();
        match defect {
            "missing" => {
                choices.message_versions.remove(&hash);
            }
            "different-message" => {
                choices
                    .messages
                    .get_mut(&pk)
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .hash = messages[1].content_hash().unwrap().0.to_vec();
            }
            "block_selection" => {
                Arc::make_mut(&mut choices.message_versions.get_mut(&hash).unwrap().value)
                    .prepared
                    .block_selection
                    .block_count += 1;
            }
            "prepared-id" => {
                Arc::make_mut(&mut choices.message_versions.get_mut(&hash).unwrap().value)
                    .prepared
                    .id = messages[1].id.clone();
            }
            _ => unreachable!(),
        }
        let error = apply_prepared(&mut conn, b).0.unwrap_err();
        assert!(matches!(error, StoreError::Invalid(_)), "{defect}: {error}");
        for table in ["message", "message_version", "text", "input_ref"] {
            let count: i64 = conn
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 0, "{defect}: {table}");
        }
        apply_prepared(&mut conn, retry).0.unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM message", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }
}

#[test]
fn composed_and_sparse_selections_keep_their_owned_metadata_and_values() {
    let mut conn = database();
    let base = text_message(
        "mixed",
        "s",
        Role::Assistant,
        "shared body",
        Some("end_turn"),
    );
    apply(&mut conn, reading("live", vec![base.clone()], Vec::new()))
        .0
        .unwrap();
    let mut richer = base.clone();
    richer.model = Some("synthetic-model".into());
    let mut b = reading("export", vec![richer], Vec::new());
    b.input.route = Route::Export;
    resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    let choices = &b.parse[0].resolution.as_ref().unwrap().choices;
    let choice = choices.messages.values().flatten().next().unwrap();
    assert!(matches!(choice.value, Some(SelectedMessage::Value(_))));
    let selected = choices.selected_message(choice).unwrap().unwrap();
    assert_eq!(selected.message.model.as_deref(), Some("synthetic-model"));
    assert_eq!(
        selected.prepared.items,
        polylogue_model::read::message_entries(
            &selected.message,
            &selected.prepared.block_selection.block_ordinals,
            Default::default()
        )
    );
    apply_prepared(&mut conn, b).0.unwrap();

    let mut full = text_message("sparse", "s", Role::Assistant, "first", Some("end_turn"));
    full.blocks.push(Block::Text {
        text: "second".into(),
    });
    let mut part = full.clone();
    part.blocks.remove(0);
    let mut b = reading("sparse", vec![part], Vec::new());
    b.parse[0].output.block_selections.insert(
        0,
        polylogue_model::bundle::MessageProjection {
            source_version: full.content_hash().unwrap(),
            block_count: 2,
            block_ordinals: vec![1],
        },
    );
    resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    let choice = b.parse[0]
        .resolution
        .as_ref()
        .unwrap()
        .choices
        .messages
        .values()
        .flatten()
        .next()
        .unwrap();
    assert!(matches!(choice.value, Some(SelectedMessage::Value(_))));
    apply_prepared(&mut conn, b).0.unwrap();
}

#[test]
fn complete_staged_reading_reopens_its_selected_version_without_payload_duplication() {
    let w = test_archive();
    let body = "disk-staged unique payload ".repeat(4096);
    let message = text_message("disk", "s", Role::Assistant, &body, Some("end_turn"));
    let b = reading(
        "disk-source",
        vec![message.clone()],
        vec![reference(&message, "disk")],
    );
    let mut b = w.commit_source(b, &SOURCE_RECORD).unwrap();
    let mut staged = ChainReading::new().unwrap();
    staged.push(&mut b.parse[0]).unwrap();
    w.writer
        .replace_chain(b.input, 1, b.length, staged)
        .unwrap();
    w.writer.derive().unwrap();
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    assert_eq!(
        crate::versions::read_message(&conn, &message.content_hash().unwrap().0)
            .unwrap()
            .unwrap()
            .0,
        message
    );
    assert!(
        conn.prepare("PRAGMA foreign_key_check")
            .unwrap()
            .query([])
            .unwrap()
            .next()
            .unwrap()
            .is_none()
    );
    super::super::items::assert_matches_rows(&w, "s");
}

#[test]
fn referenced_complete_values_preserve_unknown_time_and_prepared_headers() {
    for unknown in [false, true] {
        let mut conn = database();
        let mut message = text_message(
            "clock",
            "s",
            Role::Assistant,
            "clock value",
            Some("end_turn"),
        );
        if unknown {
            message.at = None;
            message.at_quality = polylogue_model::TimeQuality::Unknown;
            message.at_precision = None;
            message.at_until = None;
        }
        let hash = message.content_hash().unwrap();
        let mut b = reading(
            "clock-source",
            vec![message.clone()],
            vec![reference(&message, "clock")],
        );
        resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
        let choices = &b.parse[0].resolution.as_ref().unwrap().choices;
        let choice = choices.messages.values().flatten().next().unwrap();
        assert!(matches!(
            choice.value,
            Some(SelectedMessage::Version { .. })
        ));
        let selected = choices.selected_message(choice).unwrap().unwrap();
        assert_eq!(
            selected.prepared.items[0].at,
            message.at.map(Timestamp::unix_ms)
        );
        apply_prepared(&mut conn, b).0.unwrap();
        let stored = crate::versions::read_message(&conn, &hash.0)
            .unwrap()
            .unwrap()
            .0;
        assert_eq!(stored, message);
    }
}

#[test]
fn canonical_block_payloads_keep_historical_and_selected_values_exact() {
    let mut conn = database();
    let mut original = text_message(
        "payload",
        "s",
        Role::Assistant,
        "first exact text",
        Some("end_turn"),
    );
    original.blocks.push(
        serde_json::from_value(serde_json::json!({
            "type":"tool_result", "call_id":"call", "outcome":{"kind":"ok"},
            "output":"result text", "data":{"preserved":[null,false,"structured"]},
            "attachments":[{"direction":"output","name":"artifact"}]
        }))
        .unwrap(),
    );
    let mut newer = original.clone();
    newer.blocks[0] = Block::Text {
        text: "changed selected text".into(),
    };
    let old_hash = original.content_hash().unwrap();
    let new_hash = newer.content_hash().unwrap();
    for (input, message) in [("old", original.clone()), ("new", newer.clone())] {
        let (result, _) = apply(
            &mut conn,
            reading(
                input,
                vec![message.clone()],
                vec![reference(&message, "key")],
            ),
        );
        result.unwrap();
    }
    for (hash, expected) in [(old_hash, &original), (new_hash, &newer)] {
        assert_eq!(
            crate::versions::read_message(&conn, &hash.0)
                .unwrap()
                .unwrap()
                .0,
            *expected
        );
    }
    let blocks: i64 = conn
        .query_row("SELECT count(*) FROM message_version_block", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        blocks, 4,
        "two canonical block positions per retained version"
    );
    let mismatches: i64 = conn.query_row("SELECT count(*) FROM block b JOIN message m ON m.pk=b.message_pk
        JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
        JOIN message_version_block p ON p.pk=b.payload_pk
        WHERE p.version_pk<>s.version_pk OR p.idx<>b.idx OR p.text_id IS NOT b.text_id",[],|r|r.get(0)).unwrap();
    assert_eq!(mismatches, 0);
    let stored_bytes: i64 = conn
        .query_row(
            "SELECT sum(length(body)+coalesce(length(data),0)) FROM message_version_block",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let headers: i64 = conn
        .query_row(
            "SELECT sum(length(body)) FROM message_version_body",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let check: i64 = conn
        .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(check, 0);
    println!(
        "canonical retained_versions=2 blocks={blocks} block_payload_bytes={stored_bytes} header_bytes={headers} selected_payload_copies=0"
    );
}

#[test]
fn sparse_payload_enrichment_keeps_block_keys_and_rolls_back_atomically() {
    let mut conn = database();
    let mut complete = text_message(
        "enrich",
        "s",
        Role::Assistant,
        "first block",
        Some("end_turn"),
    );
    complete.blocks.push(Block::Text {
        text: "second block".into(),
    });
    let hash = complete.content_hash().unwrap();
    let mut partial = complete.clone();
    partial.blocks.truncate(1);
    let projection = polylogue_model::bundle::MessageProjection {
        source_version: hash,
        block_count: 2,
        block_ordinals: vec![0],
    };
    let prepared = crate::encoding::PreparedMessage::projected(&partial, &projection).unwrap();
    let texts = crate::text_rows::TextIds::new(&conn, false, 1 << 20).unwrap();
    let packed = crate::text_rows::PreparedTexts::pack(["first block"]).unwrap();
    let registered = crate::versions::store_message(
        &conn,
        &partial,
        &prepared,
        crate::versions::TextRows {
            packed: &packed,
            texts: &texts,
            membership: None,
            upkeep: &resolve::IndexUpkeep::PerTransaction,
        },
        false,
        None,
    )
    .unwrap();
    let first = registered.blocks().unwrap()[&0].pk;
    drop(registered);
    let expected = crate::versions::read_message(&conn, &hash.0).unwrap();
    {
        let tx = conn.transaction().unwrap();
        let identity_transaction = texts.transaction();
        let all = crate::encoding::PreparedMessage::new(&complete).unwrap();
        let packed =
            crate::text_rows::PreparedTexts::pack(["first block", "second block"]).unwrap();
        let registered = crate::versions::store_message(
            &tx,
            &complete,
            &all,
            crate::versions::TextRows {
                packed: &packed,
                texts: &texts,
                membership: None,
                upkeep: &resolve::IndexUpkeep::PerTransaction,
            },
            true,
            None,
        )
        .unwrap();
        assert_eq!(registered.blocks().unwrap()[&0].pk, first);
        assert_eq!(
            crate::versions::read_message(&tx, &hash.0)
                .unwrap()
                .unwrap()
                .0,
            complete
        );
        drop(registered);
        tx.rollback().unwrap();
        drop(identity_transaction);
    }
    assert_eq!(
        crate::versions::read_message(&conn, &hash.0).unwrap(),
        expected
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM message_version_block", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
}
