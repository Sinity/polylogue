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
        .value = Some(SelectedMessage::Value {
        columns: crate::writer::message_columns::MessageColumns::new(&value.message),
        value: Box::new(value.clone()),
    });
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
            matches!(&c.value, Some(SelectedMessage::Version { message, .. }) if *message == messages[0].id)))
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
    richer.model = Some("claude-opus-4-1".into());
    richer.usage.input = Some(1_000_000);
    richer.usage.output = Some(1_000_000);
    let mut b = reading("export", vec![richer], Vec::new());
    b.input.route = Route::Export;
    resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    let choices = &b.parse[0].resolution.as_ref().unwrap().choices;
    let choice = choices.messages.values().flatten().next().unwrap();
    assert!(matches!(choice.value, Some(SelectedMessage::Value { .. })));
    let selected = choices.selected_message(choice).unwrap().unwrap();
    assert_eq!(selected.message.model.as_deref(), Some("claude-opus-4-1"));
    assert_eq!(
        selected.prepared.items,
        polylogue_model::read::message_entries(
            &selected.message,
            &selected.prepared.block_selection.block_ordinals,
            Default::default()
        )
    );
    apply_prepared(&mut conn, b).0.unwrap();
    let columns: (i64, i64) = conn
        .query_row("SELECT words,cost_nusd FROM message", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(columns, (2, 90_000_000_000));

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
    assert!(matches!(choice.value, Some(SelectedMessage::Value { .. })));
    apply_prepared(&mut conn, b).0.unwrap();
}

#[test]
fn selected_columns_keep_exact_metadata_and_change_with_a_replacement_reading() {
    let mut conn = database();
    let mut message = text_message(
        "columns",
        "s",
        Role::Assistant,
        "alpha\u{2003}beta\n gamma",
        Some("end_turn"),
    );
    message.blocks.push(Block::Thinking {
        text: Some("delta epsilon".into()),
        signature: None,
        redacted: false,
    });
    message.blocks.push(Block::ToolResult {
        reported_duration_ms: None,
        call_id: None,
        outcome: ToolOutcome::Ok,
        output: Some("tool output does not add prose words".into()),
        data: None,
        attachments: Vec::new(),
    });
    message.model = Some("claude-opus-4-1".into());
    message.usage.input = Some(1_000_000);
    message.usage.output = Some(1_000_000);
    message.reported_cost_usd = Some("0.123456789".parse().unwrap());
    message.production.iterations = Some(3);
    message.production.phase = Some("analysis".into());
    message.attrs.insert(
        "nested".into(),
        serde_json::json!({"exact": 9007199254740993_i64}),
    );
    let mut b = reading("column-source", vec![message.clone()], Vec::new());
    b.parse[0].output.messages[0].offset = 7;
    b.parse[0].output.messages[0].end = 77;
    resolve::prepare_index(&conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    let choices = &b.parse[0].resolution.as_ref().unwrap().choices;
    let encoded = serde_json::to_vec(choices).unwrap();
    let decoded = crate::encoding::read_json(&encoded).unwrap();
    b.parse[0].resolution.as_mut().unwrap().choices = decoded;
    apply_prepared(&mut conn, b).0.unwrap();
    let (role, author, words, cost, provider_cost, attrs, start, end): (
        String,
        String,
        i64,
        i64,
        i64,
        String,
        i64,
        i64,
    ) = conn
        .query_row(
            "SELECT role,written_by,words,cost_nusd,provider_cost_nusd,attrs_json,
                    source_start,source_end FROM message",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(
        (role.as_str(), author.as_str(), words),
        ("assistant", "model", 5)
    );
    assert_eq!((cost, provider_cost), (90_000_000_000, 123_456_789));
    assert_eq!((start, end), (7, 77));
    assert_eq!(
        serde_json::from_str::<Json>(&attrs).unwrap(),
        serde_json::json!({
            "attrs": {"nested": {"exact": 9007199254740993_i64}},
            "reported_cost_usd": 0.123456789,
            "production": {"iterations": 3, "phase": "analysis"}
        })
    );
    assert_eq!(
        crate::versions::read_message(&conn, &message.content_hash().unwrap().0)
            .unwrap()
            .unwrap()
            .0,
        message
    );

    message.role = Role::User;
    message.prompt_kind = Some(polylogue_model::PromptKind::Prose);
    message.blocks = vec![Block::Text {
        text: "new content".into(),
    }];
    message.model = None;
    message.usage = Default::default();
    message.reported_cost_usd = None;
    message.production = Default::default();
    message.attrs.clear();
    let mut b = reading("column-source", vec![message.clone()], Vec::new());
    b.parse[0].reset = true;
    apply(&mut conn, b).0.unwrap();
    let columns: (String, String, String, i64, Option<i64>, Option<i64>, Option<String>) = conn
        .query_row(
            "SELECT role,written_by,prompt_kind,words,cost_nusd,provider_cost_nusd,attrs_json FROM message",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                row.get(4)?, row.get(5)?, row.get(6)?)),
        )
        .unwrap();
    assert_eq!(
        columns,
        (
            "user".into(),
            "human".into(),
            "prose".into(),
            2,
            None,
            None,
            None
        )
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
