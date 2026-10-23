use super::*;

fn structured_message(reverse: bool) -> Message {
    let entries = if reverse {
        [("z", Json::String("λ🙂".into())), ("a", Json::from(7))]
    } else {
        [("a", Json::from(7)), ("z", Json::String("λ🙂".into()))]
    };
    let nested = Json::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    );
    let data = serde_json::json!({"array":[nested, null, false, "exact\ntext"]});
    let mut message = text_message(
        "hash-order",
        "hash-order",
        Role::Assistant,
        "Synthetic body",
        Some("end_turn"),
    );
    message.blocks.extend([
        Block::ToolCall {
            call_id: Some("call".into()),
            tool: Some("synthetic".into()),
            action: polylogue_model::ToolAction::Other,
            input: data.clone(),
            command: None,
            path: None,
        },
        Block::ToolResult {
            call_id: Some("call".into()),
            outcome: polylogue_model::ToolOutcome::Ok,
            output: Some("Synthetic output".into()),
            data: Some(data.clone()),
            attachments: vec![],
        },
        Block::Other {
            type_name: "synthetic".into(),
            data,
        },
    ]);
    message
}

#[test]
#[ignore = "discriminating probe requires --features serde_json/preserve_order"]
fn message_version_identity_ignores_nested_json_member_order() {
    let first = structured_message(false);
    let second = structured_message(true);
    assert_eq!(first, second, "member order does not change JSON values");
    assert_ne!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap(),
        "this probe must run with preserve_order; sorted maps cannot discriminate"
    );
    let mut counts = Vec::new();
    for prepared in [false, true] {
        let w = world();
        for (input, rank, message) in [("first", 0, first.clone()), ("second", 1, second.clone())] {
            let mut b = batch(input, rank, 0, vec![row(1, message)], &["hash-order"]);
            if prepared {
                b.parse[0].prepare();
            }
            w.writer.send(b).unwrap();
        }
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader.messages(&sid("hash-order"), false).unwrap(),
            [first.clone()]
        );
        let sources = reader.message_versions(&first.id).unwrap();
        assert_eq!(
            sources.len(),
            2,
            "both physical source associations survive"
        );
        assert!(
            sources.iter().all(|version| version.message == first),
            "exact structured values survive both read paths"
        );
        let rows = reader
            .query("SELECT count(*) AS n FROM message_version", &[])
            .unwrap();
        let count = rows[0]["n"].as_u64().unwrap();
        let hashes = reader
            .query(
                "SELECT hex(version_hash) AS hash FROM message_source ORDER BY input_pk",
                &[],
            )
            .unwrap();
        println!("prepared={prepared}: versions={count}, hashes={hashes:?}");
        counts.push((prepared, count));
    }
    assert!(
        counts.iter().all(|(_, count)| *count == 1),
        "equal message values require one retained logical hash in both paths: {counts:?}"
    );
}

#[test]
#[ignore = "migration probe; does not change production hash semantics"]
fn fully_sorted_message_json_changes_legacy_struct_hash_even_without_json_blocks() {
    let message = text_message(
        "legacy",
        "legacy",
        Role::User,
        "Synthetic unchanged text",
        None,
    );
    let legacy = serde_json::to_vec(&message).unwrap();
    let mut canonical = serde_json::to_value(&message).unwrap();
    canonical.sort_all_objects();
    let canonical = serde_json::to_vec(&canonical).unwrap();
    assert_ne!(legacy, canonical);
    let old_hash = blake3::hash(&legacy);
    let new_hash = blake3::hash(&canonical);
    println!("plain-text legacy={old_hash} fully-sorted={new_hash}");
    assert_ne!(
        old_hash, new_hash,
        "a full canonical rule needs an explicit index-generation boundary"
    );
    assert_eq!(
        serde_json::from_slice::<Message>(&canonical).unwrap(),
        message
    );
}
