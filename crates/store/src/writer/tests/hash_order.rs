use super::*;

pub(super) fn structured_message(reverse: bool) -> Message {
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
            run_by_script: false,
            call_id: Some("call".into()),
            tool: Some("synthetic".into()),
            action: polylogue_model::ToolAction::Other,
            input: data.clone(),
            command: None,
            path: None,
        },
        Block::ToolResult {
            reported_duration_ms: None,
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
fn version_hash_ignores_nested_json_member_order() {
    let first = structured_message(false);
    let second = structured_message(true);
    // Handwritten canonical JSON fixes the source facts independently of Message's serializer.
    let canonical = r#"{"at":"1970-01-01T00:00:01Z","at_precision":"instant","at_quality":"intrinsic","blocks":[{"text":"Synthetic body","type":"text"},{"action":"other","call_id":"call","input":{"array":[{"a":7,"z":"λ🙂"},null,false,"exact\ntext"]},"tool":"synthetic","type":"tool_call"},{"call_id":"call","data":{"array":[{"a":7,"z":"λ🙂"},null,false,"exact\ntext"]},"outcome":{"kind":"ok"},"output":"Synthetic output","type":"tool_result"},{"data":{"array":[{"a":7,"z":"λ🙂"},null,false,"exact\ntext"]},"type":"other","type_name":"synthetic"}],"id":"claude-code:hash-order","role":"assistant","session":"claude-code:hash-order","stop_reason":"end_turn"}"#;
    let mut value = serde_json::to_value(&first).unwrap();
    value.sort_all_objects();
    assert_eq!(serde_json::to_vec(&value).unwrap(), canonical.as_bytes());
    assert_eq!(
        blake3::Hash::from_bytes(crate::encoding::PreparedMessage::new(&first).unwrap().hash)
            .to_hex()
            .as_str(),
        "d94d0f12773c570a79ff4e9bd940775ec4c36efac5e63da9c3d9dfbb00e1e9ac",
        "the archive fixture stays stable across JSON map features"
    );
    assert_eq!(first, second, "member order does not change JSON values");
    assert_ne!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap(),
        "this probe must run with preserve_order; sorted maps cannot discriminate"
    );
    let mut counts = Vec::new();
    for prepared in [false, true] {
        let w = test_archive();
        for (input, rank, message) in [("first", 0, first.clone()), ("second", 1, second.clone())] {
            let mut b = batch(input, rank, 0, vec![row(1, message)], &["hash-order"]);
            if prepared {
                b.parse[0].prepare();
            }
            w.store_reading(b, Some(&SOURCE_RECORD)).unwrap();
        }
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .messages(&sid("hash-order"), false)
                .unwrap()
                .as_slice(),
            std::slice::from_ref(&first)
        );
        let rows = reader
            .query(
                "SELECT count(*) AS sources, count(DISTINCT version_hash) AS n FROM message_source",
                &[],
            )
            .unwrap();
        assert_eq!(
            rows[0]["sources"], 2,
            "both physical source associations survive"
        );
        let count = rows[0]["n"].as_u64().unwrap();
        let hashes = reader
            .query(
                "SELECT hex(version_hash) AS hash FROM message_source ORDER BY input_pk",
                &[],
            )
            .unwrap();
        println!("prepared={prepared}: hashes={count}, {hashes:?}");
        counts.push((prepared, count));
    }
    assert!(
        counts.iter().all(|(_, count)| *count == 1),
        "equal message values require one hash in both paths: {counts:?}"
    );
}

#[test]
fn distinct_structured_values_remain_inspectable_versions() {
    for prepared in [false, true] {
        let first = structured_message(false);
        let mut different = first.clone();
        if let Some(Block::Other { data, .. }) = different.blocks.last_mut() {
            data["array"].as_array_mut().unwrap().reverse();
        }
        assert_ne!(first, different);
        let w = test_archive();
        for (input, rank, message) in [
            ("first", 0, first.clone()),
            ("different", 1, different.clone()),
        ] {
            let mut b = batch(input, rank, 0, vec![row(1, message)], &["hash-order"]);
            if prepared {
                b.parse[0].prepare();
            }
            w.store_reading(b, Some(&SOURCE_RECORD)).unwrap();
        }
        assert!(w.writer.flush().unwrap().errors.is_empty());
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .query(
                    "SELECT count(DISTINCT version_hash) AS n FROM message_source",
                    &[]
                )
                .unwrap()[0]["n"],
            2
        );
    }
}
