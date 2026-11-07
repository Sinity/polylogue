use super::*;
use crate::writer::attachments::normalize;
use polylogue_model::Attachment;

#[test]
fn absent_and_present_blobs_keep_other_prepared_values() {
    let archive = Connection::open_in_memory().unwrap();
    archive
        .execute_batch("CREATE TABLE blob(hash BLOB, size INTEGER)")
        .unwrap();
    let present = BlobHash::of(b"hello");
    archive
        .execute("INSERT INTO blob VALUES (?1, 5)", [&present.0[..]])
        .unwrap();
    for prepared in [false, true] {
        let mut message = text_message("m", "s", Role::Assistant, "text", Some("end_turn"));
        message.blocks.push(Block::ToolCall {
            call_id: Some("call".into()),
            tool: Some("Bash".into()),
            action: ToolAction::Shell,
            input: serde_json::json!({"command":"printf hello"}),
            command: None,
            path: None,
            run_by_script: false,
        });
        message.blocks.push(Block::ToolResult {
            reported_duration_ms: None,
            call_id: Some("call".into()),
            outcome: ToolOutcome::Ok,
            output: Some("hello".into()),
            data: Some(serde_json::json!({"answer":42})),
            attachments: vec![
                Attachment {
                    blob: Some(BlobHash::of(b"absent")),
                    ..Attachment::default()
                },
                Attachment {
                    blob: Some(present),
                    ..Attachment::default()
                },
            ],
        });
        message.blocks.push(Block::Other {
            type_name: "detail".into(),
            data: serde_json::json!({"kept":true}),
        });
        let mut original = batch("input1", 0, 1, vec![row(0, message.clone())], &["s"]);
        if prepared {
            original.parse[0].prepare();
        }
        normalize(&mut original, &archive, &HashMap::new()).unwrap();
        let indexed = original;
        let actual = &indexed.parse[0].output.messages[0].message;
        if let Block::ToolResult {
            reported_duration_ms: None,
            attachments,
            ..
        } = &mut message.blocks[2]
        {
            attachments[0].blob = None;
            attachments[1].size = Some(5);
        }
        assert_eq!(actual, &message);
        assert_eq!(actual.session, sid("s"));
        if prepared {
            assert_eq!(
                indexed.parse[0].prepared.messages[0],
                PreparedMessage::new(&message).unwrap()
            );
        }
    }
}
