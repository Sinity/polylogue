use super::*;
use serde_json::json;

fn prompt(chunks: Vec<Json>) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "runSettings": {"temperature": 1.0, "model": "models/gemini-3-pro-preview",
                        "thinkingLevel": "HIGH"},
        "systemInstruction": {"text": "Answer briefly."},
        "chunkedPrompt": {"chunks": chunks, "pendingInputs": [{"text": "", "role": "user"}]}
    }))
    .unwrap()
}

fn parse(native: &str, bytes: &[u8]) -> crate::parse::Step {
    let parser = AiStudio {
        native: native.into(),
        title: Some("Electrons".into()),
        state: PromptState::default(),
    };
    crate::export::fixture::parse(&mut Parse::Export(Box::new(parser)), bytes).unwrap()
}

#[test]
fn chunks_become_messages_with_thoughts_split_out() {
    let bytes = prompt(vec![
        json!({"text": "what is an electron", "role": "user", "tokenCount": 6,
               "createTime": "2026-05-17T12:27:24.102Z"}),
        json!({"text": "Considering charge.", "role": "model", "isThought": true,
               "thinkingBudget": -1, "tokenCount": 40,
               "parts": [{"text": "Considering ", "thought": true},
                         {"text": "charge.", "thought": true, "thoughtSignature": "sig"}]}),
        json!({"text": "A lepton.", "role": "model", "tokenCount": 3, "finishReason": "STOP",
               "parts": [{"text": "A lepton."}]}),
        json!({"text": "watch this", "role": "user", "youtubeVideo": {"id": "abc123"}}),
        json!({"role": "model", "errorMessage": "An internal error has occurred."}),
    ]);
    let step = parse("Electrons", &bytes);
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs.len(), 6);
    assert_eq!(msgs[0].role, Role::System);
    assert_eq!(step.output.messages[0].offset, 0);
    assert_eq!(step.output.messages[0].end, bytes.len() as u64);
    assert_eq!(step.output.events[0].range, Some((0, bytes.len() as u64)));
    assert_eq!(msgs[1].text(), "what is an electron");
    assert_eq!(msgs[1].usage.input, Some(6));
    assert!(msgs[1].at.is_some());
    match &msgs[2].blocks[0] {
        Block::Thinking {
            text, signature, ..
        } => {
            assert_eq!(text.as_deref(), Some("Considering charge."));
            assert_eq!(signature.as_deref(), Some("sig"));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(msgs[2].usage.reasoning, Some(40));
    assert_eq!(msgs[3].usage.output, Some(3));
    assert_eq!(msgs[3].stop_reason.as_deref(), Some("STOP"));
    assert_eq!(msgs[3].model.as_deref(), Some("gemini-3-pro-preview"));
    assert_eq!(
        msgs[3].parent.as_ref().unwrap().native().as_str(),
        "9:Electrons#position:1:1"
    );
    assert!(
        matches!(&msgs[4].blocks[1], Block::File(a) if a.source_id.as_deref() == Some("abc123"))
    );
    assert!(matches!(&msgs[5].blocks[0], Block::Other { type_name, .. } if type_name == "error"));
    // A chunk without an ID keeps its identity by its place.
    assert_eq!(msgs[1].id.native().as_str(), "9:Electrons#position:1:0");
    assert_eq!(
        step.output.sessions[0].title.as_ref().unwrap().text,
        "Electrons"
    );
    assert_eq!(
        step.output.sessions[0].title.as_ref().unwrap().source,
        TitleSource::FileName
    );
    assert!(
        step.counts.unread_fields.is_empty(),
        "{:?}",
        step.counts.unread_fields
    );
    assert_eq!(step.counts.records, 6);
}

#[test]
fn inline_bytes_are_stored_and_drive_files_keep_their_id() {
    let bytes = prompt(vec![
        json!({"role": "user", "inlineImage": {"mimeType": "image/png", "data": "aGVsbG8="},
               "driveDocument": {"id": "1AbC"}}),
        json!({"role": "model", "executableCode": {"language": "PYTHON", "code": "print(1)"},
               "codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1\n"}}),
    ]);
    let step = parse("p", &bytes);
    assert_eq!(step.output.blobs.len(), 1);
    assert_eq!(step.output.blobs[0].bytes, b"hello");
    let user = &step.output.messages[1].message;
    assert!(matches!(&user.blocks[0], Block::File(a) if a.source_id.as_deref() == Some("1AbC")));
    assert!(matches!(&user.blocks[1], Block::Image(a) if a.blob.is_some()));
    let model = &step.output.messages[2].message;
    assert!(
        matches!(&model.blocks[0], Block::ToolCall { command: Some(c), .. } if c == "print(1)")
    );
    assert!(matches!(
        &model.blocks[1],
        Block::ToolResult {
            outcome: ToolOutcome::Ok,
            ..
        }
    ));
}

#[test]
fn code_and_images_inside_parts_are_read() {
    let bytes = prompt(vec![json!({"role": "model", "text": "plot", "parts": [
        {"text": "plot"},
        {"executableCode": {"language": "PYTHON", "code": "plot()"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_FAILED", "output": "boom"}},
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}
    ], "safetyRatings": [], "thoughtSignatures": ["s"], "isEdited": true})]);
    let step = parse("p", &bytes);
    let m = &step.output.messages[1].message;
    assert!(
        matches!(&m.blocks[1], Block::ToolCall { call_id: Some(c), .. } if c == "1:p#position:1:0:code1")
    );
    assert!(matches!(
        &m.blocks[2],
        Block::ToolResult { call_id: Some(c), outcome: ToolOutcome::Error { .. }, .. } if c == "1:p#position:1:0:code1"
    ));
    assert!(matches!(&m.blocks[3], Block::Image(a) if a.blob.is_some()));
    assert!(
        step.counts.unread_fields.is_empty(),
        "{:?}",
        step.counts.unread_fields
    );
}

#[test]
fn a_branched_prompt_links_to_the_prompt_it_came_from() {
    let bytes = prompt(vec![json!({"role": "user", "text": "go on",
                                   "branchParent": {"promptId": "1Parent"}})]);
    let step = parse("1Child", &bytes);
    let link = &step.output.links[0];
    assert_eq!(link.kind, LinkKind::Fork);
    assert_eq!(link.parent.native().as_str(), "1Parent");
    assert_eq!(link.child.native().as_str(), "1Child");
}

#[test]
fn recognizes_prompts_by_their_first_key() {
    assert!(recognize(
        Path::new("/d/Electrons.json"),
        b"{\n  \"runSettings\": {"
    ));
    assert!(recognize(
        Path::new("/d/Untitled prompt"),
        b"{\"chunkedPrompt\": {"
    ));
    assert!(!recognize(Path::new("/d/x.json"), b"{\"sessionId\": 1}"));
    assert_eq!(
        native_of(Path::new(
            "/t.zip!/Takeout/Drive/Google AI Studio/Electrons"
        ))
        .as_deref(),
        Some("Electrons")
    );
    assert_eq!(
        title_of(Path::new("/s/polylogue/drive/ai-studio/1AbC.json")),
        None
    );
}

#[test]
fn the_listing_names_the_prompts() {
    let bytes = serde_json::to_vec(&json!({"files": [
        {"id": "1AbC", "name": "Electrons", "mimeType": "application/vnd.google-makersuite.prompt",
         "createdTime": "2026-05-17T12:20:00Z", "modifiedTime": "2026-05-17T12:30:00Z"}
    ]}))
    .unwrap();
    let step =
        crate::export::fixture::parse(&mut Parse::Export(Box::new(Listing::default())), &bytes)
            .unwrap();
    let s = &step.output.sessions[0];
    assert_eq!(s.id.native().as_str(), "1AbC");
    assert_eq!(s.title.as_ref().unwrap().text, "Electrons");
    assert!(step.output.messages.is_empty());
    assert!(recognize(
        Path::new("/s/polylogue/drive/ai-studio/files.json"),
        br#"{"files": ["#
    ));
}

#[test]
fn listings_distinguish_empty_invalid_and_refused_entries() {
    for (bytes, records, refused, sessions) in [
        (br#"{"files":[]}"#.as_slice(), 0, 0, 0),
        (br#"{"files":false}"#.as_slice(), 1, 1, 0),
        (
            br#"{"files":[{"name":"missing ID"},{"id":"p","name":"Useful"}]}"#.as_slice(),
            2,
            1,
            1,
        ),
        (
            br#"{"files":[{"id":"not-a-message"}]} trailing"#.as_slice(),
            1,
            1,
            0,
        ),
    ] {
        let step =
            crate::export::fixture::parse(&mut Parse::Export(Box::new(Listing::default())), bytes)
                .unwrap();
        assert_eq!(step.counts.records, records);
        assert_eq!(step.counts.unreadable.values().sum::<u64>(), refused);
        assert_eq!(step.output.sessions.len(), sessions);
    }
}

#[test]
fn headers_and_unusable_chunks_do_not_make_a_session() {
    for chunks in [
        json!([]),
        json!(false),
        json!([null, {}, {"role": "future", "text": "unknown"}]),
    ] {
        let bytes = serde_json::to_vec(&json!({
            "runSettings": {"model": "models/test"},
            "systemInstruction": {"text": "A system header alone is not a chat."},
            "chunkedPrompt": {"chunks": chunks}
        }))
        .unwrap();
        let step = parse("empty", &bytes);
        assert!(
            step.output.sessions.is_empty(),
            "{chunks}: {:?}",
            step.output.sessions
        );
        assert!(step.output.messages.is_empty(), "{chunks}");
        assert!(step.output.events.is_empty(), "{chunks}");
    }
}

#[test]
fn adjacent_chunks_keep_their_own_ids_usage_and_source_ranges() {
    let chunks = vec![
        json!({"role": "user", "id": "question", "text": "ask", "tokenCount": 3}),
        json!({"role": "model", "id": "thought", "isThought": true, "text": "think", "tokenCount": 7}),
        json!({"role": "model", "text": "answer", "tokenCount": 5}),
        json!({"role": "unknown", "text": "unreadable"}),
        json!({"role": "user", "text": "next", "inlineImage": {"mimeType": "image/png", "data": "aGk="}}),
    ];
    let bytes = prompt(chunks.clone());
    let step = parse("p", &bytes);
    let rows = &step.output.messages;
    assert_eq!(rows.len(), 5);
    let expected = [
        "1:p#id:8:question",
        "1:p#id:7:thought",
        "1:p#position:1:2",
        "1:p#position:1:4",
    ];
    for (i, row) in rows[1..].iter().enumerate() {
        assert_eq!(row.message.id.native().as_str(), expected[i]);
        assert_eq!(row.message.parent.as_ref(), Some(&rows[i].message.id));
        let source: Json =
            serde_json::from_slice(&bytes[row.offset as usize..row.end as usize]).unwrap();
        assert_eq!(source, chunks[if i == 3 { 4 } else { i }]);
    }
    assert_eq!(rows[2].message.usage.reasoning, Some(7));
    assert_eq!(rows[3].message.usage.output, Some(5));
    assert_eq!(step.output.blobs[0].bytes, b"hi");
    assert_eq!(step.counts.records, 6);
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 1);
}

#[test]
fn native_ids_positions_and_the_system_header_cannot_name_the_same_message() {
    let bytes = prompt(vec![
        json!({"role": "user", "id": "1", "text": "native ID"}),
        json!({"role": "model", "text": "position one"}),
        json!({"role": "user", "id": "system", "text": "native system ID"}),
        json!({"role": "model", "id": "id:1:1", "text": "separators"}),
    ]);
    let step = parse("p", &bytes);
    let ids: Vec<_> = step
        .output
        .messages
        .iter()
        .map(|r| r.message.id.to_string())
        .collect();
    let unique: std::collections::BTreeSet<_> = ids.iter().collect();
    assert_eq!(ids.len(), 5);
    assert_eq!(unique.len(), ids.len(), "{ids:?}");
    assert_eq!(ids[1], "ai-studio:1:p#id:1:1");
    assert_eq!(ids[2], "ai-studio:1:p#position:1:1");
    let other = parse(
        "p#id:1",
        &prompt(vec![
            json!({"role": "user", "id": "1", "text": "another file"}),
        ]),
    );
    for row in other.output.messages {
        assert!(!ids.contains(&row.message.id.to_string()));
    }
}

#[test]
fn metadata_waits_for_a_usable_chunk_but_fork_links_do_not() {
    let chunks = vec![
        json!({"role":"future", "branchParent":{"promptId":"parent"}}),
        json!({"role":"user"}),
    ];
    let step = parse("child", &prompt(chunks.clone()));
    assert!(step.output.sessions.is_empty());
    assert!(step.output.messages.is_empty());
    assert!(step.output.events.is_empty());
    assert_eq!(step.output.links.len(), 1);
    assert_eq!(step.counts.used.values().sum::<u64>(), 0);
    let mut chunks = chunks;
    chunks.push(json!({"role":"model", "text":"usable", "branchParent":{"promptId":"another"}}));
    let step = parse("child", &prompt(chunks));
    assert_eq!(step.output.links.len(), 2);
    assert_eq!(step.output.messages.len(), 2);
    assert_eq!(step.output.messages[0].message.role, Role::System);
    assert_eq!(
        step.output.messages[1].message.parent,
        Some(step.output.messages[0].message.id.clone())
    );
    assert_eq!(
        step.output.messages[1].message.model.as_deref(),
        Some("gemini-3-pro-preview")
    );
}
