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

/// A prompt the Drive collector fetched, named by its Drive file ID.
fn parse(drive_id: &str, bytes: &[u8]) -> crate::parse::Step {
    let parser = AiStudio {
        drive_id: Some(drive_id.into()),
        native: String::new(),
        title: Some("Electrons".into()),
        state: PromptState::default(),
    };
    crate::export::fixture::parse(&mut Parse::Export(Box::new(parser)), bytes).unwrap()
}

/// A prompt read from a download or Takeout, which names the file by title.
fn parse_at(path: &str, bytes: &[u8]) -> crate::parse::Step {
    crate::export::fixture::parse(&mut open(Path::new(path)).unwrap(), bytes).unwrap()
}

use crate::content_ids::tests::expected;

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
    assert_eq!(msgs[3].parent.as_ref(), Some(&msgs[2].id));
    assert!(
        matches!(&msgs[4].blocks[1], Block::File(a) if a.source_id.as_deref() == Some("abc123"))
    );
    assert!(matches!(&msgs[5].blocks[0], Block::Other { type_name, .. } if type_name == "error"));
    // A chunk without an ID is named by its content.
    assert_eq!(
        msgs[1].id.native().as_str(),
        expected(
            "Electrons",
            "user",
            r#"["user",{"createTime":"2026-05-17T12:27:24.102Z","role":"user","text":"what is an electron","tokenCount":6}]"#,
            0
        )
    );
    assert_eq!(step.output.messages[1].id_kind, IdKind::Content);
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
    // A call is named after the message that makes it.
    let call = format!("{}:code1", m.id.native().as_str());
    assert!(matches!(&m.blocks[1], Block::ToolCall { call_id: Some(c), .. } if *c == call));
    assert!(matches!(
        &m.blocks[2],
        Block::ToolResult { call_id: Some(c), outcome: ToolOutcome::Error { .. }, .. } if *c == call
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
        name_of(Path::new(
            "/t.zip!/Takeout/Drive/Google AI Studio/Electrons"
        ))
        .as_deref(),
        Some("Electrons")
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
    let wanted = [
        "1:p#id:8:question".to_owned(),
        "1:p#id:7:thought".to_owned(),
        expected(
            "p",
            "assistant",
            r#"["assistant",{"role":"model","text":"answer","tokenCount":5}]"#,
            0,
        ),
        expected(
            "p",
            "user",
            r#"["user",{"inlineImage":{"data":"aGk=","mimeType":"image/png"},"role":"user","text":"next"}]"#,
            0,
        ),
    ];
    for (i, row) in rows[1..].iter().enumerate() {
        assert_eq!(row.message.id.native().as_str(), wanted[i]);
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
fn native_ids_content_ids_and_the_system_header_cannot_name_the_same_message() {
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
    assert_eq!(
        ids[2],
        format!(
            "ai-studio:{}",
            expected(
                "p",
                "assistant",
                r#"["assistant",{"role":"model","text":"position one"}]"#,
                0
            )
        )
    );
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
    assert!(step.output.links.is_empty());
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

#[test]
fn repeated_prompt_parents_make_one_fork_and_conflicting_parents_make_none() {
    for (parents, count) in [(["parent", "parent"], 1), (["first", "second"], 0)] {
        let chunks = parents.into_iter().map(|parent| {
            json!({"role":"user", "text":"continue", "branchParent":{"promptId":parent}})
        }).collect();
        let step = parse("child", &prompt(chunks));
        assert_eq!(step.output.links.len(), count);
    }
}

#[test]
fn native_chunk_clock_is_intrinsic_and_drive_file_clock_is_file_mtime() {
    use polylogue_model::{Precision, TimeQuality};
    let step = parse(
        "clock",
        &prompt(vec![
            json!({"text":"q","role":"user","createTime":"2026-01-02T03:04:05Z"}),
            json!({"text":"a","role":"model"}),
        ]),
    );
    let messages: Vec<_> = step
        .output
        .messages
        .iter()
        .filter(|r| r.message.role != Role::System)
        .collect();
    assert_eq!(messages[0].message.at_precision, Some(Precision::Second));
    assert_eq!(messages[0].message.at_quality, TimeQuality::Intrinsic);
    assert_eq!(messages[1].message.at, None);
    assert_eq!(messages[1].message.at_quality, TimeQuality::Unknown);
    let bytes = br#"{"files":[{"id":"file","name":"Clock","createdTime":"2026-01-02T03:04:05Z"}]}"#;
    let listed =
        crate::export::fixture::parse(&mut Parse::Export(Box::new(Listing::default())), bytes)
            .unwrap();
    assert_eq!(
        listed.output.sessions[0].started_quality,
        TimeQuality::FileMtime
    );
    assert_eq!(
        listed.output.sessions[0].started_precision,
        Some(Precision::Second)
    );
    assert!(listed.output.messages.is_empty());
}

#[test]
fn ai_studio_deleted_chunk_keeps_the_other_ids() {
    let chunks = vec![
        json!({"role": "user", "text": "A"}),
        json!({"role": "model", "text": "B"}),
        json!({"role": "user", "text": "C"}),
        json!({"role": "model", "text": "D"}),
    ];
    let ids = |chunks: Vec<Json>| -> Vec<String> {
        parse("p", &prompt(chunks))
            .output
            .messages
            .iter()
            .skip(1)
            .map(|row| row.message.id.to_string())
            .collect()
    };
    let before = ids(chunks.clone());
    let mut without_b = chunks;
    without_b.remove(1);
    assert_eq!(
        ids(without_b),
        [before[0].clone(), before[2].clone(), before[3].clone()]
    );
}

#[test]
fn unchanged_chunk_reference_survives_rewrite() {
    // A Drive revision that edits one chunk and appends another: the
    // unchanged chunks keep their IDs, so a note on them keeps its target.
    let before = parse(
        "p",
        &prompt(vec![
            json!({"role": "user", "text": "keep me"}),
            json!({"role": "model", "text": "first draft"}),
        ]),
    );
    let after = parse(
        "p",
        &prompt(vec![
            json!({"role": "user", "text": "keep me"}),
            json!({"role": "model", "text": "second draft"}),
            json!({"role": "user", "text": "more"}),
        ]),
    );
    let id = |step: &crate::parse::Step, i: usize| step.output.messages[i].message.id.clone();
    assert_eq!(id(&before, 1), id(&after, 1));
    assert_ne!(id(&before, 2), id(&after, 2));
}

#[test]
fn two_untitled_prompts_give_two_sessions() {
    let first = parse_at(
        "/h/Takeout/Drive/Google AI Studio/Untitled prompt",
        &prompt(vec![
            json!({"role": "user", "text": "about rivers", "createTime": "2026-01-01T00:00:00Z"}),
        ]),
    );
    let second = parse_at(
        "/h/other/Takeout/Drive/Google AI Studio/Untitled prompt",
        &prompt(vec![
            json!({"role": "user", "text": "about mountains", "createTime": "2026-02-01T00:00:00Z"}),
        ]),
    );
    let session = |step: &crate::parse::Step| step.output.sessions[0].id.clone();
    assert_ne!(session(&first), session(&second));
    assert!(session(&first).native().as_str().starts_with("content-"));
    // The title is still the file name; it is not the identity.
    assert_eq!(
        first.output.sessions[0].title.as_ref().unwrap().text,
        "Untitled prompt"
    );
    let renamed = parse_at(
        "/h/Takeout/Drive/Google AI Studio/Rivers",
        &prompt(vec![
            json!({"role": "user", "text": "about rivers", "createTime": "2026-01-01T00:00:00Z"}),
        ]),
    );
    assert_eq!(session(&renamed), session(&first));
}

#[test]
fn the_drive_collector_names_a_prompt_by_its_file_id() {
    let step = parse_at(
        "/h/.local/share/polylogue/drive/ai-studio/1AbC.json",
        &prompt(vec![json!({"role": "user", "text": "hello"})]),
    );
    assert_eq!(step.output.sessions[0].id.to_string(), "ai-studio:1AbC");
    assert!(step.output.sessions[0].title.is_none());
}
