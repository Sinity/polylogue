use super::*;
use crate::parse::run_document;
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
    crate::array::fixture::parse(&mut Parse::Array(Box::new(parser)), bytes).unwrap()
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
        "Electrons#1"
    );
    assert!(
        matches!(&msgs[4].blocks[1], Block::File(a) if a.source_id.as_deref() == Some("abc123"))
    );
    assert!(matches!(&msgs[5].blocks[0], Block::Other { type_name, .. } if type_name == "error"));
    // A chunk without an ID keeps its identity by its place.
    assert_eq!(msgs[1].id.native().as_str(), "Electrons#0");
    assert_eq!(
        step.output.sessions[0].title.as_ref().unwrap().text,
        "Electrons"
    );
    assert!(
        step.accounting.unread_fields.is_empty(),
        "{:?}",
        step.accounting.unread_fields
    );
    assert_eq!(step.accounting.framed, 6);
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
    assert!(matches!(&m.blocks[1], Block::ToolCall { call_id: Some(c), .. } if c == "p#0:code1"));
    assert!(matches!(
        &m.blocks[2],
        Block::ToolResult { call_id: Some(c), outcome: ToolOutcome::Error { .. }, .. } if c == "p#0:code1"
    ));
    assert!(matches!(&m.blocks[3], Block::Image(a) if a.blob.is_some()));
    assert!(
        step.accounting.unread_fields.is_empty(),
        "{:?}",
        step.accounting.unread_fields
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
    let step = run_document(&Listing, &bytes).unwrap();
    let s = &step.output.sessions[0];
    assert_eq!(s.id.native().as_str(), "1AbC");
    assert_eq!(s.title.as_ref().unwrap().text, "Electrons");
    assert!(step.output.messages.is_empty());
    assert!(recognize(
        Path::new("/s/polylogue/drive/ai-studio/files.json"),
        br#"{"files": ["#
    ));
}
