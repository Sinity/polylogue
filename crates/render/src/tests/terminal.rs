use super::*;

const HOSTILE: &str = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";

fn safe(text: &str) {
    assert!(
        !text
            .chars()
            .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}'),
        "{text:?}"
    );
    assert!(text.contains("\\x1b"), "{text:?}");
}

#[test]
fn every_terminal_item_escapes_archived_fields() {
    let mut session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    session.cwd = vec![HOSTILE.into()];
    let mut messages = conversation();
    for message in &mut messages {
        for block in &mut message.blocks {
            match block {
                Block::Text { text } => *text = HOSTILE.into(),
                Block::Thinking { text, .. } => *text = Some(HOSTILE.into()),
                Block::ToolCall { tool, command, .. } => {
                    *tool = Some(HOSTILE.into());
                    *command = Some(HOSTILE.into());
                }
                Block::ToolResult { output, .. } => {
                    *output = Some(format!("{}\n{HOSTILE}", "line\n".repeat(25)))
                }
                _ => {}
            }
        }
    }
    let result = crate::session(
        &session,
        Some(HOSTILE),
        messages,
        View::Full,
        Format::Terminal,
        &[HOSTILE.into()],
    )
    .unwrap();
    let text = result["content"].as_str().unwrap();
    safe(text);
    assert!(text.matches("  line\n").count() >= 25);
    let mut doc = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("claude-ai:doc:a").unwrap(),
        polylogue_model::DocumentPurpose::Memory,
    );
    doc.title = Some(HOSTILE.into());
    doc.blocks = vec![Block::Text {
        text: HOSTILE.into(),
    }];
    safe(
        documents::document(doc, View::Full, Format::Terminal).unwrap()["content"]
            .as_str()
            .unwrap(),
    );
}

#[test]
fn query_columns_snippets_nested_rows_and_footer_are_escaped() {
    let data = json!({"columns":[{"name":HOSTILE}], "rows":[{HOSTILE: HOSTILE, "snippet":{"text":HOSTILE,"marks":[[0,1]]}, "notes":[{"body":HOSTILE}]}], "count":1,"next":HOSTILE,"notes":[HOSTILE],"why":[{"clause":HOSTILE}]});
    let mut bytes = Vec::new();
    results::render(&data, Format::Terminal, &mut bytes).unwrap();
    safe(&String::from_utf8(bytes).unwrap());
}

#[test]
fn other_formats_preserve_data_and_escape_their_own_syntax() {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let content = format!("{HOSTILE}<script> & ```");
    for format in [Format::Json, Format::Html, Format::Markdown] {
        let message = message(
            "1",
            Role::User,
            None,
            vec![Block::Text {
                text: content.clone(),
            }],
        );
        let value = crate::session(&session, None, vec![message], View::Full, format, &[]).unwrap();
        if format == Format::Json {
            assert_eq!(value["messages"][0]["blocks"][0]["text"], content);
        } else {
            let text = value["content"].as_str().unwrap();
            assert!(text.contains(HOSTILE));
            if format == Format::Html {
                assert!(text.contains("&lt;script&gt; &amp;"));
            }
        }
    }
}

#[test]
fn event_and_schema_text_escape_controls() {
    let event = Event {
        session: None,
        kind: HOSTILE.into(),
        at: None,
        after: None,
        range: None,
        data: json!({"text": HOSTILE}),
    };
    safe(
        events::event(event, Format::Terminal).unwrap()["content"]
            .as_str()
            .unwrap(),
    );
    let mut bytes = Vec::new();
    schema::render(&json!({"before":{"origin":HOSTILE,"selection":[{"input":HOSTILE}]},"reasons":[HOSTILE],"changes":[{"kind":HOSTILE,"path":HOSTILE}]}), Format::Terminal, &mut bytes).unwrap();
    safe(&String::from_utf8(bytes).unwrap());
    safe(&measures::usage(
        &json!({"by_model":[{"model":HOSTILE}],"price_catalog":HOSTILE}),
    ));
    safe(&measures::quota(
        &json!({"tools":[{"tool":HOSTILE,"plan":HOSTILE}]}),
    ));
    let data = json!([{"repo":HOSTILE,"command":HOSTILE,"format":HOSTILE,"what":HOSTILE}]);
    safe(&measures::failing(&data));
    safe(&measures::drift(&data));
}
