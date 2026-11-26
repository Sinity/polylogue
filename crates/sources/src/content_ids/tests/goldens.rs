use polylogue_model::{IdKind, Json};
use serde_json::json;
use std::path::Path;

#[test]
fn content_named_origins_keep_fixed_ids_and_number_equal_messages() {
    let text = "Read the module";
    let repeated = |value: Json| vec![value; 3];
    let lines = |values: Vec<Json>| {
        values
            .into_iter()
            .map(|value| format!("{value}\n"))
            .collect::<String>()
    };
    let claude = lines(repeated(json!({
        "type":"user", "sessionId":"s", "message":{"role":"user", "content":text}
    })));
    let codex = format!(
        "{}{}",
        lines(vec![json!({"id":"s", "timestamp":"2026-01-01T00:00:00Z"})]),
        lines(repeated(
            json!({"type":"message", "role":"user", "content":[{"type":"input_text", "text":text}]})
        ))
    );
    let grok_cli = lines(repeated(json!({"type":"user", "content":text})));
    let grok = json!({"conversations":[{
        "conversation":{"id":"s"},
        "responses":repeated(json!({"sender":"human", "message":text}))
    }]})
    .to_string();
    let ai_studio = json!({"chunkedPrompt":{"chunks":
        repeated(json!({"role":"user", "text":text}))
    }})
    .to_string();
    let claude_ai = json!([{"uuid":"s", "chat_messages":(0..3).map(|i| json!({
        "uuid":format!("m{i}"), "sender":"human", "text":"A prompt", "content":[],
        "compaction_summary":[{"type":"text", "text":text}]
    })).collect::<Vec<_>>()}])
    .to_string();
    let browser = json!({
        "polylogue_capture_kind":"conversation", "schema_version":1,
        "raw_provider_payload":{},
        "session":{"provider":"gemini", "provider_session_id":"s",
            "turns":(0..3).map(|i| json!({"role":"user", "text":text, "ordinal":i, "blocks":[]})).collect::<Vec<_>>()}
    }).to_string();
    let cell = format!(
        "<div class=\"outer-cell mdl-cell\"><div class=\"mdl-typography--body-1\">Prompted {text}<br>Jul 27, 2026, 10:59:30 PM CEST<br></div><div class=\"mdl-typography--caption\">Details:<br>https://gemini.google.com/app/s</div></div>"
    );
    let gemini = format!("<html><body>{}</body></html>", cell.repeat(3));
    let aider = format!(
        "TO LLM 2026-01-01T10:00:00\n{}\n",
        vec![format!("-------\nUSER {text}"); 3].join("\n")
    );

    // Hashes of hand-written canonical JSON, with the domain bytes
    // `polylogue-message-content\0` before it:
    // ["user",[{"text":"Read the module","type":"text"}]]
    // ["user",{"role":"user","text":"Read the module"}]
    // ["user","Read the module"]
    for (origin, path, bytes, hash) in [
        (
            "claude-code",
            "projects/p/s.jsonl",
            &claude,
            "35085b468aa12a8a",
        ),
        ("codex", "rollout-s.jsonl", &codex, "35085b468aa12a8a"),
        (
            "grok-cli",
            ".grok/sessions/p/s/chat_history.jsonl",
            &grok_cli,
            "35085b468aa12a8a",
        ),
        ("grok", "conversations.json", &grok, "35085b468aa12a8a"),
        (
            "claude-ai",
            "conversations.json",
            &claude_ai,
            "35085b468aa12a8a",
        ),
        ("aider", ".aider.llm.history", &aider, "35085b468aa12a8a"),
        (
            "ai-studio",
            "polylogue/drive/ai-studio/s.json",
            &ai_studio,
            "f5a06152eb8400e2",
        ),
        ("gemini-app", "MyActivity.html", &gemini, "fb6ba35e81bde94f"),
        (
            "browser-capture",
            "capture.json",
            &browser,
            "fb6ba35e81bde94f",
        ),
    ] {
        let mut parser = (crate::origin::by_name(origin).unwrap().open)(Path::new(path))
            .unwrap_or_else(|| panic!("{origin}: no parser for {path}"));
        let step = crate::export::fixture::parse(&mut parser, bytes.as_bytes()).unwrap();
        let rows: Vec<_> = step
            .output
            .messages
            .iter()
            .filter(|row| row.id_kind == IdKind::Content)
            .collect();
        assert_eq!(rows.len(), 3, "{origin}");
        let session = rows[0].message.session.native().as_str();
        let ids: Vec<_> = rows
            .iter()
            .map(|row| row.message.id.native().as_str())
            .collect();
        assert_eq!(
            ids,
            [
                format!("{session}~user-{hash}-0"),
                format!("{session}~user-{hash}-1"),
                format!("{session}~user-{hash}-2")
            ],
            "{origin}"
        );
    }
}
