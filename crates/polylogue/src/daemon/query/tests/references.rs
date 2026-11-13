//! References as conditions: `block:` and `assertion:` name one thing.

use super::*;

#[test]
fn provider_conversation_urls_select_native_sessions_through_declared_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = polylogue_store::fixture::Fixture::new(dir.path()).unwrap();
    for (index, id) in [
        "chatgpt:a%2Fb",
        "chatgpt:a%2fb",
        "chatgpt:a%61b",
        "chatgpt:a%25b",
        "chatgpt:a@b",
        "chatgpt:a/b",
        "chatgpt:a%b",
        "chatgpt:aab",
        "claude-ai:claude-id",
        "gemini-app:gemini-id",
        "ai-studio:studio-id",
        "grok:grok-id",
    ]
    .into_iter()
    .enumerate()
    {
        let session = fx
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        let origin = id.split_once(':').unwrap().0;
        fx.message(
            session,
            MessageFacts {
                id: &format!("{origin}:provider-message-{index}"),
                role: "user",
                written_by: "human",
                prompt_kind: Some("prose"),
                text: Some("original provider session"),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let archive = TestArchive { _dir: dir, fx };
    for (url, expected) in [
        ("https://chatgpt.com/c/a%2Fb", "chatgpt:a%2Fb"),
        ("https://chatgpt.com/c/a%2fb", "chatgpt:a%2fb"),
        ("https://chatgpt.com/c/a%61b", "chatgpt:a%61b"),
        ("https://chatgpt.com/c/a%25b", "chatgpt:a%25b"),
        ("https://chatgpt.com/c/a@b", "chatgpt:a@b"),
        (
            "https://www.chat.openai.com/c/a%2Fb/?model=x#turn",
            "chatgpt:a%2Fb",
        ),
        ("https://chatgpt.com/g/project/c/a%2Fb", "chatgpt:a%2Fb"),
        ("http://chatgpt.com/c/a%2Fb", "chatgpt:a%2Fb"),
        ("https://claude.ai/chat/claude-id", "claude-ai:claude-id"),
        (
            "https://gemini.google.com/u/0/app/gemini-id",
            "gemini-app:gemini-id",
        ),
        (
            "https://aistudio.google.com/u/0/prompts/studio-id",
            "ai-studio:studio-id",
        ),
        ("https://grok.com/c/grok-id", "grok:grok-id"),
    ] {
        assert_eq!(ids(&q(&archive, url)), [expected], "{url}");
    }
    for url in [
        "https://chatgpt.com/other/c/a%2Fb",
        "https://chatgpt.com/c/a%2Fb/extra",
        "https://chatgpt.com/share/a%2Fb",
        "https://chatgpt.com.evil.invalid/c/a%2Fb",
        "https://claude.ai/other/chat/claude-id",
    ] {
        let failure = run_q(&archive, ask(url)).unwrap_err();
        assert_eq!(failure.code(), "invalid_query", "{url}: {failure:?}");
        assert!(
            failure.message.contains("not a conversation URL"),
            "{url}: {failure:?}"
        );
    }
}

#[test]
fn a_block_reference_names_its_block_message_and_session() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = polylogue_store::fixture::Fixture::new(dir.path()).unwrap();
    let session = fx
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fx.message(
        session,
        MessageFacts {
            id: "test:message",
            role: "user",
            written_by: "human",
            prompt_kind: Some("prose"),
            text: Some("synthetic"),
            ..Default::default()
        },
    )
    .unwrap();
    let w = TestArchive { _dir: dir, fx };
    let blocks = q(&w, "blocks where block:test:message#0");
    assert_eq!(blocks["count"], 1, "{blocks}");
    assert_eq!(
        q(&w, "messages where block:test:message#0")["rows"][0]["id"],
        "test:message"
    );
    assert_eq!(ids(&q(&w, "block:test:message#0")), ["test:session"]);
    assert_eq!(q(&w, "messages where block:test:message#999")["count"], 0);
    let bad = run_q(&w, ask("block:test:message:0")).unwrap_err();
    assert_eq!(bad.code(), "invalid_query");
}

#[test]
fn an_assertion_reference_names_the_current_revision_or_one_revision() {
    let w = morrow();
    let first = q(&w, "assertions where kind:decision");
    let id = first["rows"][0]["id"].as_str().unwrap().to_owned();
    let root = first["rows"][0]["root"].as_str().unwrap().to_owned();
    let by_root = q(&w, &format!("assertions where assertion:{root}"));
    assert_eq!(by_root["rows"][0]["id"], id, "{by_root}");
    let by_revision = q(&w, &format!("assertions where assertion:{root}@{id}"));
    assert_eq!(by_revision["count"], 1);
    let elsewhere = run_q(&w, ask(&format!("assertion:{root}"))).unwrap_err();
    assert_eq!(elsewhere.code(), "not_built");
    assert_eq!(
        elsewhere.message,
        "this reference at the selected row grain is not built yet"
    );
    // Pages and commits are not built: refused by name, never empty.
    for later in ["page:plan", "file:src/a.rs@HEAD"] {
        assert_eq!(
            run_q(&w, ask(later)).unwrap_err().code(),
            "not_built",
            "{later}"
        );
    }
}
