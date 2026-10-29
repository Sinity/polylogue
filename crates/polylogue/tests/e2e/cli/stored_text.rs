//! Stored text stays exact; presentation escapes active controls and markup.
use super::*;
use polylogue_model::read::{ItemKind, ReadPage};
use serde_json::{Value, json};

pub(super) fn with_source(records: &[Value], check: impl FnOnce(&crate::Fixture, &Path)) {
    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("home");
    let project = home.join(".claude/projects/-stored-text");
    std::fs::create_dir_all(&project).unwrap();
    let source = project.join("stored.jsonl");
    let original: String = records.iter().map(|value| format!("{value}\n")).collect();
    std::fs::write(&source, &original).unwrap();
    let fixture = crate::Fixture::new(
        &directory.path().join("archive"),
        &home,
        &directory.path().join("config"),
    );
    let _daemon = start_daemon(&fixture);
    wait_status(&fixture, |status| {
        status["store"]["messages"] == records.len()
    });
    check(&fixture, directory.path());
    assert_eq!(std::fs::read_to_string(source).unwrap(), original);
}
pub(super) fn native(id: &str, role: &str, content: Value) -> Value {
    json!({"type":role,"uuid":id,"sessionId":"stored","timestamp":"2026-03-01T10:00:00Z","message":{"role":role,"content":content}})
}
pub(super) fn page(fixture: &crate::Fixture, options: &[&str]) -> ReadPage {
    let mut arguments = vec![
        "--deadline",
        "none",
        "--json",
        "read",
        "session:claude-code:stored",
        "--budget",
        "100000",
    ];
    arguments.extend_from_slice(options);
    let output = cli(fixture, &arguments);
    assert!(output.status.success(), "{output:?}");
    let page: ReadPage = serde_json::from_value(json(&output)).unwrap();
    assert!(
        page.document.footer.next.is_none(),
        "small fixture must be complete"
    );
    page
}
pub(super) fn texts(page: &ReadPage) -> Vec<&str> {
    page.document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}
pub(super) fn bundle(
    fixture: &crate::Fixture,
    directory: &Path,
) -> Vec<polylogue_model::bundle::Record> {
    std::fs::create_dir(directory).unwrap();
    let output = cli(
        fixture,
        &[
            "--deadline",
            "none",
            "--json",
            "read",
            "session:claude-code:stored",
            "--format",
            "jsonl",
            "--to",
            directory.to_str().unwrap(),
        ],
    );
    assert!(matches!(output.status.code(), Some(0 | 5)), "{output:?}");
    std::fs::read_to_string(directory.join("records.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn credential_shaped_fragments_remain_exact_and_in_message_order() {
    let expected = [
        "prefix -----BEGIN PRIVATE KEY-----private-one",
        "private-two-----END PRIVATE KEY----- visible λ🙂",
    ];
    with_source(
        &[
            native("u1", "user", json!(expected[0])),
            native("a1", "assistant", json!(expected[1])),
        ],
        |fixture, _| {
            let shown = page(fixture, &[]);
            assert_eq!(texts(&shown), expected);
            let ids: Vec<_> = shown
                .document
                .items
                .iter()
                .filter(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
                .map(|item| item.source.message.as_ref().unwrap().to_string())
                .collect();
            assert_eq!(ids, ["claude-code:u1", "claude-code:a1"]);
        },
    );
}

#[test]
fn tool_aliases_preserve_exact_input_values_and_result_bytes() {
    let input = json!({"command":"private-command-tail-----END PRIVATE KEY----- visible command","notes":"independent notes","number":0,"enabled":false,"empty":"","nil":null});
    let output = "-----BEGIN PRIVATE KEY-----private-result λ🙂";
    with_source(
        &[
            native(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"Bash","input":input}]),
            ),
            native(
                "u1",
                "user",
                json!([{"type":"tool_result","tool_use_id":"c1","content":output}]),
            ),
        ],
        |fixture, _| {
            let shown = page(fixture, &["--tools", "full", "--output-lines", "0"]);
            let calls: Vec<_> = shown
                .document
                .items
                .iter()
                .filter_map(|item| match &item.kind {
                    ItemKind::Call { call } => Some(call),
                    _ => None,
                })
                .collect();
            let call = calls
                .iter()
                .find(|call| call.input.is_some())
                .expect("prepared tool input");
            assert_eq!(call.input.as_ref(), Some(&input));
            assert!(
                calls
                    .iter()
                    .any(|call| call.output.as_deref() == Some(output)),
                "{calls:?}"
            );
        },
    );
}

#[test]
fn dialogue_filters_tool_output_without_rewriting_the_remaining_source() {
    let visible = "private-hidden-tail-----END PRIVATE KEY----- visible";
    with_source(
        &[native(
            "u1",
            "user",
            json!([{"type":"tool_result","tool_use_id":"missing","content":"-----BEGIN PRIVATE KEY-----private-hidden"},{"type":"text","text":visible}]),
        )],
        |fixture, _| {
            assert_eq!(texts(&page(fixture, &["--view", "dialogue"])), [visible]);
            let full = page(fixture, &["--tools", "full", "--output-lines", "0"]);
            assert!(
                serde_json::to_string(&full.document.items)
                    .unwrap()
                    .contains("private-hidden")
            );
        },
    );
}

#[test]
fn title_derivation_preserves_credential_shaped_source_text() {
    let title = "--token=synthetic-title-value visible title";
    with_source(&[native("u1", "user", json!(title))], |fixture, _| {
        let shown = page(fixture, &[]);
        assert_eq!(
            shown.document.header.title.value().map(String::as_str),
            Some(title)
        );
        assert_eq!(texts(&shown), [title]);
    });
}
