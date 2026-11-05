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
    wait_for_source(&fixture, records.len());
    check(&fixture, directory.path());
    assert_eq!(std::fs::read_to_string(source).unwrap(), original);
}
// Serving precedes the initial ingest and count pass. The startup poll lifts
// its deadline: status then answers once the daemon can, so the only answer
// accepted is a successful one.
fn wait_for_source(fixture: &crate::Fixture, messages: usize) {
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let output = cli(fixture, &["--deadline", "none", "--json", "status"]);
        let answer: command_surface::Answer<Value> = serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|error| panic!("invalid startup status answer: {error}: {output:?}"));
        assert_eq!(
            output.status.code(),
            Some(i32::from(answer.exit_code())),
            "startup status exit disagrees with its answer: {output:?}"
        );
        // A standalone archive's status is current, so never degraded.
        let value = match answer {
            command_surface::Answer {
                outcome: command_surface::Outcome::Ok(value),
                ..
            } => value,
            answer => panic!("startup status failed: {answer:?}; {output:?}"),
        };
        if value["busy"] == false && value["store"]["messages"] == messages {
            return;
        }
        assert!(
            Instant::now() < until,
            "source never became ready; last status: {value}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
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
        "--page-tokens",
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
                .map(|item| item.source.reference.as_str())
                .collect();
            assert_eq!(ids, ["message:claude-code:u1", "message:claude-code:a1"]);
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
            let shown = page(fixture, &["--show", "tools:full", "--output-lines", "0"]);
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
            assert_eq!(calls.len(), 1, "one call must own its paired result");
            assert_eq!(
                call.result_source.as_ref().unwrap().reference,
                "block:claude-code:u1#0"
            );
            let expanded = page(
                fixture,
                &["--show", "tools:full,harness:full", "--output-lines", "0"],
            );
            assert_eq!(
                expanded
                    .document
                    .items
                    .iter()
                    .filter(|item| matches!(item.kind, ItemKind::Call { .. }))
                    .count(),
                1,
                "explicit harness expansion must not emit the matched result twice"
            );
            let direct = cli(
                fixture,
                &[
                    "--deadline",
                    "none",
                    "--json",
                    "read",
                    "block:claude-code:a1#0",
                    "--show",
                    "tools:full",
                    "--output-lines",
                    "0",
                    "--page-tokens",
                    "100000",
                ],
            );
            assert!(direct.status.success(), "{direct:?}");
            let direct: ReadPage = serde_json::from_value(json(&direct)).unwrap();
            let direct_calls: Vec<_> = direct
                .document
                .items
                .iter()
                .filter_map(|item| match &item.kind {
                    ItemKind::Call { call } => Some(call),
                    _ => None,
                })
                .collect();
            assert_eq!(direct_calls.len(), 1);
            assert_eq!(direct_calls[0].input.as_ref(), Some(&input));
            assert_eq!(direct_calls[0].output.as_deref(), Some(output));
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
            let full = page(fixture, &["--show", "tools:full", "--output-lines", "0"]);
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

#[test]
fn indexed_tool_result_pages_reconstruct_the_exact_pair_once() {
    let output = "λ🙂 exact remote result\n".repeat(1500);
    let input = json!({"command":"synthetic command","enabled":false});
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
            let mut args: Vec<String> = [
                "--deadline",
                "none",
                "--json",
                "read",
                "block:claude-code:a1#0",
                "--show",
                "tools:full",
                "--output-lines",
                "0",
                "--page-tokens",
                "1000",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            let mut items = super::read_current::Items::default();
            let mut cursors = std::collections::BTreeSet::new();
            let mut pages = 0usize;
            loop {
                let raw = cli(
                    fixture,
                    &args.iter().map(String::as_str).collect::<Vec<_>>(),
                );
                assert!(raw.status.success(), "{raw:?}");
                let shown: ReadPage = serde_json::from_value(json(&raw)).unwrap();
                pages += 1;
                assert!(
                    pages <= output.len() + input.to_string().len(),
                    "finite source must advance"
                );
                items.push(&shown);
                let Some(next) = &shown.document.footer.next else {
                    break;
                };
                assert!(cursors.insert(serde_json::to_string(next).unwrap()));
                let stages = polylogue_model::command::effective_stages(&next.cli);
                assert_eq!(stages.len(), 1);
                args = vec!["--json".into()];
                args.extend(stages[0].args.clone());
            }
            let values = items.into_values();
            let calls: Vec<_> = values
                .iter()
                .filter_map(|item| match &item.kind {
                    ItemKind::Call { call } => Some(call),
                    _ => None,
                })
                .collect();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].input.as_ref(), Some(&input));
            assert_eq!(calls[0].output.as_deref(), Some(output.as_str()));
            assert!(
                !cursors.is_empty(),
                "large paired output must exercise actual fragments"
            );
        },
    );
}

#[test]
fn collapsed_tool_pairs_keep_the_selected_native_input_target() {
    with_source(
        &[
            native(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"synthetic_unknown_tool","input":{"opaque_native_argument":"native-value"}}]),
            ),
            native(
                "u1",
                "user",
                json!([{"type":"tool_result","tool_use_id":"c1","content":"native result"}]),
            ),
        ],
        |fixture, _| {
            let shown = page(fixture, &[]);
            let calls: Vec<_> = shown
                .document
                .items
                .iter()
                .filter_map(|item| match &item.kind {
                    ItemKind::Call { call } => Some(call),
                    _ => None,
                })
                .collect();
            assert_eq!(calls.len(), 1);
            assert_eq!(
                calls[0].target.value().map(String::as_str),
                Some("native-value")
            );
            assert_eq!(calls[0].outcome.value().map(String::as_str), Some("ok"));
            assert!(calls[0].input.is_none());
            assert!(calls[0].output.is_none());
            assert_eq!(
                calls[0].result_source.as_ref().unwrap().reference,
                "block:claude-code:u1#0"
            );
        },
    );
}
