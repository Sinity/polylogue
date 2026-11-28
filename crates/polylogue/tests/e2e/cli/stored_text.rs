//! Stored text stays exact; presentation escapes active controls and markup.
use super::*;
use polylogue_model::read::{IndicatorKind, ItemKind, ReadPage};
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

pub(super) fn source_record(id: &str, role: &str, content: Value) -> Value {
    json!({"type":role,"uuid":id,"sessionId":"stored","timestamp":"2026-03-01T10:00:00Z","message":{"role":role,"content":content}})
}
pub(super) fn page(fixture: &crate::Fixture, options: &[&str]) -> ReadPage {
    let mut arguments = vec![
        "--deadline",
        "none",
        "--json",
        "session:claude-code:stored",
        "read",
        "--budget",
        "100000",
    ];
    arguments.extend_from_slice(options);
    let output = cli(fixture, &arguments);
    assert!(output.status.success(), "{output:?}");
    let page: ReadPage = read_page(json(&output));
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
    wait_caught_up(fixture);
    let output = cli(
        fixture,
        &[
            "--deadline",
            "none",
            "--json",
            "session:claude-code:stored",
            "read",
            "--format",
            "jsonl",
            "--to",
            directory.to_str().unwrap(),
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    std::fs::read_to_string(directory.join("records.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn credential_shaped_fragments_stay_unchanged_and_in_order() {
    let expected = [
        "prefix -----BEGIN PRIVATE KEY-----private-one",
        "private-two-----END PRIVATE KEY----- visible λ🙂",
    ];
    with_source(
        &[
            source_record("u1", "user", json!(expected[0])),
            source_record("a1", "assistant", json!(expected[1])),
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
fn tool_aliases_preserve_input_values_and_result_bytes() {
    let input = json!({"command":"private-command-tail-----END PRIVATE KEY----- visible command","notes":"independent notes","number":0,"enabled":false,"empty":"","nil":null});
    let output = "-----BEGIN PRIVATE KEY-----private-result λ🙂";
    with_source(
        &[
            source_record(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"Bash","input":input}]),
            ),
            source_record(
                "u1",
                "user",
                json!([{"type":"tool_result","tool_use_id":"c1","content":output}]),
            ),
        ],
        |fixture, _| {
            let shown = page(fixture, &["--show", "tools:full"]);
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
            let expanded = page(fixture, &["--show", "tools:full,harness:full"]);
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
                    "block:claude-code:a1#0",
                    "read",
                    "--show",
                    "tools:full",
                    "--budget",
                    "100000",
                ],
            );
            assert!(direct.status.success(), "{direct:?}");
            let direct: ReadPage = read_page(json(&direct));
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
        &[source_record(
            "u1",
            "user",
            json!([{"type":"tool_result","tool_use_id":"missing","content":"-----BEGIN PRIVATE KEY-----private-hidden"},{"type":"text","text":visible}]),
        )],
        |fixture, _| {
            assert_eq!(texts(&page(fixture, &["--view", "dialogue"])), [visible]);
            let full = page(fixture, &["--show", "tools:full"]);
            let results = full
                .document
                .items
                .iter()
                .filter(|item| match &item.kind {
                    ItemKind::Call { call } => {
                        call.output.as_deref() == Some("-----BEGIN PRIVATE KEY-----private-hidden")
                    }
                    _ => false,
                })
                .count();
            assert_eq!(results, 1, "{:?}", full.document.items);
        },
    );
}

#[test]
fn title_derivation_preserves_credential_shaped_source_text() {
    let title = "--token=synthetic-title-value visible title";
    with_source(
        &[source_record("u1", "user", json!(title))],
        |fixture, _| {
            let shown = page(fixture, &[]);
            assert_eq!(
                shown.document.header.title.value().map(String::as_str),
                Some(title)
            );
            assert_eq!(texts(&shown), [title]);
        },
    );
}

#[test]
fn indexed_tool_result_pages_reconstruct_the_pair_once() {
    let output = "λ🙂 exact remote result\n".repeat(1500);
    let input = json!({"command":"synthetic command","enabled":false});
    with_source(
        &[
            source_record(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"Bash","input":input}]),
            ),
            source_record(
                "u1",
                "user",
                json!([{"type":"tool_result","tool_use_id":"c1","content":output}]),
            ),
        ],
        |fixture, _| {
            let args: Vec<String> = [
                "--deadline",
                "none",
                "--json",
                "block:claude-code:a1#0",
                "read",
                "--show",
                "tools:full",
                "--budget",
                "1000",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            let raw = cli(
                fixture,
                &args.iter().map(String::as_str).collect::<Vec<_>>(),
            );
            assert!(raw.status.success(), "{raw:?}");
            let mut shown = read_page(json(&raw));
            let mut items = super::read_current::Items::default();
            let mut cursors = std::collections::BTreeSet::new();
            let mut pages = 0usize;
            loop {
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
                shown = followed_page(fixture, next);
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
                "large paired output must produce fragments"
            );
        },
    );
}

#[test]
fn a_selected_result_owns_the_pair_when_the_call_is_outside_the_reading() {
    let input = json!({"command":"synthetic command","detail":"call input"});
    let output = "error: synthetic result λ🙂";
    with_source(
        &[
            source_record(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"Bash","input":input}]),
            ),
            source_record(
                "u1",
                "user",
                json!([{"type":"tool_result","tool_use_id":"c1","is_error":true,"content":output}]),
            ),
        ],
        |fixture, _| {
            for (show, full) in [
                ("tools:line,harness:full", false),
                ("tools:full,harness:full", true),
            ] {
                let whole = page(fixture, &["--show", show]);
                let range = page(fixture, &["--show", show, "--last", "1"]);
                let result = cli(
                    fixture,
                    &[
                        "--deadline",
                        "none",
                        "--json",
                        "message:claude-code:u1",
                        "read",
                        "--show",
                        show,
                        "--budget",
                        "100000",
                    ],
                );
                assert!(result.status.success(), "{result:?}");
                let single = read_page(json(&result));
                for (page, expected_source) in [
                    (whole, "block:claude-code:a1#0"),
                    (range, "block:claude-code:u1#0"),
                    (single, "block:claude-code:u1#0"),
                ] {
                    let calls: Vec<_> = page
                        .document
                        .items
                        .iter()
                        .filter_map(|item| match &item.kind {
                            ItemKind::Call { call } => Some((&item.source, call)),
                            _ => None,
                        })
                        .collect();
                    let [(source, call)] = calls.as_slice() else {
                        panic!("expected one pair in {expected_source}: {page:?}");
                    };
                    assert_eq!(source.reference, expected_source);
                    assert_eq!(call.call_id.as_deref(), Some("c1"));
                    assert_eq!(call.tool.value().map(String::as_str), Some("Bash"));
                    assert_eq!(call.outcome.value().map(String::as_str), Some("error"));
                    assert_eq!(call.error_line.as_deref(), Some(output));
                    assert_eq!(call.output.as_deref(), full.then_some(output));
                    assert_eq!(
                        call.input.as_ref(),
                        (full && expected_source.ends_with("a1#0")).then_some(&input)
                    );
                    assert_eq!(
                        call.result_source.as_ref().unwrap().reference,
                        "block:claude-code:u1#0"
                    );
                }
            }
        },
    );
}

#[test]
fn collapsed_tool_pairs_keep_selected_input_target() {
    with_source(
        &[
            source_record(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"synthetic_unknown_tool","input":{"opaque_tool_argument":"native-value"}}]),
            ),
            source_record(
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

#[test]
fn around_tool_call_reads_holding_message() {
    with_source(
        &[
            source_record("m0", "user", json!("outside the window")),
            source_record("m1", "user", json!("before the call")),
            source_record(
                "m2",
                "assistant",
                json!([
                    {"type":"text","text":"beside the call"},
                    {"type":"tool_use","id":"c1","name":"Bash","input":{"command":"synthetic"}}
                ]),
            ),
            source_record(
                "m3",
                "user",
                json!([{"type":"tool_result","tool_use_id":"c1","content":"synthetic result"}]),
            ),
            source_record("m4", "assistant", json!("outside after the result")),
        ],
        |fixture, _| {
            let target = "tool:claude-code:stored#c1";
            for (show, hide, following_is_call) in [
                ("tools:full,harness:full", "inherited", true),
                ("harness:full", "tools", false),
            ] {
                let output = cli(
                    fixture,
                    &[
                        "--deadline",
                        "none",
                        "--json",
                        target,
                        "read",
                        "--around",
                        "1",
                        "--show",
                        show,
                        "--hide",
                        hide,
                        "--budget",
                        "100000",
                    ],
                );
                assert!(output.status.success(), "{output:?}");
                let page = read_page(json(&output));
                let items = &page.document.items;
                let headers: Vec<_> = items
                    .iter()
                    .filter(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
                    .map(|item| item.source.reference.as_str())
                    .collect();
                assert_eq!(
                    headers,
                    [
                        "message:claude-code:m1",
                        "message:claude-code:m2",
                        "message:claude-code:m3"
                    ]
                );
                let marks: Vec<_> = items
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| {
                        matches!(
                            item.kind,
                            ItemKind::Indicator {
                                indicator: IndicatorKind::Selected,
                                ..
                            }
                        )
                    })
                    .collect();
                let [(at, mark)] = marks.as_slice() else {
                    panic!("expected one selection mark: {items:?}");
                };
                assert_eq!(mark.source.reference, target);
                assert_eq!(
                    mark.source.message.as_ref().unwrap().to_string(),
                    "claude-code:m2"
                );
                assert_eq!(mark.source.block, Some(1));
                if following_is_call {
                    let ItemKind::Call { call } = &items[*at + 1].kind else {
                        panic!("selection should mark the call: {items:?}");
                    };
                    assert_eq!(call.output.as_deref(), Some("synthetic result"));
                } else {
                    assert_eq!(items[*at - 1].source.reference, "message:claude-code:m2");
                }
            }
        },
    );
}

#[test]
fn span_read_marks_item() {
    with_source(
        &[
            source_record("m1", "user", json!("outside")),
            source_record(
                "m2",
                "assistant",
                json!([{"type":"text","text":"first"},{"type":"text","text":"aλz"}]),
            ),
        ],
        |fixture, _| {
            for target in [
                "block:claude-code:m2#1",
                "span:block:claude-code:m2#1#b=1-3",
                "span:message:claude-code:m2#c=8-9",
            ] {
                let output = cli(
                    fixture,
                    &[
                        "--deadline",
                        "none",
                        "--json",
                        target,
                        "read",
                        "--budget",
                        "100000",
                    ],
                );
                assert!(output.status.success(), "{output:?}");
                let page = read_page(json(&output));
                assert_eq!(texts(&page), ["first", "aλz"]);
                let items = &page.document.items;
                let marks: Vec<_> = items
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| {
                        matches!(
                            item.kind,
                            ItemKind::Indicator {
                                indicator: IndicatorKind::Selected,
                                ..
                            }
                        )
                    })
                    .collect();
                let [(at, mark)] = marks.as_slice() else {
                    panic!("expected one selection mark: {items:?}");
                };
                assert_eq!(mark.source.reference, target);
                assert_eq!(
                    mark.source.message.as_ref().unwrap().to_string(),
                    "claude-code:m2"
                );
                assert_eq!(mark.source.block, Some(1));
                assert!(matches!(&items[*at + 1].kind, ItemKind::Text { text } if text == "aλz"));
            }
        },
    );
}

#[test]
fn tool_output_span_reads_and_traces_the_result_message() {
    use polylogue_model::{
        Reference,
        read::{Card, ChainStep},
    };

    with_source(
        &[
            source_record(
                "a1",
                "assistant",
                json!([{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"synthetic input"}}]),
            ),
            source_record(
                "u1",
                "user",
                json!([
                    {"type":"text","text":"separate body"},
                    {"type":"tool_result","tool_use_id":"c1","content":"a💡z"}
                ]),
            ),
        ],
        |fixture, _| {
            let target = "span:tool:claude-code:stored#c1#u16=1-3";
            let output = cli(
                fixture,
                &[
                    "--deadline",
                    "none",
                    "--json",
                    target,
                    "read",
                    "--show",
                    "tools:full,harness:full",
                    "--budget",
                    "100000",
                ],
            );
            assert!(output.status.success(), "{output:?}");
            let read = read_page(json(&output));
            let headers: Vec<_> = read
                .document
                .items
                .iter()
                .filter(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
                .map(|item| item.source.reference.as_str())
                .collect();
            assert_eq!(headers, ["message:claude-code:u1"]);
            let marks: Vec<_> = read
                .document
                .items
                .iter()
                .filter(|item| {
                    matches!(
                        item.kind,
                        ItemKind::Indicator {
                            indicator: IndicatorKind::Selected,
                            ..
                        }
                    )
                })
                .collect();
            let [mark] = marks.as_slice() else {
                panic!("expected one selection mark: {read:?}");
            };
            assert_eq!(mark.source.reference, target);
            assert_eq!(mark.source.block, Some(1));
            assert_eq!(
                mark.source.message.as_ref().unwrap().to_string(),
                "claude-code:u1"
            );
            let calls: Vec<_> = read
                .document
                .items
                .iter()
                .filter_map(|item| match &item.kind {
                    ItemKind::Call { call } => Some(call),
                    _ => None,
                })
                .collect();
            let [call] = calls.as_slice() else {
                panic!("expected the call's result once: {read:?}");
            };
            assert_eq!(call.output.as_deref(), Some("a💡z"));
            assert!(call.input.is_none());

            let output = cli(
                fixture,
                &[
                    "--deadline",
                    "none",
                    "--json",
                    target,
                    "read",
                    "--view",
                    "trace",
                    "--budget",
                    "100000",
                ],
            );
            assert!(output.status.success(), "{output:?}");
            let traced = read_page(json(&output));
            let trace = traced
                .document
                .items
                .iter()
                .find_map(|item| match &item.kind {
                    ItemKind::Card { card } => match card.as_ref() {
                        Card::Trace(trace) => Some(trace),
                        _ => None,
                    },
                    _ => None,
                })
                .expect("result record trace");
            assert_eq!(trace.item, Reference::parse(target).unwrap());
            let records: Vec<_> = trace
                .chain
                .iter()
                .filter_map(|step| match step {
                    ChainStep::Record { reference } => Some(reference.to_string()),
                    _ => None,
                })
                .collect();
            assert_eq!(records, [target, "message:claude-code:u1"]);
        },
    );
}
