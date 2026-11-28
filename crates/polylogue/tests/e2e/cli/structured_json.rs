use super::stored_text::{bundle, page, source_record, texts, with_source};
use super::*;
use serde_json::json;

const SECRET: &str = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AAAAAAAA";

#[test]
fn crossing_quoted_credentials_are_preserved_by_typed_read_without_changing_source() {
    let expected = [
        "--secret=\"unfinished --token=\"synthetic-later-value\" tail-double",
        "--secret='unfinished --token='synthetic-later-value' tail-single",
    ];
    with_source(
        &[
            source_record("u1", "user", json!(expected[0])),
            source_record("u2", "user", json!(expected[1])),
        ],
        |fixture, _| {
            assert_eq!(texts(&page(fixture, &[])), expected);
        },
    );
}

#[test]
fn read_to_bundle_import_round_trip_preserves_structured_values_and_unknown_tool_fields() {
    let input = json!({"command":"printf synthetic","empty":"","nil":null,"count":0,"enabled":false,"nested":{"paths":["a","b"]},"unknown_provider_field":{"credential":SECRET}});
    with_source(
        &[source_record(
            "a1",
            "assistant",
            json!([{"type":"tool_use","id":"c1","name":"Bash","input":input}]),
        )],
        |fixture, directory| {
            let first = bundle(fixture, &directory.join("transfer"));
            let messages: Vec<_> = first
                .iter()
                .filter_map(|record| match record {
                    polylogue_model::bundle::Record::Message(record) => {
                        Some(record.message.clone())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(messages.len(), 1);
            let encoded = serde_json::to_value(&messages[0]).unwrap();
            assert_eq!(encoded["blocks"][0]["input"], input);
            let restored = crate::Fixture::new(
                &directory.join("restored"),
                &directory.join("empty-home"),
                &directory.join("restore-config"),
            );
            let _restored = start_daemon(&restored);
            let imported = cli(
                &restored,
                &[
                    "--deadline",
                    "none",
                    "--json",
                    "import",
                    directory.join("transfer").to_str().unwrap(),
                ],
            );
            assert!(imported.status.success(), "{imported:?}");
            let second = bundle(&restored, &directory.join("second"));
            let actual: Vec<_> = second
                .iter()
                .filter_map(|record| match record {
                    polylogue_model::bundle::Record::Message(record) => {
                        Some(record.message.clone())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(actual, messages);
        },
    );
}

#[test]
fn source_controls_are_escaped_and_html_remains_inert() {
    let expected =
        "λ🙂 --token=synthetic-later-value <script>alert('synthetic')</script>\u{1b}[31m\u{7}";
    with_source(
        &[source_record("u1", "user", json!(expected))],
        |fixture, directory| {
            assert_eq!(texts(&page(fixture, &[])), [expected]);
            let terminal = cli(
                fixture,
                &[
                    "--deadline",
                    "none",
                    "session:claude-code:stored",
                    "read",
                    "--format",
                    "terminal",
                ],
            );
            assert!(terminal.status.success(), "{terminal:?}");
            assert!(!terminal.stdout.contains(&0x1b) && !terminal.stdout.contains(&7));
            let path = directory.join("reading.html");
            wait_caught_up(fixture);
            let html = cli(
                fixture,
                &[
                    "--deadline",
                    "none",
                    "session:claude-code:stored",
                    "read",
                    "--format",
                    "html",
                    "--to",
                    path.to_str().unwrap(),
                ],
            );
            assert_eq!(html.status.code(), Some(0), "{html:?}");
            let html = std::fs::read_to_string(path).unwrap();
            assert!(!html.contains("<script>alert('synthetic')</script>"));
            assert!(html.contains("&lt;script&gt;") && html.contains("synthetic-later-value"));
        },
    );
}
