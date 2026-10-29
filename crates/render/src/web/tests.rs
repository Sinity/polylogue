use super::*;
use command_surface::{
    Answer, Deadline, JsonSchema, LatencyClass, Operation, OperationFuture, OperationSpec,
    RequestContext,
};
use polylogue_model::{Author, ChangePosition, Timestamp, bundle::ReadSnapshot, read::*};
use serde::{Deserialize, Serialize};
use serde_json::json;
#[derive(Serialize, Deserialize, JsonSchema)]
struct Arguments {
    target: String,
    view: Option<View>,
}
struct Read;
impl Operation for Read {
    type Request = Arguments;
    type Response = Value;
    fn spec(&self) -> OperationSpec {
        OperationSpec {
            name: "read".into(),
            command: vec!["read".into()],
            summary: "Read a synthetic record".into(),
            reason: "Test shared frame integration".into(),
            http_route: "/api/v1/read".into(),
            latency: LatencyClass::Read,
            failures: vec![FailureKind::BadRequest],
            write: false,
        }
    }
    fn execute(&self, _: Arguments, _: RequestContext) -> OperationFuture<'_, Answer<Value>> {
        Box::pin(async { Answer::ok(Value::Null) })
    }
}
fn page() -> ReadPage {
    ReadPage {
        snapshot: ReadSnapshot {
            query: format!("query:{}", "a".repeat(64)),
            query_text: "sessions".into(),
            ran_at: Timestamp::from_unix_ms(0).unwrap(),
            read_at: ChangePosition {
                generation: "synthetic".into(),
                index: 4,
                archive: 7,
            },
            caught_up: false,
            currency: IndexCurrency::primary(Some(4), Some(7), Some(1.), Some(true), None),
            enabled_variants: vec![],
            author: Author::Operator,
            notes: vec![],
            incomplete: vec![],
        },
        records: vec![],
        document: ReadDocument {
            header: ReadHeader {
                target: "session:test:session".into(),
                view: View::Transcript,
                title: ValueState::known("Synthetic <title>".into()),
                kind: HeaderKind::Record {
                    record_type: "test".into(),
                },
                figures: vec![],
            },
            items: vec![ReadItem {
                source: Source::record("block:test:session:message#0"),
                kind: ItemKind::Text {
                    text: "<script>archived()</script>".into(),
                },
            }],
            footer: ReadFooter::default(),
        },
    }
}
#[test]
fn typed_page_uses_one_common_frame_and_keeps_read_currency_visible() {
    let mut table = OperationTable::new("polylogue");
    table.register(Read).unwrap();
    let request = RunnableRequest {
        operation: "read".into(),
        arguments: json!({"target":"session:test:session","view":"transcript","cursor":"old"}),
        request_id: "synthetic".into(),
        deadline: Deadline::Default,
    };
    let output = read(&page(), &table, &request).unwrap();
    assert!(output.contains("/assets/shell.js"));
    assert!(output.contains("/assets/tokens.css"));
    assert!(!output.contains("reader.js"));
    assert!(!output.contains("reader.css"));
    assert!(output.contains("&lt;script&gt;"));
    assert!(!output.contains("<script>archived"));
    assert!(output.contains("data-row=\"\""));
    assert!(output.contains("data-state=\"behind\""));
    assert!(output.contains("applied 4"));
    assert!(output.contains("owner 7"));
    assert!(output.contains("polylogue read --request"));
    assert!(output.contains("Read view"));
}
#[test]
fn continuation_encodes_the_complete_ready_call_without_decoding_its_cursor() {
    let call = HttpCall {
        method: "GET".into(),
        path: "/api/v1/read".into(),
        arguments: json!({"target":"session:test:session","cursor":"sealed +λ&?","variant":["en","de"],"flag":false}),
    };
    let url = http_url(&call).unwrap();
    assert!(url.contains("cursor=sealed%20%2B%CE%BB%26%3F"));
    assert!(url.contains("variant=en&variant=de"));
    assert!(url.contains("flag=false"));
    assert!(url.contains("target=session%3Atest%3Asession"));
    for path in [
        "https://example.test/read",
        "//example.test/read",
        "javascript:run",
        "/read?target=other",
    ] {
        assert!(
            http_url(&HttpCall {
                path: path.into(),
                ..call.clone()
            })
            .is_err()
        );
    }
}
#[test]
fn continuation_links_use_service_calls_instead_of_template_cursor_state() {
    let mut page = page();
    page.document.footer.next = Some(ReadyCall {
        cli: "polylogue read --request synthetic".into(),
        mcp: json!({"target":"session:test:session"}),
        http: HttpCall {
            method: "GET".into(),
            path: "/api/v1/read".into(),
            arguments: json!({"target":"session:test:session","cursor":"sealed"}),
        },
    });
    let output = navigation(&page).into_string();
    assert!(output.contains("data-next"));
    assert!(output.contains("cursor=sealed"));
    assert!(!output.contains("with_parents"));
}

#[test]
fn typed_lineage_issues_keep_unknowns_and_exact_evidence_in_every_encoder() {
    let mut page = page();
    page.document.items = vec![
        ReadItem {
            source: Source::record("session:codex:c"),
            kind: ItemKind::Indicator {
                indicator: IndicatorKind::LinkIssue,
                text:
                    "missing_anchor · parent codex:p · child codex:c · native.<field> · <unknown>"
                        .into(),
                expand: None,
            },
        },
        ReadItem {
            source: Source::record("session:codex:p"),
            kind: ItemKind::Card {
                card: Box::new(Card::LinkNode {
                    reference: "session:codex:p".into(),
                    title: ValueState::Unknown {
                        reason: Some("not stored".into()),
                    },
                    parents: vec![],
                    own_messages: ValueState::known(0),
                    inherited_messages: ValueState::Unknown {
                        reason: Some("missing anchor".into()),
                    },
                    own_cost: ValueState::NotRecorded,
                    tree_cost: ValueState::default(),
                    seed: false,
                    open: None,
                }),
            },
        },
    ];
    let mut output = vec![];
    crate::encode_document(
        &page.document,
        crate::Format::Html,
        crate::EncodeOptions::default(),
        &mut output,
    )
    .unwrap();
    let html = String::from_utf8(output).unwrap();
    assert!(html.contains("missing_anchor"));
    assert!(html.contains("native.&lt;field&gt;"));
    assert!(!html.contains("native.<field>"));
    assert!(html.contains("not stored"));
    assert!(html.contains("missing anchor"));
    assert!(html.contains("data-state=\"measured\">0</span> own"));
}

fn url_parameters(url: &str) -> std::collections::BTreeMap<String, String> {
    fn decode(text: &str) -> String {
        let mut bytes = vec![];
        let mut input = text.as_bytes().iter().copied();
        while let Some(byte) = input.next() {
            if byte == b'%' {
                let digits = [input.next().unwrap(), input.next().unwrap()];
                bytes.push(u8::from_str_radix(std::str::from_utf8(&digits).unwrap(), 16).unwrap());
            } else {
                bytes.push(byte);
            }
        }
        String::from_utf8(bytes).unwrap()
    }
    url.split_once('?')
        .unwrap()
        .1
        .split('&')
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap();
            (decode(key), decode(value))
        })
        .collect()
}
#[test]
fn view_switches_use_model_option_acceptance_and_preserve_request_metadata() {
    let request = RunnableRequest {
        operation: "read".into(),
        arguments: json!({"target":"session:test:session","view":"transcript","tools":"headers","harness":"headers","thinking":"full","inherited":true,"branches":true,"cursor":"sealed-old"}),
        request_id: "synthetic".into(),
        deadline: Deadline::Milliseconds(1234),
    };
    let fields = url_parameters(&view_url(&request, "/api/v1/read", View::Dialogue).unwrap());
    let args: Value = serde_json::from_str(&fields["arguments"]).unwrap();
    assert_eq!(args["view"], "dialogue");
    assert_eq!(args["inherited"], true);
    assert_eq!(args["branches"], true);
    for key in ["tools", "thinking", "harness", "cursor"] {
        assert!(args.get(key).is_none());
    }
    assert_eq!(fields["request_id"], "synthetic");
    assert_eq!(fields["operation"], "read");
    let deadline: Value = serde_json::from_str(&fields["deadline"]).unwrap();
    assert_eq!(deadline, serde_json::to_value(request.deadline).unwrap());
    let fields = url_parameters(&view_url(&request, "/api/v1/read", View::Summary).unwrap());
    let args: Value = serde_json::from_str(&fields["arguments"]).unwrap();
    assert!(args.get("inherited").is_none());
    assert!(args.get("branches").is_none());
    assert_eq!(args["target"], "session:test:session");
}

#[test]
fn copied_current_invocation_preserves_controls_as_valid_json_escapes() {
    let mut table = OperationTable::new("polylogue");
    table.register(Read).unwrap();
    let request = RunnableRequest {
        operation: "read".into(),
        arguments: json!({"target":"messages where text:'\u{9b}\u{202e}'","view":"transcript"}),
        request_id: "synthetic".into(),
        deadline: Deadline::Default,
    };
    let output = read(&page(), &table, &request).unwrap();
    assert!(!output.contains('\u{9b}'));
    assert!(!output.contains('\u{202e}'));
    assert!(output.contains("\\u009b"));
    assert!(output.contains("\\u202e"));
    assert!(!output.contains("\\x9b"));
}
