use super::*;
use command_surface::{Deadline, JsonSchema, OperationDeclarations, OperationSpec, ToolFamily};
use polylogue_model::{ChangePosition, Timestamp, read::*};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
#[derive(Serialize, Deserialize, JsonSchema)]
struct Arguments {
    target: String,
    view: Option<View>,
}
fn read_table() -> Arc<OperationDeclarations> {
    let mut declarations = OperationDeclarations::builder("polylogue", Default::default());
    declarations
        .declare_tool(ToolFamily {
            name: "read".into(),
            description: "Read a synthetic record".into(),
            open_world: false,
            idempotent: true,
        })
        .unwrap();
    let mut spec = OperationSpec::request(
        "read",
        &["read"],
        "read",
        "Read a synthetic record",
        "Test shared frame integration",
    );
    spec.http_route = Some("/api/v1/read".into());
    spec.failures = vec![FailureKind::BadRequest];
    declarations
        .declare::<Arguments, Value>(spec, |answer, _| answer.to_string())
        .unwrap();
    declarations.finish()
}
fn page() -> ReadPage {
    ReadPage {
        format: Format::Html,
        snapshot: ReadPageSnapshot {
            query: format!("query:{}", "a".repeat(64)),
            ran_at: Timestamp::from_unix_ms(0).unwrap(),
            read_at: ChangePosition {
                generation: "synthetic".into(),
                index: 4,
                archive: 7,
            },
            freshness: polylogue_model::freshness::ReadFreshness {
                local: polylogue_model::freshness::ReadCompleteness::empty(),
                sinex: IndexFreshness::primary(Some(4), Some(7), Some(1.), Some(true), None),
            },
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
                zone: None,
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
fn typed_page_shares_the_frame_and_shows_freshness() {
    let table = read_table();
    let request = RunnableRequest::new(
        "read",
        json!({"target":"session:test:session","view":"transcript","cursor":"old"}),
        "synthetic",
    );
    let output = read(&page(), &table, &request).unwrap();
    assert!(output.contains("/assets/shell.js"));
    assert!(output.contains("/assets/tokens.css"));
    assert!(output.contains("&lt;script&gt;"));
    assert!(!output.contains("<script>archived"));
    assert!(output.contains(
        "<span data-mode=\"primary\" data-state=\"behind\">Primary: behind · applied 4 · Sinex 7 · observed 1 seconds ago · reachable true · lag 3</span>"
    ));
    assert!(output.contains("polylogue read --request"));
    assert!(output.contains("Read view"));
}
#[test]
fn unavailable_sinex_note_carries_its_own_state() {
    let request = RunnableRequest::new(
        "read",
        json!({"target":"session:test:session","view":"transcript"}),
        "synthetic",
    );
    let mut unavailable = page();
    unavailable.snapshot.freshness.sinex =
        IndexFreshness::primary(Some(4), Some(7), Some(1.), Some(false), None);
    let output = read(&unavailable, &read_table(), &request).unwrap();
    assert!(output.contains("Sinex is unavailable"));
    assert!(!output.contains("data-state=\"behind\""));
    assert_eq!(output.matches("data-state=\"unavailable\"").count(), 2);
}
#[test]
fn continuation_encodes_the_complete_ready_call_without_decoding_its_cursor() {
    let call = RunnableRequest {
        cursor: Some("sealed +λ&?".into()),
        ..RunnableRequest::new(
            "read",
            json!({"target":"session:test:session","alternate":["en","de"],"flag":false}),
            "synthetic",
        )
    };
    let url = crate::ready::http_url(&read_table(), &call).unwrap();
    let fields = url_parameters(&url);
    assert_eq!(fields["cursor"], "sealed +λ&?");
    let args: Value = serde_json::from_str(&fields["arguments"]).unwrap();
    assert_eq!(args, call.arguments);
}
#[test]
fn continuation_links_use_service_calls_instead_of_template_cursor_state() {
    let mut page = page();
    page.document.footer.next = Some(ReadyCall {
        request: RunnableRequest {
            cursor: Some("sealed".into()),
            ..RunnableRequest::new(
                "read",
                json!({"target":"session:test:session"}),
                "synthetic",
            )
        },
    });
    let output = navigation(&page, &read_table()).unwrap().into_string();
    assert_eq!(
        output,
        "<nav class=\"paging\" aria-label=\"Content pages\"><a href=\"/api/v1/read?operation=read&amp;arguments=%7B%22target%22%3A%22session%3Atest%3Asession%22%7D&amp;request_id=synthetic&amp;cursor=sealed\" rel=\"next\">Next</a></nav>"
    );
}

#[test]
fn typed_lineage_issues_keep_unknowns_and_evidence_in_html_and_terminal() {
    let mut page = page();
    page.document.items =
        vec![
        ReadItem {
            source: Source::record("session:codex:c"),
            kind: ItemKind::Indicator {
                indicator: IndicatorKind::LinkIssue,
                text:
                    "missing_branch_point · parent codex:p · child codex:c · native.<field> · <unknown>"
                        .into(),
                expand: None,
            },
        },
        ReadItem {
            source: Source::record("session:codex:p"),
            kind: ItemKind::Card {
                card: Box::new(Card::LinkNode {
                    reference: "session:codex:p".into(),
                    lineage: Some(Box::new(polylogue_model::read::SessionLineage {
                        root:None,depth:None,root_pending:true,
                        issue:Some(polylogue_model::read::LineageIssue::MissingParent {
                            child:polylogue_model::SessionId::parse("codex:p").unwrap(),
                            parent:polylogue_model::SessionId::parse("codex:<missing>").unwrap(),
                        }),
                    })),
                    title: ValueState::Unknown {
                        reason: Some("not stored".into()),
                    },
                    parents: vec![],
                    own_messages: ValueState::known(0),
                    inherited_messages: ValueState::Unknown {
                        reason: Some("missing branch point".into()),
                    },
                    own_cost: ValueState::NotRecorded,
                    tree_cost: ValueState::default(),
                    asked: false,
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
    let mut text = Vec::new();
    crate::encode_document(
        &page.document,
        crate::Format::Terminal,
        crate::EncodeOptions::default(),
        &mut text,
    )
    .unwrap();
    let text = String::from_utf8(text).unwrap();
    for line in [
        "· missing_branch_point · parent codex:p · child codex:c · native.<field> · <unknown>",
        "↳ session:codex:p · unknown (not stored) · parents  · 0 own + unknown (missing branch point) inherited · cost not recorded own / unknown tree",
        "root pending · parent codex:<missing> of codex:p is not available",
    ] {
        assert!(text.lines().any(|actual| actual == line), "{line}: {text}");
    }
    let html = String::from_utf8(output).unwrap();
    for fragment in [
        "<aside class=\"indicator\" data-kind=\"link_issue\">· missing_branch_point · parent codex:p · child codex:c · native.&lt;field&gt; · &lt;unknown&gt;</aside>",
        "<h3>session:codex:p · <span class=\"value-state\" data-state=\"unknown\">unknown (not stored)</span></h3>",
        "<p>root pending · parent codex:&lt;missing&gt; of codex:p is not available</p>",
        "<p><span class=\"value-state\" data-state=\"known\">0</span> own + <span class=\"value-state\" data-state=\"unknown\">unknown (missing branch point)</span> inherited · cost <span class=\"value-state\" data-state=\"not-recorded\">not recorded</span> own / <span class=\"value-state\" data-state=\"unknown\">unknown</span> tree</p>",
    ] {
        assert!(html.contains(fragment), "{fragment}: {html}");
    }
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
        deadline: Deadline::Milliseconds(1234),
        cursor: Some("request-old-cursor".into()),
        ..RunnableRequest::new(
            "read",
            json!({"target":"session:test:session","view":"transcript","show":["tools:line","harness:line","thinking:full","inherited","branches"],"cursor":"sealed-old"}),
            "synthetic",
        )
    };
    let fields = url_parameters(&view_url(&read_table(), &request, View::Dialogue).unwrap());
    let args: Value = serde_json::from_str(&fields["arguments"]).unwrap();
    assert_eq!(args["view"], "dialogue");
    assert_eq!(args["show"], json!(["inherited", "branches"]));
    assert!(args.get("cursor").is_none());
    assert_eq!(fields["request_id"], "synthetic");
    assert!(!fields.contains_key("cursor"));
    assert_eq!(fields["operation"], "read");
    let deadline: Value = serde_json::from_str(&fields["deadline"]).unwrap();
    assert_eq!(deadline, json!(1234));
    let fields = url_parameters(&view_url(&read_table(), &request, View::Overview).unwrap());
    let args: Value = serde_json::from_str(&fields["arguments"]).unwrap();
    assert_eq!(args["show"], json!([]));
    assert_eq!(args["target"], "session:test:session");
}

#[test]
fn copied_current_invocation_preserves_controls_as_valid_json_escapes() {
    let table = read_table();
    let request = RunnableRequest::new(
        "read",
        json!({"target":"messages where text:'\u{9b}\u{202e}'","view":"transcript"}),
        "synthetic",
    );
    let output = read(&page(), &table, &request).unwrap();
    assert!(!output.contains('\u{9b}'));
    assert!(!output.contains('\u{202e}'));
    assert!(output.contains("\\u009b"));
    assert!(output.contains("\\u202e"));
    assert!(!output.contains("\\x9b"));
    let copied: Value = output
        .split("data-copy=\"")
        .skip(1)
        .find_map(|attribute| {
            let encoded = attribute.split_once('"')?.0;
            let decoded = encoded
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&");
            let target: Value = serde_json::from_str(&decoded).ok()?;
            (target["label"] == "page").then_some(target)
        })
        .expect("page command copy target");
    let command = copied["forms"][0]["text"].as_str().unwrap();
    let shell = std::process::Command::new("sh")
        .args(["-c", &format!("printf '%s\\0' {command}")])
        .output()
        .unwrap();
    assert!(shell.status.success());
    let words = shell
        .stdout
        .strip_suffix(&[0])
        .unwrap()
        .split(|byte| *byte == 0)
        .map(|word| String::from_utf8(word.to_vec()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(words[0], "polylogue");
    let parsed = table.parse_cli(&words[1..], "unused").unwrap();
    assert_eq!(parsed.request, request);
}
