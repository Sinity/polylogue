use super::*;
use axum::body::{Body, to_bytes};
use axum::http::Request as Http;
use polylogue_model::{
    Reference,
    annotation::{Annotation, LinkRole},
};
use serde_json::{Value, json};
use tower::ServiceExt;

struct Fixture {
    daemon: crate::test_daemon::TestDaemon,
    web: Arc<Web>,
}
fn fixture() -> Fixture {
    let daemon = crate::test_daemon::start("synthetic workbench prompt");
    let web = Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "notes".into(),
        token: "token".into(),
        port: 8765,
    });
    Fixture { daemon, web }
}
fn run(test: impl std::future::Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(test);
}
async fn send_raw(
    f: &Fixture,
    operation: &str,
    id: &str,
    arguments: Value,
) -> (StatusCode, Vec<u8>) {
    let envelope = command_surface::Request {
        operation: operation.into(),
        arguments,
        request_id: id.into(),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let command = crate::commands::command(operation).expect("declared HTTP operation");
    let (builder, body) = if command.writes {
        let body = serde_json::to_vec(&envelope).unwrap();
        (Http::post(format!("/api/v1/{operation}")), Body::from(body))
    } else {
        let encoded = serde_json::to_value(&envelope).unwrap();
        let component = |text: &str| {
            text.bytes()
                .map(|byte| format!("%{byte:02X}"))
                .collect::<String>()
        };
        let pairs = encoded
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| {
                let value = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                format!("{}={}", component(key), component(&value))
            })
            .collect::<Vec<_>>()
            .join("&");
        (
            Http::get(format!("/api/v1/{operation}?{pairs}")),
            Body::empty(),
        )
    };
    let request = builder
        .header(header::ORIGIN, "http://127.0.0.1:8765")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::HOST, "127.0.0.1:8765")
        .header(header::COOKIE, "polylogue_notes=token")
        .body(body)
        .unwrap();
    let response = router(Arc::clone(&f.web)).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, bytes.to_vec())
}
async fn send(
    f: &Fixture,
    operation: &str,
    id: &str,
    arguments: Value,
) -> (StatusCode, command_surface::Answer<Value>) {
    let (status, bytes) = send_raw(f, operation, id, arguments).await;
    (status, serde_json::from_slice(&bytes).unwrap())
}
async fn command(f: &Fixture, operation: &str, id: &str, arguments: Value) -> Value {
    let (status, answer) = send(f, operation, id, arguments).await;
    assert_eq!(status, StatusCode::OK, "{answer:?}");
    let value = answer
        .value()
        .expect("successful answer has its typed value");
    match value["kind"].as_str() {
        Some("listing") => value["reply"].clone(),
        Some("steps") => match crate::test_answers::step(answer) {
            crate::protocol::line::StepAnswer::Wrote(reply) => serde_json::to_value(reply).unwrap(),
            crate::protocol::line::StepAnswer::WroteEach(reply) => {
                serde_json::to_value(reply).unwrap()
            }
            crate::protocol::line::StepAnswer::Saved(reply) => serde_json::to_value(reply).unwrap(),
            other => panic!("expected an authored step: {other:?}"),
        },
        _ => value.clone(),
    }
}
async fn query(f: &Fixture, text: &str) -> Value {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    command(
        f,
        "rows",
        &format!(
            "web-query-{}",
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ),
        json!({"selection":text}),
    )
    .await
}
fn note_line(kind: &str, key: Option<&str>, text: Value, about: &[String]) -> Value {
    json!({"references":about,"steps":[{"verb":"note","as":kind,"key":key,"text":text}]})
}
fn note(text: &str) -> Value {
    note_line("note", None, json!(text), &["repo:morrow".into()])
}
async fn write_one(f: &Fixture, id: &str, line: Value) -> Value {
    command(f, "rows", id, line).await
}
async fn revise(f: &Fixture, id: &str, root: &str, base: &str, value: Value) {
    command(f,"rows",id,json!({"selection":format!("annotation:{root}-1@{base}-1"),"steps":[{"verb":"edit","text":value}]})).await;
}
async fn stored(f: &Fixture, root: &str, revision: &str) -> Annotation {
    let reference = format!("annotation:{root}@{revision}");
    // The original object-read oracle now checks the complete answer stream.
    // No deadline is lifted and no generated child is rewritten.
    let (status, bytes) = send_raw(
        f,
        "rows",
        &format!("read-{revision}"),
        json!({"selection":reference,"steps":[{"verb":"read","format":"jsonl","delivery":"stream"}]}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let mut began = false;
    let mut completed = false;
    let mut found = Vec::new();
    crate::client::receive(&mut bytes.as_slice(), None, |frame| {
        match frame.event.as_str() {
            "begin" => {
                assert!(!began);
                let _: polylogue_model::read::ReadHeader =
                    serde_json::from_str(&frame.data).unwrap();
                began = true;
            }
            "record" => {
                assert!(began && !completed);
                let record: polylogue_model::read::ReadRecord =
                    serde_json::from_str(&frame.data).unwrap();
                if let Some(record) = record.record() {
                    match record {
                        polylogue_model::bundle::Record::Authored(row) => {
                            found.extend(
                                row.prepared
                                    .effect
                                    .annotations
                                    .iter()
                                    .filter(|fact| fact.root == root && fact.id == revision)
                                    .cloned(),
                            );
                        }
                        polylogue_model::bundle::Record::Annotation(row)
                            if row.annotation.root == root && row.annotation.id == revision =>
                        {
                            found.push(row.annotation.clone())
                        }
                        _ => {}
                    }
                }
            }
            "complete" => {
                assert!(began && !completed);
                completed = true;
            }
            other => panic!("unexpected annotation read frame {other}: {}", frame.data),
        }
        Ok(completed)
    })
    .unwrap();
    assert!(completed);
    let Reference::Annotation(exact) = Reference::parse(&reference).unwrap() else {
        unreachable!()
    };
    let row = f
        .daemon
        .daemon
        .reads
        .reader()
        .unwrap()
        .annotation(&exact)
        .unwrap()
        .unwrap();
    assert_eq!(
        found,
        vec![row.fact.clone()],
        "HTTP must return the whole annotation, including links, original authorship and clock"
    );
    row
}
fn agent_write(f: &Fixture, request: crate::test_daemon::Call) -> Value {
    use polylogue_model::authorship::{Authorship, Channel};
    let authorship = Authorship {
        author: polylogue_model::Author::Agent(Some(
            polylogue_model::SessionId::parse("claude-code:web-fixture-agent").unwrap(),
        )),
        via: Channel::Mcp,
    };
    let answer = request.authorship(authorship).answer(&f.daemon.daemon);
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    match crate::test_answers::step(answer) {
        crate::protocol::line::StepAnswer::Wrote(reply) => serde_json::to_value(reply).unwrap(),
        crate::protocol::line::StepAnswer::WroteEach(reply) => serde_json::to_value(reply).unwrap(),
        other => panic!("expected an authored step: {other:?}"),
    }
}
fn agent_assert(f: &Fixture, id: &str, value: &str) {
    agent_write(f, crate::test_daemon::Call::new("rows", note(value)).id(id));
}
fn about(row: &Annotation) -> Vec<String> {
    row.fact
        .links
        .iter()
        .filter(|link| link.role == LinkRole::About)
        .map(|link| link.reference.to_string())
        .collect()
}

#[test]
fn operator_edits_and_judgments_keep_earlier_revisions() {
    run(async {
        let f = fixture();
        write_one(&f, "own", note("first <script>")).await;
        revise(&f, "edit", "own", "own", json!("second")).await;
        let old = stored(&f, "own-1", "own-1").await;
        assert_eq!(old.fact.value, Some(json!("first <script>")));
        assert_eq!(old.fact.author, polylogue_model::Author::Operator);
        assert_eq!(old.fact.via, polylogue_model::authorship::Channel::Web);
        agent_assert(&f, "agent", "proposal");
        for (id, key, reason) in [
            ("accept", "accepted", "checked"),
            ("reject", "rejected", "reconsidered"),
        ] {
            command(&f, "rows", id, json!({"selection":"annotation:agent-1@agent-1","steps":[{"verb":"judge","verdict":if key == "accepted" {"accept"} else {"reject"},"reason":reason}]})).await;
        }
        // Judging the revision again revises the operator's judgment of it.
        let current = query(
            &f,
            "annotations where kind:judgment about:\"annotation:agent-1@agent-1\"",
        )
        .await;
        assert_eq!(current["count"], 1, "{current}");
        assert_eq!(current["rows"][0]["id"], "reject-1", "{current}");
        assert_eq!(
            query(
                &f,
                "annotations where kind:judgment current:* about:\"annotation:agent-1@agent-1\""
            )
            .await["count"],
            2
        );
        agent_write(
            &f,
            crate::test_daemon::Call::new("rows", json!({"selection":"annotation:agent-1@agent-1", "steps":[{"verb":"edit","text":"new proposal"}]})).id("agent-new"),
        );
        let pending = query(&f, "annotations where agent:true judged:unknown").await;
        assert_eq!(pending["count"], 1);
        assert_eq!(pending["rows"][0]["id"], "agent-new-1");
        let retry = command(&f,"rows","reject",json!({"selection":"annotation:agent-1@agent-1","steps":[{"verb":"judge","verdict":"reject","reason":"reconsidered"}]})).await;
        assert_eq!(retry["repeat"], true);
        assert_eq!(
            query(
                &f,
                "annotations where kind:judgment current:* about:\"annotation:agent-1@agent-1\""
            )
            .await["count"],
            2
        );
        command(
            &f,
            "rows",
            "withdraw-agent",
            json!({"selection":"annotation:agent-1@agent-new-1","steps":[{"verb":"withdraw","reason":"closed"}]}),
        )
        .await;
        assert_eq!(
            query(&f, "annotations where agent:true judged:unknown").await["count"],
            0
        );
        let revision = stored(&f, "agent-1", "agent-new-1").await;
        assert_eq!(revision.fact.value, Some(json!("new proposal")));
        assert_eq!(
            revision.fact.author,
            polylogue_model::Author::Agent(Some(
                polylogue_model::SessionId::parse("claude-code:web-fixture-agent").unwrap()
            ))
        );
        assert_eq!(revision.fact.via, polylogue_model::authorship::Channel::Mcp);
    });
}

#[test]
fn selected_current_and_kept_rows_have_distinct_populations() {
    run(async {
        let f = fixture();
        let all = query(&f, "messages").await;
        let first_id = all["rows"][0]["id"].as_str().unwrap();
        assert_eq!(
            command(
                &f,
                "rows",
                "kept",
                json!({"selection":"annotations where kind:question","steps":[{"verb":"save"}]})
            )
            .await["saved_rows"]["result"],
            "result:kept-1"
        );
        assert_eq!(
            query(&f, "annotations where kind:question").await["count"],
            0
        );
        write_one(
            &f,
            "question",
            note_line(
                "question",
                None,
                json!("new member"),
                &["repo:morrow".into()],
            ),
        )
        .await;
        let result=command(&f,"rows","bulk",json!({"selection":"annotations where kind:question","all":true,"steps":[{"verb":"note","text":"all when applied"}]})).await;
        assert_eq!(result["targets"], 1);
        assert_eq!(
            about(&stored(&f, "bulk-1", "bulk-1").await),
            vec!["annotation:question-1@question-1"]
        );
        write_one(
            &f,
            "selected",
            note_line(
                "note",
                None,
                json!("selected only"),
                &[format!("message:{first_id}")],
            ),
        )
        .await;
        assert_eq!(
            query(
                &f,
                "annotations where about:\"annotation:question-1@question-1\""
            )
            .await["count"],
            1
        );
        assert_eq!(query(&f, "result:kept-1").await["count"], 0);
        command(
            &f,
            "rows",
            "kept-questions",
            json!({"selection":"annotations where kind:question","steps":[{"verb":"save"}]}),
        )
        .await;
        write_one(
            &f,
            "another",
            note_line(
                "question",
                None,
                json!("later member"),
                &["repo:morrow".into()],
            ),
        )
        .await;
        let args = json!({"selection":"result:kept-questions-1","all":true,"steps":[{"verb":"note","text":"fixed reviewed population"}]});
        assert_eq!(
            command(&f, "rows", "fixed", args.clone()).await["targets"],
            1
        );
        assert_eq!(command(&f, "rows", "fixed", args).await["repeat"], true);
        assert_eq!(
            about(&stored(&f, "fixed-1", "fixed-1").await),
            vec!["annotation:question-1@question-1"]
        );
    });
}

#[test]
fn editing_keyed_annotations_preserves_relations_and_expiry() {
    run(async {
        let f = fixture();
        write_one(&f, "other", note("independent evidence")).await;
        let mut content = note_line(
            "finding",
            Some("result"),
            json!("original"),
            &["repo:morrow".into()],
        );
        content["steps"][0]["valid_until"] = json!("2099-01-01T00:00:00Z");
        content["steps"][0]["derived_from"] = json!(["annotation:other-1@other-1"]);
        write_one(&f, "keyed", content).await;
        revise(&f, "keyed-edit", "keyed", "keyed", json!("edited")).await;
        let row = stored(&f, "keyed-1", "keyed-edit-1").await;
        assert_eq!(row.fact.key.as_deref(), Some("result"));
        assert_eq!(row.fact.root, "keyed-1");
        assert_eq!(row.fact.value, Some(json!("edited")));
        assert_eq!(
            row.fact.valid_until.unwrap().to_string(),
            "2099-01-01T00:00:00Z"
        );
        assert!(
            row.fact
                .links
                .iter()
                .any(|link| link.role == LinkRole::DerivedFrom
                    && link.reference.to_string() == "annotation:other-1@other-1")
        );
        assert!(
            row.fact
                .links
                .iter()
                .any(|link| link.role == LinkRole::Revises
                    && link.reference.to_string() == "annotation:keyed-1@keyed-1")
        );
        assert_eq!(about(&row), vec!["repo:morrow"]);
    });
}

#[test]
fn semantic_and_mixed_annotation_use_the_public_query_population() {
    run(async {
        use std::io::{Read, Write};
        let f = fixture();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let provider = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
                if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            line.split_once(':')
                                .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                                .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let body = r#"{"data":[{"index":0,"embedding":[1.0,0.0]}]}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        for (key, value) in [
            ("embed.provider", "local"),
            ("embed.calls", "on"),
            ("embed.model", "notes-test"),
            ("embed.endpoint", endpoint.as_str()),
        ] {
            command(&f, "set", key, json!({"key":key,"value":value})).await;
        }
        {
            let reader = f.daemon.daemon.reads.reader().unwrap();
            let hashes = reader
                .query("SELECT hex(hash) AS hash FROM text", &[])
                .unwrap();
            let vectors = hashes
                .iter()
                .map(|row| {
                    let text = row["hash"].as_str().unwrap();
                    polylogue_store::vectors::Vector {
                        text_hash: (0..text.len())
                            .step_by(2)
                            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
                            .collect(),
                        chunk: 0,
                        values: vec![1.0, 0.0],
                    }
                })
                .collect::<Vec<_>>();
            reader
                .vectors_writer()
                .unwrap()
                .put(
                    "local/notes-test/2/document/unit",
                    &vectors,
                    &polylogue_model::WallClock::new(jiff::Timestamp::now),
                )
                .unwrap();
        }
        for (n, text) in [
            "messages where similar:synthetic origin:claude-code",
            "messages where synthetic similar:synthetic origin:claude-code",
        ]
        .into_iter()
        .enumerate()
        {
            let expected = query(&f, text).await;
            assert_eq!(expected["ranked"], true);
            let id = format!("semantic-{n}");
            let annotation = format!("reviewed semantic population {n}");
            let applied = command(
                &f,
                "rows",
                &id,
                json!({"selection":text,"all":true,"steps":[{"verb":"note","text":annotation}]}),
            )
            .await;
            assert_eq!(applied["targets"], expected["count"]);
            let selected = expected["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| format!("message:{}", row["id"].as_str().unwrap()))
                .collect::<std::collections::BTreeSet<_>>();
            let reader = f.daemon.daemon.reads.reader().unwrap();
            let recorded = reader.user_request(&format!("{id}-1")).unwrap().unwrap();
            let polylogue_model::user::UserWrite::AssertQuery { resolved, .. } = recorded.write
            else {
                panic!("bulk request must keep its recorded population")
            };
            assert_eq!(resolved.len(), selected.len());
            let actual = resolved
                .iter()
                .map(ToString::to_string)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(actual, selected);
            for reference in resolved {
                let mut matching = Vec::new();
                let flow = reader
                    .visit_annotation_references_linked(
                        &reference,
                        &[polylogue_model::annotation::LinkRole::About],
                        |revision| {
                            if let Some(stored) = reader.annotation(&revision)?
                                && stored.fact.value == Some(json!(annotation))
                            {
                                matching.push(stored.fact.id);
                            }
                            Ok(std::ops::ControlFlow::Continue(()))
                        },
                    )
                    .unwrap();
                assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
                assert_eq!(
                    matching.len(),
                    1,
                    "every committed population member gets exactly one new annotation in this run"
                );
            }
        }
        provider.join().unwrap();
    });
}

#[test]
fn bulk_retry_uses_committed_targets_after_the_query_becomes_empty() {
    run(async {
        let f = fixture();
        write_one(
            &f,
            "question",
            note_line("question", None, json!("open"), &["repo:morrow".into()]),
        )
        .await;
        let args = json!({"selection":"annotations where kind:question","all":true,"steps":[{"verb":"note","text":"reviewed"}]});
        assert_eq!(
            command(&f, "rows", "bulk-retry", args.clone()).await["targets"],
            1
        );
        command(
            &f,
            "rows",
            "closed",
            json!({"selection":"annotation:question-1@question-1","steps":[{"verb":"withdraw"}]}),
        )
        .await;
        assert_eq!(
            query(&f, "annotations where kind:question").await["count"],
            0
        );
        let repeated = command(&f, "rows", "bulk-retry", args.clone()).await;
        assert_eq!(repeated["repeat"], true);
        assert_eq!(repeated["targets"], 1);
        assert_eq!(
            about(&stored(&f, "bulk-retry-1", "bulk-retry-1").await),
            vec!["annotation:question-1@question-1"]
        );
        for field in ["selection", "steps"] {
            let mut changed = args.clone();
            changed[field] = if field == "selection" {
                json!("sessions")
            } else {
                json!([{"verb":"note","text":"different"}])
            };
            assert_eq!(
                send(&f, "rows", "bulk-retry", changed).await.0,
                StatusCode::CONFLICT
            );
        }
    });
}

#[test]
fn annotation_fields_and_saved_query_edits_keep_metadata() {
    run(async {
        let f = fixture();
        command(
            &f,
            "rows",
            "Saved:Case",
            json!({"selection":"messages","steps":[{"verb":"save","name":"morrow"}]}),
        )
        .await;
        command(
            &f,
            "rows",
            "Saved:Case:new",
            json!({"selection":"sessions","steps":[{"verb":"save","name":"morrow"}]}),
        )
        .await;
        let page=command(&f,"rows","projection",json!({"selection":"annotations where root = \"Saved:Case-1\" current:*","fields":["id","key","about","judged"]})).await;
        assert_eq!(page["count"], 2);
        let mut ids: Vec<&str> = page["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, ["Saved:Case-1", "Saved:Case:new-1"]);
        assert!(
            page["rows"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["key"] == "morrow")
        );
        assert_eq!(
            query(&f, "annotations where root = \"saved:case-1\" current:*").await["count"],
            0
        );
        let row = stored(&f, "Saved:Case-1", "Saved:Case:new-1").await;
        assert_eq!(row.fact.key.as_deref(), Some("morrow"));
        assert_eq!(row.fact.root, "Saved:Case-1");
        assert_eq!(row.fact.value.as_ref().unwrap()["text"], "sessions");
        assert!(
            row.fact
                .links
                .iter()
                .any(|link| link.role == LinkRole::Revises
                    && link.reference.to_string() == "annotation:Saved:Case-1@Saved:Case-1")
        );
        for field in ["derived_from", "valid_until"] {
            let mut invalid = note("invalid");
            invalid["steps"][0][field] = json!(if field == "derived_from" {
                "invalid"
            } else {
                "tomorrow"
            });
            assert_eq!(
                send(&f, "rows", "invalid", invalid).await.0,
                StatusCode::BAD_REQUEST
            );
        }
    });
}

#[test]
fn grouped_annotation_population_returns_a_typed_refusal() {
    run(async {
        let f = fixture();
        let (status,answer)=send(&f,"rows","grouped",json!({"selection":"sessions | group by origin","all":true,"steps":[{"verb":"note","text":"review"}]})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
        let error = crate::test_answers::refusal(answer);
        assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
        assert_eq!(error.code.as_deref(), Some("bad_request"));
    });
}
