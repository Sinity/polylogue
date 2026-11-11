use super::*;
use axum::body::{Body, to_bytes};
use axum::http::Request as Http;
use polylogue_model::{
    Reference,
    assertion::{Assertion, LinkRole, NewAssertion},
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
        operations: Arc::clone(daemon.operations()),
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
    answer
        .value()
        .expect("successful canonical answer has its typed value")
        .clone()
}
async fn query(f: &Fixture, text: &str) -> Value {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    command(
        f,
        "query",
        &format!(
            "web-query-{}",
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ),
        json!({"query":text}),
    )
    .await
}
fn statement(kind: &str, key: Option<&str>, value: Value, about: &[String]) -> Value {
    json!({"kind":kind,"key":key,"value":value,"valid_until":null,"links":about.iter().map(|reference|json!({"role":"about","reference":reference})).collect::<Vec<_>>()})
}
fn note(text: &str) -> Value {
    statement("note", None, json!(text), &["repo:morrow".into()])
}
async fn assert_one(f: &Fixture, id: &str, statement: Value) -> Value {
    command(f, "assert", id, json!({"assertions":[statement]})).await
}
async fn revise(f: &Fixture, id: &str, root: &str, base: &str, value: Value) {
    command(f,"revise",id,json!({"bases":[format!("assertion:{root}@{base}")],"patch":{"value":{"change":"set","value":value}}})).await;
}
async fn stored(f: &Fixture, root: &str, revision: &str) -> Assertion {
    let reference = format!("assertion:{root}@{revision}");
    // The original object-read oracle now checks the complete canonical stream.
    // No deadline is lifted and no generated child is rewritten.
    let (status, bytes) = send_raw(
        f,
        "read",
        &format!("read-{revision}"),
        json!({"target":reference,"format":"jsonl","delivery":"stream"}),
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
    crate::client::stream::receive(bytes.as_slice(), |frame| {
        match frame.event.as_str() {
            "begin" => {
                assert!(!began);
                let _: polylogue_model::read::ReadBegin =
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
                                    .assertions
                                    .iter()
                                    .filter(|fact| fact.root == root && fact.id == revision)
                                    .cloned(),
                            );
                        }
                        polylogue_model::bundle::Record::Assertion(row)
                            if row.assertion.root == root && row.assertion.id == revision =>
                        {
                            found.push(row.assertion.clone())
                        }
                        _ => {}
                    }
                }
            }
            "complete" => {
                assert!(began && !completed);
                completed = true;
            }
            other => panic!("unexpected assertion read frame {other}: {}", frame.data),
        }
        Ok(completed)
    })
    .unwrap();
    assert!(completed);
    let Reference::Assertion(exact) = Reference::parse(&reference).unwrap() else {
        unreachable!()
    };
    let row = f
        .daemon
        .daemon
        .reads
        .reader()
        .unwrap()
        .assertion(&exact)
        .unwrap()
        .unwrap();
    assert_eq!(
        found,
        vec![row.fact.clone()],
        "HTTP must retain the complete exact fact, including links, original authorship and clock"
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
    answer.value().unwrap().clone()
}
fn agent_assert(f: &Fixture, id: &str, value: &str) {
    agent_write(
        f,
        crate::test_daemon::Call::new("assert", serde_json::json!({"assertions": [serde_json::from_value::<NewAssertion>(note(value)).unwrap()]})).id(id),
    );
}
fn about(row: &Assertion) -> Vec<String> {
    row.fact
        .links
        .iter()
        .filter(|link| link.role == LinkRole::About)
        .map(|link| link.reference.to_string())
        .collect()
}

#[test]
fn operator_edits_and_judgments_of_agent_revisions_keep_exact_evidence() {
    run(async {
        let f = fixture();
        assert_one(&f, "own", note("first <script>")).await;
        revise(&f, "edit", "own", "own", json!("second")).await;
        let old = stored(&f, "own", "own").await;
        assert_eq!(old.fact.value, Some(json!("first <script>")));
        assert_eq!(old.fact.author, polylogue_model::Author::Operator);
        assert_eq!(old.fact.via, polylogue_model::authorship::Channel::Web);
        agent_assert(&f, "agent", "proposal");
        for (id, key, reason) in [
            ("accept", "accepted", "checked"),
            ("reject", "rejected", "reconsidered"),
        ] {
            assert_one(
                &f,
                id,
                statement(
                    "judgment",
                    Some(key),
                    json!(reason),
                    &["assertion:agent@agent".into()],
                ),
            )
            .await;
        }
        // Judging the revision again revises the operator's judgment of it.
        let current = query(
            &f,
            "assertions where kind:judgment about:\"assertion:agent@agent\"",
        )
        .await;
        assert_eq!(current["count"], 1, "{current}");
        assert_eq!(current["rows"][0]["id"], "reject", "{current}");
        assert_eq!(
            query(
                &f,
                "assertions where kind:judgment current:* about:\"assertion:agent@agent\""
            )
            .await["count"],
            2
        );
        agent_write(
            &f,
            crate::test_daemon::Call::new("revise", serde_json::json!({"bases": ["assertion:agent@agent"], "patch": {"value":{"change":"set","value":"new proposal"}}})).id("agent-new"),
        );
        let pending = query(&f, "assertions where agent:true judged:unknown").await;
        assert_eq!(pending["count"], 1);
        assert_eq!(pending["rows"][0]["id"], "agent-new");
        let retry = assert_one(
            &f,
            "reject",
            statement(
                "judgment",
                Some("rejected"),
                json!("reconsidered"),
                &["assertion:agent@agent".into()],
            ),
        )
        .await;
        assert_eq!(retry["repeat"], true);
        assert_eq!(
            query(
                &f,
                "assertions where kind:judgment current:* about:\"assertion:agent@agent\""
            )
            .await["count"],
            2
        );
        command(
            &f,
            "withdraw",
            "withdraw-agent",
            json!({"targets":["assertion:agent@agent-new"],"reason":"closed"}),
        )
        .await;
        assert_eq!(
            query(&f, "assertions where agent:true judged:unknown").await["count"],
            0
        );
        let revision = stored(&f, "agent", "agent-new").await;
        assert_eq!(revision.fact.value, Some(json!("new proposal")));
        assert!(revision.fact.author.is_agent());
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
                "keep",
                "kept",
                json!({"query":"assertions where kind:question"})
            )
            .await["id"],
            "kept"
        );
        assert_eq!(
            query(&f, "assertions where kind:question").await["count"],
            0
        );
        assert_one(
            &f,
            "question",
            statement(
                "question",
                None,
                json!("new member"),
                &["repo:morrow".into()],
            ),
        )
        .await;
        let result=command(&f,"assert-query","bulk",json!({"query":"assertions where kind:question","assertion":statement("note",None,json!("all when applied"),&[])})).await;
        assert_eq!(result["targets"], 1);
        assert_eq!(
            about(&stored(&f, "bulk", "bulk").await),
            vec!["assertion:question@question"]
        );
        assert_one(
            &f,
            "selected",
            statement(
                "note",
                None,
                json!("selected only"),
                &[format!("message:{first_id}")],
            ),
        )
        .await;
        assert_eq!(
            query(&f, "assertions where about:\"assertion:question@question\"").await["count"],
            1
        );
        assert_eq!(query(&f, "result:kept").await["count"], 0);
        command(
            &f,
            "keep",
            "kept-questions",
            json!({"query":"assertions where kind:question"}),
        )
        .await;
        assert_one(
            &f,
            "another",
            statement(
                "question",
                None,
                json!("later member"),
                &["repo:morrow".into()],
            ),
        )
        .await;
        let args = json!({"query":"result:kept-questions","assertion":statement("note",None,json!("fixed reviewed population"),&[])});
        assert_eq!(
            command(&f, "assert-query", "fixed", args.clone()).await["targets"],
            1
        );
        assert_eq!(
            command(&f, "assert-query", "fixed", args).await["repeat"],
            true
        );
        assert_eq!(
            about(&stored(&f, "fixed", "fixed").await),
            vec!["assertion:question@question"]
        );
    });
}

#[test]
fn editing_keyed_assertions_preserves_relations_and_expiry() {
    run(async {
        let f = fixture();
        assert_one(&f, "other", note("independent evidence")).await;
        let mut content = statement(
            "finding",
            Some("result"),
            json!("original"),
            &["repo:morrow".into()],
        );
        content["valid_until"] = json!("2099-01-01T00:00:00Z");
        content["links"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role":"derived_from","reference":"assertion:other@other"}));
        assert_one(&f, "keyed", content).await;
        revise(&f, "keyed-edit", "keyed", "keyed", json!("edited")).await;
        let row = stored(&f, "keyed", "keyed-edit").await;
        assert_eq!(row.fact.key.as_deref(), Some("result"));
        assert_eq!(row.fact.root, "keyed");
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
                    && link.reference.to_string() == "assertion:other@other")
        );
        assert!(
            row.fact
                .links
                .iter()
                .any(|link| link.role == LinkRole::Revises
                    && link.reference.to_string() == "assertion:keyed@keyed")
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
            command(&f, "setting", key, json!({"key":key,"value":value})).await;
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
                .put("local/notes-test/2/document/unit", &vectors)
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
                "assert-query",
                &id,
                json!({"query":text,"assertion":statement("note",None,json!(annotation),&[])}),
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
            let recorded = reader.user_request(&id).unwrap().unwrap();
            let polylogue_model::user::UserWrite::AssertQuery { resolved, .. } = recorded.write
            else {
                panic!("canonical bulk intent must retain its frozen population")
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
                    .visit_assertion_references_linked(
                        &reference,
                        &[polylogue_model::assertion::LinkRole::About],
                        |revision| {
                            if let Some(assertion) = reader.assertion(&revision)?
                                && assertion.fact.value == Some(json!(annotation))
                            {
                                matching.push(assertion.fact.id);
                            }
                            Ok(std::ops::ControlFlow::Continue(()))
                        },
                    )
                    .unwrap();
                assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
                assert_eq!(
                    matching.len(),
                    1,
                    "every committed population member gets exactly one new assertion in this run"
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
        assert_one(
            &f,
            "question",
            statement("question", None, json!("open"), &["repo:morrow".into()]),
        )
        .await;
        let args = json!({"query":"assertions where kind:question","assertion":statement("note",None,json!("reviewed"),&[])});
        assert_eq!(
            command(&f, "assert-query", "bulk-retry", args.clone()).await["targets"],
            1
        );
        command(
            &f,
            "withdraw",
            "closed",
            json!({"targets":["assertion:question@question"]}),
        )
        .await;
        assert_eq!(
            query(&f, "assertions where kind:question").await["count"],
            0
        );
        let repeated = command(&f, "assert-query", "bulk-retry", args.clone()).await;
        assert_eq!(repeated["repeat"], true);
        assert_eq!(repeated["targets"], 1);
        assert_eq!(
            about(&stored(&f, "bulk-retry", "bulk-retry").await),
            vec!["assertion:question@question"]
        );
        for field in ["query", "assertion"] {
            let mut changed = args.clone();
            changed[field] = if field == "query" {
                json!("sessions")
            } else {
                statement("note", None, json!("different"), &[])
            };
            assert_eq!(
                send(&f, "assert-query", "bulk-retry", changed).await.0,
                StatusCode::CONFLICT
            );
        }
    });
}

#[test]
fn assertion_projection_and_saved_query_edits_preserve_exact_metadata() {
    run(async {
        let f = fixture();
        command(
            &f,
            "save",
            "Saved:Case",
            json!({"name":"morrow","query":"messages"}),
        )
        .await;
        command(
            &f,
            "save",
            "Saved:Case:new",
            json!({"name":"morrow","query":"sessions"}),
        )
        .await;
        let page=command(&f,"query","projection",json!({"query":"assertions where root = \"Saved:Case\" current:*","fields":["id","key","about","judged"]})).await;
        assert_eq!(page["count"], 2);
        assert_eq!(
            query(&f, "assertions where root = \"saved:case\" current:*").await["count"],
            0
        );
        let row = stored(&f, "Saved:Case", "Saved:Case:new").await;
        assert_eq!(row.fact.key.as_deref(), Some("morrow"));
        assert_eq!(row.fact.root, "Saved:Case");
        assert_eq!(row.fact.value.as_ref().unwrap()["text"], "sessions");
        assert!(
            row.fact
                .links
                .iter()
                .any(|link| link.role == LinkRole::Revises
                    && link.reference.to_string() == "assertion:Saved:Case@Saved:Case")
        );
        for field in ["links", "valid_until"] {
            let mut invalid = note("invalid");
            invalid[field] = json!(if field == "links" {
                "invalid"
            } else {
                "tomorrow"
            });
            assert_eq!(
                send(&f, "assert", "invalid", json!({"assertions":[invalid]}))
                    .await
                    .0,
                StatusCode::BAD_REQUEST
            );
        }
    });
}

#[test]
fn grouped_assertion_population_returns_a_typed_refusal() {
    run(async {
        let f = fixture();
        let (status,answer)=send(&f,"assert-query","grouped",json!({"query":"sessions | group by origin","assertion":statement("note",None,json!("review"),&[])})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = answer
        else {
            panic!("group rows cannot name assertion targets")
        };
        assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
        assert_eq!(error.code.as_deref(), Some("bad_request"));
    });
}
