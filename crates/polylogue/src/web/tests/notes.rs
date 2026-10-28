use super::*;
use axum::body::{Body, to_bytes};
use axum::http::Request as Http;
use polylogue_render::html::encode_path_segment as encode;
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

async fn send(f: &Fixture, path: &str, body: Option<Value>) -> (StatusCode, String) {
    let request = if let Some(body) = body {
        Http::post(path)
            .header(header::ORIGIN, "http://127.0.0.1:8765")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
    } else {
        Http::get(path).body(Body::empty())
    }
    .unwrap();
    let (mut parts, body) = request.into_parts();
    parts
        .headers
        .insert(header::HOST, HeaderValue::from_static("127.0.0.1:8765"));
    parts.headers.insert(
        header::COOKIE,
        HeaderValue::from_static("polylogue_notes=token"),
    );
    let response = router(Arc::clone(&f.web))
        .oneshot(Http::from_parts(parts, body))
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

async fn command(f: &Fixture, path: &str, body: Value) -> Value {
    let (status, text) = send(f, &format!("/api/v1/{path}"), Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    serde_json::from_str(&text).unwrap()
}

async fn query(f: &Fixture, text: &str) -> Value {
    let (status, body) = send(f, &format!("/api/v1/query?query={}", encode(text)), None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap()
}

#[test]
fn operator_edits_and_agent_judgments_keep_exact_revisions() {
    run(async {
        let f = fixture();
        command(
            &f,
            "note/add",
            json!({"request_id":"own", "targets":["repo:morrow"],"body":"first <script>"}),
        )
        .await;
        command(&f, "note/add", json!({"request_id":"edit", "revises":"own", "targets":["repo:morrow"],"body":"second"})).await;
        let (_, old) = send(&f, "/api/v1/note/read?id=own", None).await;
        assert_eq!(
            serde_json::from_str::<Value>(&old).unwrap()["value"],
            "first <script>"
        );
        command(&f, "note/add", json!({"request_id":"agent", "targets":["repo:morrow"],"body":"proposal", "agent":true})).await;
        command(
            &f,
            "note/accept",
            json!({"request_id":"accept", "note":"agent", "reason":"checked"}),
        )
        .await;
        command(
            &f,
            "note/reject",
            json!({"request_id":"reject", "note":"agent", "reason":"reconsidered"}),
        )
        .await;
        let judgments = query(&f, "notes where kind:judgment target:\"note:agent\"").await;
        assert_eq!(judgments["count"], 2);
        command(&f, "note/add", json!({"request_id":"agent-new", "revises":"agent", "targets":["repo:morrow"],"body":"new proposal", "agent":true})).await;
        let pending = query(&f, "notes where agent:true judged:unknown").await;
        assert_eq!(pending["count"], 1);
        assert_eq!(pending["rows"][0]["id"], "agent-new");
        let retry = command(
            &f,
            "note/reject",
            json!({"request_id":"reject", "note":"agent", "reason":"reconsidered"}),
        )
        .await;
        assert_eq!(retry["repeat"], true);
        assert_eq!(
            query(
                &f,
                "notes where kind:judgment current:* target:\"note:agent\""
            )
            .await["count"],
            2
        );
        command(
            &f,
            "note/withdraw",
            json!({"request_id":"withdraw-agent", "note":"agent-new", "reason":"closed"}),
        )
        .await;
        assert_eq!(
            query(&f, "notes where agent:true judged:unknown").await["count"],
            0
        );
        let (status, revision) = send(&f, "/api/v1/note/read?id=agent-new", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            serde_json::from_str::<Value>(&revision).unwrap()["value"],
            "new proposal"
        );
        drop(f.daemon);
    });
}

#[test]
fn selected_current_and_kept_rows_have_distinct_populations() {
    run(async {
        let f = fixture();
        let all = query(&f, "messages").await;
        let first_id = all["rows"][0]["id"].as_str().unwrap();
        let kept = command(
            &f,
            "keep",
            json!({"request_id":"kept", "query":"notes where kind:question"}),
        )
        .await;
        assert_eq!(kept["id"], "kept");
        assert_eq!(query(&f, "notes where kind:question").await["count"], 0);
        command(&f, "note/add", json!({"request_id":"question", "targets":["repo:morrow"], "kind":"question", "body":"new member"})).await;
        let result = command(&f, "note/add", json!({"request_id":"bulk", "query":"notes where kind:question", "body":"all when applied"})).await;
        assert_eq!(result["targets"], 1);
        let (status, committed) = send(&f, "/api/v1/note/read?id=bulk", None).await;
        assert_eq!(status, StatusCode::OK);
        let committed: Value = serde_json::from_str(&committed).unwrap();
        assert_eq!(committed["targets"], json!(["note:question"]));
        command(&f, "note/add", json!({"request_id":"selected", "targets":[format!("message:{first_id}")], "body":"selected only"})).await;
        assert_eq!(
            query(&f, "notes where target:\"note:question\"").await["count"],
            1
        );
        assert_eq!(query(&f, "result:kept").await["count"], 0);
        command(
            &f,
            "keep",
            json!({"request_id":"kept-questions", "query":"notes where kind:question"}),
        )
        .await;
        command(&f, "note/add", json!({"request_id":"another", "targets":["repo:morrow"], "kind":"question", "body":"later member"})).await;
        let fixed = command(&f, "note/add", json!({"request_id":"fixed", "query":"result:kept-questions", "body":"fixed reviewed population"})).await;
        assert_eq!(fixed["targets"], 1);
        let retry = command(&f, "note/add", json!({"request_id":"fixed", "query":"result:kept-questions", "body":"fixed reviewed population"})).await;
        assert_eq!(retry["repeat"], true);
        drop(f.daemon);
    });
}

#[test]
fn editing_keyed_notes_preserves_relations_and_expiry() {
    run(async {
        let f = fixture();
        let content = json!({"request_id":"keyed", "targets":["repo:morrow"], "body":"original", "kind":"finding", "key":"result", "relations":"[[\"derived_from\",\"other\"]]", "valid_until":"2099-01-01T00:00:00Z"});
        command(&f, "note/add", content.clone()).await;
        let mut revision = content;
        revision["request_id"] = json!("keyed-edit");
        revision["revises"] = json!("keyed");
        revision["body"] = json!("edited");
        command(&f, "note/add", revision).await;
        let (status, stored) = send(&f, "/api/v1/note/read?id=keyed-edit", None).await;
        assert_eq!(status, StatusCode::OK);
        let stored: Value = serde_json::from_str(&stored).unwrap();
        assert_eq!(stored["key"], "result");
        assert_eq!(stored["first_note"], "keyed");
        assert_eq!(stored["relations"], json!([["derived_from", "other"]]));
        assert_eq!(stored["valid_until"], "2099-01-01T00:00:00Z");
        drop(f.daemon);
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
            let body = r#"{"data":[{"embedding":[1.0,0.0]}]}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        for (key, value) in [
            ("embed.provider", "local"),
            ("embed.model", "notes-test"),
            ("embed.endpoint", endpoint.as_str()),
        ] {
            command(
                &f,
                "setting",
                json!({"request_id":key,"key":key,"value":value}),
            )
            .await;
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
            "messages where near:synthetic origin:claude-code",
            "messages where synthetic near:synthetic origin:claude-code",
        ]
        .into_iter()
        .enumerate()
        {
            let expected = query(&f, text).await;
            assert_eq!(expected["ranked"], true);
            let applied = command(&f, "note/add", json!({"request_id":format!("semantic-{n}"),"query":text,"body":"reviewed semantic population"})).await;
            assert_eq!(applied["targets"], expected["count"]);
            let (_, stored) = send(&f, &format!("/api/v1/note/read?id=semantic-{n}"), None).await;
            let stored: Value = serde_json::from_str(&stored).unwrap();
            assert_eq!(
                stored["targets"][0],
                format!("message:{}", expected["rows"][0]["id"].as_str().unwrap())
            );
        }
        provider.join().unwrap();
        drop(f.daemon);
    });
}

#[test]
fn bulk_retry_uses_committed_targets_after_the_query_becomes_empty() {
    run(async {
        let f = fixture();
        command(&f, "note/add", json!({"request_id":"question", "targets":["repo:morrow"],"kind":"question", "body":"open"})).await;
        let request = json!({"request_id":"bulk-retry", "query":"notes where kind:question", "body":"reviewed"});
        let applied = command(&f, "note/add", request.clone()).await;
        assert_eq!(applied["targets"], 1);
        command(
            &f,
            "note/withdraw",
            json!({"request_id":"closed", "note":"question"}),
        )
        .await;
        assert_eq!(query(&f, "notes where kind:question").await["count"], 0);
        let repeated = command(&f, "note/add", request.clone()).await;
        assert_eq!(repeated["repeat"], true);
        assert_eq!(repeated["targets"], 1);
        for field in ["query", "body"] {
            let mut changed = request.clone();
            changed[field] = json!("different");
            let (status, _) = send(&f, "/api/v1/note/add", Some(changed)).await;
            assert_eq!(status, StatusCode::CONFLICT);
        }
    });
}

#[test]
fn note_projection_and_saved_query_edits_preserve_exact_metadata() {
    run(async {
        let f = fixture();
        command(
            &f,
            "save",
            json!({"request_id":"Saved:Case", "name":"morrow", "query":"messages"}),
        )
        .await;
        command(&f, "note/add", json!({"request_id":"Saved:Case:new", "kind":"saved_query", "key":"morrow", "body":"sessions", "revises":"Saved:Case"})).await;
        let text = "notes where first_note = \"Saved:Case\" current:*";
        let (status, body) = send(
            &f,
            &format!(
                "/api/v1/query?query={}&fields=id&fields=key&fields=target&fields=judged",
                encode(text)
            ),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let page: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(page["count"], 2);
        assert_eq!(
            query(&f, "notes where first_note = \"saved:case\" current:*").await["count"],
            0
        );
        let (status, body) = send(&f, "/api/v1/note/read?id=Saved%3ACase%3Anew", None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let note: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(note["key"], "morrow");
        assert_eq!(note["first_note"], "Saved:Case");
        assert_eq!(note["value"], "sessions");
        for (field, value) in [("relations", "invalid"), ("valid_until", "tomorrow")] {
            let mut body =
                json!({"request_id":"invalid", "targets":["repo:morrow"], "body":"invalid"});
            body[field] = json!(value);
            assert_eq!(
                send(&f, "/api/v1/note/add", Some(body)).await.0,
                StatusCode::BAD_REQUEST
            );
        }
    });
}

#[test]
fn grouped_annotation_returns_a_typed_refusal() {
    run(async {
        let f = fixture();
        let (status, body) = send(&f, "/api/v1/note/add", Some(json!({"request_id":"grouped", "query":"sessions | group by origin", "body":"review"}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["error"],
            "bad_request"
        );
    });
}
