use super::*;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn response(
    daemon: &crate::test_daemon::TestDaemon,
    method: axum::http::Method,
    uri: &str,
    body: Value,
) -> (StatusCode, command_surface::Answer<Value>) {
    let response = super::super::super::router(web(daemon))
        .oneshot(
            axum::http::Request::builder()
                .method(method)
                .uri(uri)
                .header(header::HOST, "127.0.0.1:8765")
                .header(header::AUTHORIZATION, "Bearer synthetic")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[test]
fn http_post_rows_derives_request_id() {
    let daemon = crate::test_daemon::start("HTTP derived write request");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let body = json!({"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"synthetic HTTP note"}]});
        let mut id = None;
        for repeat in [false, true] {
            let (status, answer) = response(&daemon, axum::http::Method::POST, "/api/v1/rows", body.clone()).await;
            assert_eq!(status, StatusCode::OK, "{answer:?}");
            let value = &answer.value().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["reply"];
            assert_eq!(value["repeat"], repeat, "{answer:?}");
            if let Some(id) = &id {
                assert_eq!(&value["id"], id);
            } else {
                id = Some(value["id"].clone());
            }
        }
        let (status, listed) = response(&daemon, axum::http::Method::GET, "/api/v1/rows?selection=assertions%20where%20kind%3Anote", Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{listed:?}");
        assert_eq!(listed.value().unwrap()["reply"]["rows"].as_array().unwrap().len(), 1);
    });
}

#[test]
fn http_get_rows_refuses_write_step() {
    let daemon = crate::test_daemon::start("HTTP read step access");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for query in [
            "selection=sessions&steps=note&text=synthetic",
            "selection=sessions&steps=read&steps=brief",
            "arguments=%7B%22selection%22%3A%22sessions%22%2C%22steps%22%3A%5B%7B%22verb%22%3A%22note%22%2C%22text%22%3A%22synthetic%22%7D%5D%7D",
        ] {
            let (status, answer) = response(&daemon, axum::http::Method::GET, &format!("/api/v1/rows?{query}"), Value::Null).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
            let command_surface::Outcome::Error(error) = answer.outcome else { panic!("GET write must refuse") };
            assert_eq!(error.message, "use POST /api/v1/rows");
        }
        let (status, answer) = response(&daemon, axum::http::Method::GET, "/api/v1/rows?selection=session%3Aclaude-code%3As1&steps=read&view=dialogue&around=1", Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{answer:?}");
        assert_eq!(answer.value().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["kind"], "read");
    });
}

#[test]
fn http_calling_session_resolves_self() {
    let daemon = crate::test_daemon::start("HTTP calling session");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for body in [
            json!({"session":"claude-code:s1","selection":"self"}),
            json!({"session":"claude-code:s1","operation":"rows","arguments":{"selection":"self"}}),
        ] {
            let (status, answer) =
                response(&daemon, axum::http::Method::POST, "/api/v1/rows", body).await;
            assert_eq!(status, StatusCode::OK, "{answer:?}");
            assert_eq!(
                answer.value().unwrap()["reply"]["rows"][0]["id"],
                "claude-code:s1"
            );
        }
    });
}

#[test]
fn http_failed_step_sets_status_and_keeps_prior_results() {
    let daemon = crate::test_daemon::start("HTTP partial line");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let (status, answer) = response(&daemon, axum::http::Method::POST, "/api/v1/rows", json!({
            "selection":"session:claude-code:s1",
            "steps":[{"verb":"note","text":"synthetic completed note"},{"verb":"read","budget":0}]
        })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
        let steps = &answer.value().unwrap()["reply"]["steps"];
        assert_eq!(steps[0]["outcome"]["status"], "done");
        assert_eq!(steps[0]["outcome"]["reply"]["kind"], "wrote_each");
        assert_eq!(steps[1]["outcome"]["status"], "failed");
    });
}

#[test]
fn http_get_cursor_continues_read() {
    use polylogue_model::{
        BlobHash,
        read::{ItemKind, ReadItem, SelectionLocator},
    };
    let text = "Flat HTTP λ🙂 cursor. ".repeat(300);
    let daemon = crate::test_daemon::start(&text);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let base = "/api/v1/rows?selection=session%3Aclaude-code%3As1&steps=read&budget=128";
        let mut uri = base.to_owned();
        let mut partial =
            std::collections::BTreeMap::<(SelectionLocator, u64, BlobHash), (String, u64)>::new();
        let mut items = Vec::<ReadItem>::new();
        let mut cursors = std::collections::BTreeSet::new();
        loop {
            let (status, answer) =
                response(&daemon, axum::http::Method::GET, &uri, Value::Null).await;
            assert_eq!(status, StatusCode::OK, "{answer:?}");
            let crate::protocol::line::LineAnswer::Steps(mut steps) =
                serde_json::from_value(answer.value().unwrap().clone()).unwrap()
            else {
                panic!("read must return step replies")
            };
            let crate::protocol::line::StepOutcome::Done {
                reply: crate::protocol::line::StepAnswer::Read(page),
                ..
            } = steps.steps.remove(0).outcome
            else {
                panic!("read step must finish")
            };
            assert!(!page.document.items.is_empty(), "a page must progress");
            for item in page.document.items {
                if let ItemKind::Fragment {
                    text,
                    position,
                    until,
                    total,
                } = &item.kind
                {
                    let key = (position.selection, position.item, position.item_hash);
                    let (encoded, characters) = partial.entry(key).or_default();
                    assert_eq!(
                        position.character, *characters,
                        "fragment gap or duplicate prefix"
                    );
                    assert!(*until > *characters && *until <= *total);
                    assert_eq!(*until - *characters, text.chars().count() as u64);
                    encoded.push_str(text);
                    *characters = *until;
                    if until == total {
                        let (encoded, _) = partial.remove(&key).unwrap();
                        assert_eq!(BlobHash::of(encoded.as_bytes()), position.item_hash);
                        let restored: ReadItem = serde_json::from_str(&encoded).unwrap();
                        assert!(!items.contains(&restored), "duplicate reconstructed item");
                        items.push(restored);
                    }
                } else {
                    assert!(!items.contains(&item), "duplicate item");
                    items.push(item);
                }
            }
            let Some(next) = page.document.footer.next else {
                break;
            };
            assert_eq!(next.request.operation, "rows");
            let cursor = next.request.arguments["steps"][0]["cursor"]
                .as_str()
                .expect("read continuation carries its step cursor");
            assert!(cursors.insert(cursor.to_owned()), "repeated continuation");
            uri = format!(
                "{base}&cursor={}",
                polylogue_render::html::encode_path_segment(cursor)
            );
        }
        assert!(!cursors.is_empty(), "the read must span pages");
        assert!(partial.is_empty(), "the read ended within an item");
        let actual: Vec<_> = items
            .into_iter()
            .filter_map(|item| match item.kind {
                ItemKind::Text { text } => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(actual, vec![text]);
    });
}

#[test]
fn http_html_renders_listing_brief_and_partial_steps() {
    let daemon = crate::test_daemon::start("Synthetic brief <script>λ🙂</script>");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for (body, expected_status, expected) in [
            (
                json!({"selection":"sessions","fields":["id"]}),
                StatusCode::OK,
                vec!["claude-code:s1"],
            ),
            (
                json!({"selection":"session:claude-code:s1","steps":[{"verb":"brief"}]}),
                StatusCode::OK,
                vec![r"Synthetic brief \u003cscript\u003eλ🙂\u003c/script\u003e"],
            ),
            (
                json!({"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"Synthetic completed write"},{"verb":"read","budget":0}]}),
                StatusCode::BAD_REQUEST,
                vec!["wrote_each", "undo:", "read failed:"],
            ),
        ] {
            let response = super::super::super::router(web(&daemon))
                .oneshot(
                    axum::http::Request::post("/api/v1/rows")
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .header(header::CONTENT_TYPE, "application/json")
                        .header(header::ACCEPT, "text/html")
                        .body(axum::body::Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            assert_eq!(response.headers()[header::CONTENT_TYPE], "text/html; charset=utf-8");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let html = String::from_utf8(bytes.to_vec()).unwrap();
            assert_eq!(status, expected_status, "{html}");
            for expected in expected {
                assert!(html.contains(expected), "missing {expected:?}: {html}");
            }
        }
    });
}
