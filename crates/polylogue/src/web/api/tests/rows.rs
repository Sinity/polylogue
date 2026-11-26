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
        let (status, listed) = response(&daemon, axum::http::Method::GET, "/api/v1/rows?selection=annotations%20where%20kind%3Anote", Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{listed:?}");
        assert_eq!(listed.value().unwrap()["reply"]["rows"].as_array().unwrap().len(), 1);
    });
}

#[test]
fn http_get_rows_refuses_write_step() {
    let daemon = crate::test_daemon::start("HTTP read step access");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for (query, message) in [
            (
                "selection=sessions&steps=note&text=synthetic",
                "a GET carries only the read and brief steps, not note; use POST /api/v1/rows",
            ),
            (
                "selection=sessions&steps=read&steps=brief",
                "a GET carries one step; use POST /api/v1/rows for several",
            ),
            (
                "arguments=%7B%22selection%22%3A%22sessions%22%2C%22steps%22%3A%5B%7B%22verb%22%3A%22note%22%2C%22text%22%3A%22synthetic%22%7D%5D%7D",
                "use POST /api/v1/rows",
            ),
        ] {
            let (status, answer) = response(&daemon, axum::http::Method::GET, &format!("/api/v1/rows?{query}"), Value::Null).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
            let command_surface::Outcome::Error(error) = answer.outcome else { panic!("GET write must refuse") };
            assert_eq!(error.message, message);
        }
        let (status, answer) = response(&daemon, axum::http::Method::GET, "/api/v1/rows?selection=session%3Aclaude-code%3As1&steps=read&view=dialogue&around=1", Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{answer:?}");
        assert_eq!(answer.value().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["kind"], "read");
    });
}

#[test]
fn http_get_refuses_to_change_a_setting() {
    let daemon = crate::test_daemon::start("HTTP setting by GET");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let (status, answer) = response(
            &daemon,
            axum::http::Method::GET,
            "/api/v1/set?key=embed.calls&value=on",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
        let command_surface::Outcome::Error(error) = answer.outcome else {
            panic!("a GET must not write a setting")
        };
        assert_eq!(error.message, "use POST /api/v1/set");
    });
}

#[test]
fn http_post_accepts_a_body_larger_than_the_default_body_limit() {
    let daemon = crate::test_daemon::start("HTTP large body");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        // Axum's own default limit is 2 MiB; stored text has no size cap.
        let text = "x".repeat(3 << 20);
        let body =
            json!({"selection":"session:claude-code:s1","steps":[{"verb":"note","text":text}]});
        let (status, answer) =
            response(&daemon, axum::http::Method::POST, "/api/v1/rows", body).await;
        assert_eq!(status, StatusCode::OK, "{answer:?}");
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
            "steps":[{"verb":"note","text":"synthetic completed note"},{"verb":"read","show":["not-a-show-kind"]}]
        })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer:?}");
        let steps = &answer.value().unwrap()["reply"]["steps"];
        assert_eq!(steps[0]["outcome"]["status"], "done");
        assert_eq!(steps[0]["outcome"]["reply"]["kind"], "wrote_each");
        assert_eq!(steps[1]["outcome"]["status"], "failed");
        assert_eq!(steps[0]["outcome"]["undo"].as_array().unwrap().len(), 1);
        assert_eq!(steps[1]["outcome"]["error"]["kind"], "bad_request");
        assert!(steps[1]["outcome"]["error"]["message"].as_str().unwrap().contains("not-a-show-kind"));
        let (status, stored) = response(&daemon, axum::http::Method::GET,
            "/api/v1/rows?selection=session%3Aclaude-code%3As1&steps=read", Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{stored:?}");
        assert!(stored.value().unwrap().to_string().contains("synthetic completed note"), "the prefix write remains stored: {stored:?}");
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

/// The reader route renders a line as HTML: a listing, a brief, and a
/// read that fails with its own status.
#[test]
fn reader_route_renders_listing_brief_and_a_failed_read() {
    let daemon = crate::test_daemon::start("Synthetic brief <script>λ🙂</script>");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for (uri, expected_status, expected) in [
            (
                "/rows?selection=sessions&fields=id",
                StatusCode::OK,
                vec!["claude-code:s1"],
            ),
            (
                "/rows?selection=session%3Aclaude-code%3As1&steps=brief",
                StatusCode::OK,
                vec![r"Synthetic brief \u003cscript\u003eλ🙂\u003c/script\u003e"],
            ),
            (
                "/rows?selection=session%3Aclaude-code%3As1&steps=read&show=not-a-show-kind",
                StatusCode::BAD_REQUEST,
                vec!["not-a-show-kind"],
            ),
        ] {
            let response = super::super::super::router(web(&daemon))
                .oneshot(
                    axum::http::Request::get(uri)
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            assert!(
                response.headers()[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html"),
                "{uri}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let html = String::from_utf8(bytes.to_vec()).unwrap();
            assert_eq!(status, expected_status, "{uri}: {html}");
            for expected in expected {
                assert!(html.contains(expected), "missing {expected:?}: {html}");
            }
        }
    });
}
#[test]
fn http_line_streams_reads_in_order_and_finishes_at_answer() {
    let daemon = crate::test_daemon::start("HTTP ordered stream");
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let body = json!({"operation":"rows", "request_id":"http-ordered", "deadline":null,
            "arguments":{"references":["session:claude-code:s1"], "steps":[
                {"verb":"note","text":"http-first"}, {"verb":"read","delivery":"stream"},
                {"verb":"tag","name":"http-later"}, {"verb":"read","delivery":"stream"}]}});
        let response = super::super::super::router(web(&daemon))
            .oneshot(
                axum::http::Request::post("/api/v1/rows")
                    .header(header::HOST, "127.0.0.1:8765")
                    .header(header::AUTHORIZATION, "Bearer synthetic")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let frames = command_surface::SseDecoder::new().push(&bytes).unwrap();
        assert_eq!(
            frames.iter().filter(|frame| frame.event == "begin").count(),
            2
        );
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.event == "complete")
                .count(),
            2
        );
        assert_eq!(frames.last().unwrap().event, "answer");
        let reports: Vec<Value> = frames
            .iter()
            .filter(|frame| frame.event == "steps" || frame.event == "answer")
            .flat_map(|frame| {
                let answer: command_surface::Answer<Value> =
                    serde_json::from_str(&frame.data).unwrap();
                answer.value().unwrap()["reply"]["steps"]
                    .as_array()
                    .unwrap()
                    .clone()
            })
            .collect();
        assert_eq!(
            reports
                .iter()
                .map(|report| report["request_id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "http-ordered-1",
                "http-ordered-2",
                "http-ordered-3",
                "http-ordered-4"
            ]
        );
        assert_eq!(
            reports
                .iter()
                .map(|report| report["verb"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["note", "read", "tag", "read"]
        );
    });
}
