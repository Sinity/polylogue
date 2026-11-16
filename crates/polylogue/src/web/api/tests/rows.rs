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
        assert_eq!(steps[0]["outcome"]["reply"]["kind"], "wrote");
        assert_eq!(steps[1]["outcome"]["status"], "failed");
    });
}
