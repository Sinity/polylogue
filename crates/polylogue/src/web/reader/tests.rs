use super::*;
use tower::ServiceExt;

/// An object URL already names the selection and its read step; a query
/// value that would name them again is refused.
#[test]
fn a_path_target_cannot_be_overridden_by_query_parameters() {
    let daemon = crate::test_daemon::start("reader route refusal");
    let web = Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    });
    let response = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            super::super::router(web)
                .oneshot(
                    axum::http::Request::get("/s/test:session?target=another%20selection")
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
        });
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
