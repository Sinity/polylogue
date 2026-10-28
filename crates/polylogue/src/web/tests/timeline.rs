use super::*;

#[test]
fn timeline_and_cost_share_public_operations_and_access() {
    let r = reader();
    rt().block_on(async {
        for path in [
            "/api/v1/timeline?date=2024-02-29",
            "/api/v1/cost?date=2024-02-29",
        ] {
            assert_eq!(
                send(&r, get(path).body(Body::empty()).unwrap()).await.0,
                StatusCode::UNAUTHORIZED
            );
            let (status, headers, body) =
                send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        }
        for path in [
            "/api/v1/timeline?date=2023-02-29",
            "/api/v1/cost?date=2023-02-29",
        ] {
            let (status, _, body) = send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        }
        let (status, _, body) = send(
            &r,
            authed(get("/api/v1/timeline?date=2024-02-29"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let data: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(data["calendar"].as_array().unwrap().len(), 29);
        let (status, _, body) = send(
            &r,
            authed(get("/api/v1/cost?query=id%3Aclaude-code%3As1&date=&model="))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let data: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(data["usage"]["data"].is_object(), "{data}");
    });
}

#[test]
fn invalid_cost_scope_returns_a_query_error() {
    let r = reader();
    rt().block_on(async {
        let (status, _, body) = send(
            &r,
            authed(get("/api/v1/cost?query=messages"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let data: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(data["error"], "invalid_query");
    });
}
