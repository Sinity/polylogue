use super::*;

#[test]
fn timeline_day_and_cost_share_public_operations_and_access() {
    let r = reader();
    rt().block_on(async {
        for path in ["/timeline", "/day/2024-02-29", "/cost?date=2024-02-29"] {
            assert_eq!(
                send(&r, get(path).body(Body::empty()).unwrap()).await.0,
                StatusCode::SEE_OTHER
            );
            let (status, headers, body) =
                send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        }
        for path in ["/day/2023-02-29", "/cost?date=2023-02-29"] {
            assert_eq!(
                send(&r, authed(get(path)).body(Body::empty()).unwrap())
                    .await
                    .0,
                StatusCode::BAD_REQUEST
            );
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
        let (status, _, html) = send(
            &r,
            authed(get("/day/2024-02-29")).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(html.contains("Day cost and response details"));
        let (status, _, body) = send(
            &r,
            authed(get("/cost?q=id%3Aclaude-code%3As1&date=&model="))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("Cost of selected sessions"));
    });
}
