use super::*;
use base64::Engine as _;
use serde_json::json;

#[test]
fn http_attachment_pages_preserve_credential_shaped_bytes_and_labels() {
    let r = reader();
    let message = test_daemon::assets::seed(&r.w);
    rt().block_on(async {
        let mut bytes = Vec::new();
        let mut cursor = None::<String>;
        loop {
            let mut step = json!({"verb":"read","budget":3});
            if let Some(cursor) = &cursor {
                step["cursor"] = json!(cursor);
            }
            let body = json!({
                "request_id": format!("attachment-{}", bytes.len()),
                "selection": format!("attachment:{message}#0"),
                "steps": [step],
            });
            let (status, _, raw) = send(
                &r,
                own_origin(authed(post("/api/v1/rows")))
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{raw}");
            // The test client hands back the one step's reply.
            let page: crate::daemon::attachment::BytePage = serde_json::from_str(&raw).unwrap();
            assert_eq!(page.offset, bytes.len() as u64);
            if bytes.is_empty() {
                let descriptor = serde_json::to_value(&page.descriptor).unwrap();
                assert_eq!(descriptor["state"], "available");
                assert!(descriptor.to_string().contains("synthetic-private-label"));
            }
            bytes.extend(
                base64::engine::general_purpose::STANDARD
                    .decode(&page.content_base64)
                    .unwrap(),
            );
            cursor = page.next;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(bytes, test_daemon::assets::PAYLOAD);
    });
}
