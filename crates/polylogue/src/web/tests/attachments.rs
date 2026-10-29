use super::*;
use base64::Engine as _;

#[test]
fn http_attachment_pages_preserve_credential_shaped_bytes_and_labels() {
    let r = reader();
    let message = test_daemon::assets::seed(&r.w);
    rt().block_on(async {
        let mut bytes = Vec::new();
        let mut cursor = None::<String>;
        loop {
            let mut path = format!("/api/v1/attachment?message={message}&block=0&length=7");
            if let Some(cursor) = &cursor {
                path.push_str("&cursor=");
                path.push_str(cursor);
            }
            let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let page: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(page["offset"], bytes.len());
            if bytes.is_empty() {
                assert_eq!(page["descriptor"]["state"], "available");
                assert!(
                    page["descriptor"]
                        .to_string()
                        .contains("synthetic-private-label")
                );
            }
            bytes.extend(
                base64::engine::general_purpose::STANDARD
                    .decode(page["content_base64"].as_str().unwrap())
                    .unwrap(),
            );
            cursor = page["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(bytes, test_daemon::assets::PAYLOAD);
    });
}
