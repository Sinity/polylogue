use super::*;

#[test]
fn http_attachment_descriptors_are_masked_and_reveal_is_cli_only() {
    let r = reader();
    let message = test_daemon::assets::seed(&r.w);
    rt().block_on(async {
        let path = format!("/api/v1/attachment?message={message}&block=0");
        let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let descriptor: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(descriptor["state"], "available");
        for secret in [
            "synthetic-secret-file-bytes",
            "synthetic-private-label",
            "content_base64",
        ] {
            assert!(!body.contains(secret), "{body}");
        }
        let (status, _, body) = send(
            &r,
            authed(get(&format!("{path}&reveal=true&out=synthetic.bin")))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    });
}
