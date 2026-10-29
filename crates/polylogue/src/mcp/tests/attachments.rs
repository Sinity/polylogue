use super::*;
use base64::Engine as _;

#[test]
fn mcp_attachment_pages_preserve_credential_shaped_bytes_and_labels() {
    let daemon = test_daemon::start("synthetic transport fixture");
    let message = test_daemon::assets::seed(&daemon);
    let archive = Archive::at(&daemon.daemon.archive.root);
    let mut bytes = Vec::new();
    let mut cursor = None::<String>;
    loop {
        let response = call(
            &archive,
            "attachment",
            args(json!({
                "message":message,"block":0,"length":7,"cursor":cursor
            })),
        )
        .unwrap();
        let page: serde_json::Value = serde_json::from_str(&response).unwrap();
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
}
