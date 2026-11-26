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
        let mut step = json!({"verb":"read","budget":3});
        if let Some(cursor) = &cursor {
            step["cursor"] = json!(cursor);
        }
        let answer = structured_call_as(
            &archive,
            "rows",
            json!({"selection":format!("attachment:{message}#0"),"steps":[step]})
                .as_object()
                .unwrap()
                .clone(),
            &uuid::Uuid::now_v7().to_string(),
        );
        let page = serde_json::to_value(crate::protocol::line::attachment_page(
            answer.value().expect("an attachment page"),
        ))
        .unwrap();
        assert_eq!(page["offset"], bytes.len());
        if bytes.is_empty() {
            assert_eq!(page["descriptor"]["state"], "available");
            assert!(
                page["descriptor"]
                    .to_string()
                    .contains("synthetic-private-label")
            );
        }
        let part = base64::engine::general_purpose::STANDARD
            .decode(page["content_base64"].as_str().unwrap())
            .unwrap();
        assert!(part.len() <= 9, "a budget of 3 tokens holds 9 bytes");
        bytes.extend(part);
        cursor = page["next"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(bytes, test_daemon::assets::PAYLOAD);
}
