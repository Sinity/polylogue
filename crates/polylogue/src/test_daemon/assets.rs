//! Shared synthetic source for the real MCP and HTTP attachment operations.
use super::*;

pub fn seed(daemon: &TestDaemon) -> &'static str {
    let source = daemon.home.join("asset-export");
    std::fs::create_dir_all(&source).unwrap();
    let conversation = serde_json::json!([{"id":"transport-session","title":"Synthetic transport attachment","mapping":{"n":{"id":"n","parent":null,"children":[],"message":{"id":"transport-message","author":{"role":"user"},"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Transport"}]},"metadata":{}}}},"current_node":"n"}]);
    std::fs::write(
        source.join("conversations.json"),
        serde_json::to_vec(&conversation).unwrap(),
    )
    .unwrap();
    std::fs::write(
        source.join("file-Transport.dat"),
        b"synthetic-secret-file-bytes",
    )
    .unwrap();
    let names = serde_json::json!({"file-Transport.dat":"-----BEGIN PRIVATE KEY-----synthetic-private-label-----END PRIVATE KEY-----"});
    std::fs::write(
        source.join("conversation_asset_file_names.json"),
        serde_json::to_vec(&names).unwrap(),
    )
    .unwrap();
    let imported = handle(
        Request::Import {
            request_id: "synthetic-attachment-import".into(),
            paths: vec![source.clone()],
        },
        &daemon.daemon,
    );
    let Response::Ok { data } = imported else {
        panic!("{imported:?}")
    };
    assert_eq!(data["pass"]["errors"], serde_json::json!([]));
    std::fs::remove_dir_all(source).unwrap();
    "chatgpt:transport-message"
}
