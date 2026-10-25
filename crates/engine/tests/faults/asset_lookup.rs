use super::*;
use polylogue_testkit::{ClaudeHome, delete_index};

/// After an index rebuild no message is visible before the asset lookup it
/// names: the lookup comes from stored rows, so the loop writes it before
/// it parses anything. A failure writing the lookup leaves no message.
#[test]
fn a_rebuilt_index_has_the_asset_lookup_before_any_message() {
    fault::clear();
    let w = ClaudeHome::new();
    let export = w.dir.path().join("export");
    fs::create_dir(&export).unwrap();
    let conversation = serde_json::json!([{
        "id": "asset-session", "title": "Synthetic export attachment", "current_node": "node",
        "mapping": {"node": {"id": "node", "parent": null, "children": [], "message": {
            "id": "asset-message", "author": {"role": "user"}, "create_time": 1_700_000_000,
            "content": {"content_type": "multimodal_text", "parts": [
                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-Native"}]},
            "metadata": {}}}}}]);
    fs::write(export.join("conversations.json"), conversation.to_string()).unwrap();
    fs::write(export.join("file-Native.dat"), b"synthetic asset bytes").unwrap();
    fs::write(
        export.join("conversation_asset_file_names.json"),
        br#"{"file-Native.dat":"named.png"}"#,
    )
    .unwrap();
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let report = engine.import(std::slice::from_ref(&export)).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    }
    let count = |sql: &str| -> i64 {
        rusqlite::Connection::open(w.root.join("index.db"))
            .unwrap()
            .query_row(sql, [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(count("SELECT count(*) FROM message"), 1);
    assert_eq!(count("SELECT count(*) FROM import_asset_metadata"), 1);
    delete_index(&w.root);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        fault::arm("assets.install", "", Action::Fail, 0, 1);
        assert!(engine.converge().is_err());
        fault::clear();
    }
    assert_eq!(
        count("SELECT count(*) FROM message"),
        0,
        "a message was parsed before its asset lookup was written"
    );
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(count("SELECT count(*) FROM message"), 1);
    assert_eq!(count("SELECT count(*) FROM import_asset_metadata"), 1);
}
