use super::*;
use polylogue_engine::Engine;
use polylogue_model::SessionId;
use polylogue_store::{Archive, ReadPool};

fn text(messages: &[polylogue_model::Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| m.blocks.iter().filter_map(|b| b.text()).collect::<String>())
        .collect()
}

#[test]
fn rapid_captures_show_the_latest_and_keep_retry_identity() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_at(dir.path());
    std::fs::create_dir_all(&config.home).unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let host = Arc::new(BrowserHost::at(&config, &archive).unwrap());
    let first = capture("rapid", "first earlier version");
    let second = capture("rapid", "second current version");
    let ack = exchange(&host, "POST", "/v1/captures", &first);
    assert_eq!(ack.status, 200);
    assert_eq!(exchange(&host, "POST", "/v1/captures", &second).status, 200);
    assert_eq!(ack.json()["new"], true);
    let path = host.capture_log("chatgpt", "rapid");
    let before = std::fs::read(&path).unwrap();
    assert_eq!(
        exchange(&host, "POST", "/v1/captures", &first).json()["new"],
        false
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let sid = SessionId::parse("chatgpt:rapid").unwrap();
    let mut engine = Engine::new(config.clone(), archive.clone()).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    {
        let pool = ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            text(&reader.messages(&sid, false).unwrap()),
            ["second current version"]
        );
    }
    drop(engine);
    let mut engine = Engine::new(config, archive.clone()).unwrap();
    let pass = engine.run_pass().unwrap();
    assert_eq!(pass.bytes_read, 0, "restart must use durable progress");
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = ReadPool::open(&archive).unwrap();
    assert_eq!(
        text(&pool.reader().unwrap().messages(&sid, false).unwrap()),
        ["second current version"]
    );
    drop(pool);
    // A new observation can revert to earlier content. Its provenance makes
    // it distinct from a resend of the older capture.
    let mut reversion: Value = serde_json::from_slice(&first).unwrap();
    reversion["provenance"]["captured_at"] = json!("2026-03-01T11:00:00Z");
    let reversion = serde_json::to_vec(&reversion).unwrap();
    assert_eq!(
        exchange(&host, "POST", "/v1/captures", &reversion).status,
        200
    );
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(
        pass.bytes_read,
        reversion.len() as u64 + polylogue_model::capture_log::HEADER_LEN as u64,
        "append reads only new frame bytes"
    );
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        text(&reader.messages(&sid, false).unwrap()),
        ["first earlier version"]
    );
}
