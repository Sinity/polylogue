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
    let ctx = Arc::new(Ctx::at(&config, &archive).unwrap());
    let first = capture("rapid", "first retained version");
    let second = capture("rapid", "second current version");
    let ack = exchange(&ctx, "POST", "/v1/browser-captures", &first);
    assert_eq!(ack.status, 200);
    assert_eq!(
        exchange(&ctx, "POST", "/v1/browser-captures", &second).status,
        200
    );
    let path = ctx.spool.join(ack.json()["artifact_ref"].as_str().unwrap());
    let before = std::fs::read(&path).unwrap();
    assert_eq!(
        exchange(&ctx, "POST", "/v1/browser-captures", &first).json()["outcome"],
        "noop"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let sid = SessionId::parse("chatgpt:rapid").unwrap();
    let mut engine = Engine::new(config.clone(), archive.clone()).unwrap();
    let pass = engine.converge().unwrap();
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
    let pass = engine.converge().unwrap();
    assert_eq!(pass.bytes_read, 0, "restart must use durable progress");
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = ReadPool::open(&archive).unwrap();
    assert_eq!(
        text(&pool.reader().unwrap().messages(&sid, false).unwrap()),
        ["second current version"]
    );
    drop(pool);
    // A new observation can revert to earlier content. Its provenance makes
    // it distinct from the older envelope's idempotent retry.
    let mut reversion: Value = serde_json::from_slice(&first).unwrap();
    reversion["provenance"]["captured_at"] = json!("2026-03-01T11:00:00Z");
    let reversion = serde_json::to_vec(&reversion).unwrap();
    assert_eq!(
        exchange(&ctx, "POST", "/v1/browser-captures", &reversion).status,
        200
    );
    let pass = engine.converge().unwrap();
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
        ["first retained version"]
    );
}

fn rebuild(config: polylogue_engine::Config, archive: &Archive) -> Engine {
    for path in [
        archive.index_db(),
        archive.index_db().with_extension("db-wal"),
        archive.index_db().with_extension("db-shm"),
    ] {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => panic!("{e}"),
        }
    }
    let (_archive, _lock) = Archive::open_or_create(&archive.root).unwrap();
    let mut engine = Engine::new(config, archive.clone()).unwrap();
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    engine
}

#[test]
fn text_excision_rewrites_every_capture_and_survives_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_at(dir.path());
    std::fs::create_dir_all(&config.home).unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let ctx = Arc::new(Ctx::at(&config, &archive).unwrap());
    for content in ["synthetic-private-token old", "safe current version"] {
        assert_eq!(
            exchange(
                &ctx,
                "POST",
                "/v1/browser-captures",
                &capture("redact", content)
            )
            .status,
            200
        );
    }
    let mut engine = Engine::new(config.clone(), archive.clone()).unwrap();
    assert!(engine.converge().unwrap().errors.is_empty());
    let done = engine
        .excise(polylogue_engine::ExciseRequest {
            id: "redact-history".into(),
            scope: polylogue_engine::ExciseScope::Text("synthetic-private-token".into()),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    assert!(!done.inputs_reparsed.is_empty());
    assert!(
        polylogue_store::excise::stored_blobs_containing(&archive, "synthetic-private-token")
            .unwrap()
            .is_empty()
    );
    assert!(
        polylogue_store::excise::index_columns_containing(&archive, "synthetic-private-token")
            .unwrap()
            .is_empty()
    );
    let sid = SessionId::parse("chatgpt:redact").unwrap();
    let shown = |archive: &Archive| {
        let pool = ReadPool::open(archive).unwrap();
        text(&pool.reader().unwrap().messages(&sid, false).unwrap())
    };
    assert_eq!(shown(&archive), ["safe current version"]);
    drop(engine);
    drop(_lock);
    let _engine = rebuild(config, &archive);
    assert_eq!(shown(&archive), ["safe current version"]);
}
