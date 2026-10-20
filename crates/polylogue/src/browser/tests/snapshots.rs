use super::*;
use polylogue_engine::Engine;
use polylogue_model::{MessageId, SessionId};
use polylogue_store::{Archive, ReadPool};

fn text(messages: &[polylogue_model::Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| m.blocks.iter().filter_map(|b| b.text()).collect::<String>())
        .collect()
}

#[test]
fn rapid_native_acknowledgements_keep_history_latest_and_retry_identity() {
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
    let mid = MessageId::parse("chatgpt:t1").unwrap();
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
        let versions = reader.message_versions(&mid).unwrap();
        assert_eq!(versions.len(), 2);
        assert_eq!(
            versions.iter().map(|v| v.current).collect::<Vec<_>>(),
            [false, true]
        );
        assert_eq!(
            text(
                &versions
                    .iter()
                    .map(|v| v.message.clone())
                    .collect::<Vec<_>>()
            ),
            ["first retained version", "second current version"]
        );
        for v in versions {
            assert_eq!(
                &before[v.start as usize..v.end as usize],
                if v.current { &second } else { &first }
            );
        }
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
fn legacy_conversion_keeps_all_chain_history_after_index_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_at(dir.path());
    std::fs::create_dir_all(&config.home).unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let ctx = Arc::new(Ctx::at(&config, &archive).unwrap());
    let first = capture("rewrite", "before conversion");
    let ack = exchange(&ctx, "POST", "/v1/browser-captures", &first);
    let path = ctx.spool.join(ack.json()["artifact_ref"].as_str().unwrap());
    // Existing installations have a single unframed envelope at this path.
    std::fs::write(&path, &first).unwrap();
    let mut engine = Engine::new(config.clone(), archive.clone()).unwrap();
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(
        exchange(
            &ctx,
            "POST",
            "/v1/browser-captures",
            &capture("rewrite", "after conversion")
        )
        .status,
        200
    );
    assert!(engine.converge().unwrap().errors.is_empty());
    let mid = MessageId::parse("chatgpt:t1").unwrap();
    let before = {
        let pool = ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        reader.message_versions(&mid).unwrap()
    };
    assert_eq!(before.len(), 3);
    assert_eq!(before.iter().filter(|v| v.current).count(), 1);
    drop(engine);
    drop(_lock);
    let _engine = rebuild(config, &archive);
    let pool = ReadPool::open(&archive).unwrap();
    assert_eq!(
        pool.reader().unwrap().message_versions(&mid).unwrap(),
        before
    );
    assert_eq!(
        text(
            &pool
                .reader()
                .unwrap()
                .messages(&SessionId::parse("chatgpt:rewrite").unwrap(), false)
                .unwrap()
        ),
        ["after conversion"]
    );
}

#[test]
fn text_excision_rehashes_retained_frames_and_cannot_restore_historical_plaintext() {
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
    let mid = MessageId::parse("chatgpt:t1").unwrap();
    let before = {
        let pool = ReadPool::open(&archive).unwrap();
        pool.reader().unwrap().message_versions(&mid).unwrap()
    };
    assert_eq!(before.len(), 2);
    assert!(before.iter().all(|v|!text(std::slice::from_ref(&v.message))[0].contains("synthetic-private-token")));
    assert_eq!(text(&[before[1].message.clone()]), ["safe current version"]);
    drop(engine);
    drop(_lock);
    let _engine = rebuild(config, &archive);
    let pool = ReadPool::open(&archive).unwrap();
    assert_eq!(
        pool.reader().unwrap().message_versions(&mid).unwrap(),
        before
    );
}
