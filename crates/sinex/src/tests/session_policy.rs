use super::*;

pub(super) async fn input_removal(sinex: &mut Sinex, store: &Path, root: &Path) {
    let home = root.join("session-policy-home");
    let project = home.join(".claude/projects/-synthetic");
    fs::create_dir_all(&project).unwrap();
    let line = |id, session, text| {
        serde_json::json!({
            "type":"user","uuid":id,"sessionId":session,
            "message":{"role":"user","content":text}
        })
        .to_string()
            + "\n"
    };
    fs::write(
        project.join("original.jsonl"),
        line("original", "removed:opaque", "original synthetic body"),
    )
    .unwrap();
    let source = root.join("session-policy-source");
    converge(&source, &home, vec![]);
    let archive = Archive::at(&source);
    let known = ReadPool::open(&archive)
        .unwrap()
        .reader()
        .unwrap()
        .known_inputs("proof-machine")
        .unwrap();
    assert_eq!(known.len(), 1);
    let writer = polylogue_store::Writer::start(&archive).unwrap();
    let done = writer
        .excise(
            polylogue_store::excise::ExciseRequest {
                id: "remove-input".into(),
                scope: polylogue_store::excise::Scope::Input(known[0].input_id.clone()),
                reason: None,
                dry_run: false,
            },
            std::sync::Arc::new(hashed_blobs::Store::open(source.clone(), vec![], 0o600).unwrap()),
        )
        .unwrap();
    assert_eq!(done.sessions, ["claude-code:removed:opaque"]);
    drop(writer);
    let id = crate::ArchiveRows::open(&source)
        .unwrap()
        .archive_id()
        .unwrap();
    let units = sinex.publish(&source).await.unwrap().units;
    assert!(units > 0);
    let target = root.join("session-policy-restored");
    let (restored, lock) = Archive::open_or_create(&target).unwrap();
    let deadline = Instant::now();
    loop {
        if sinex.restore(&restored, &id).await.unwrap().units == units {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let db = polylogue_store::archive::open_read(&restored.archive_db()).unwrap();
    let exclusions = polylogue_store::excise::Refused::load(&db).unwrap();
    assert!(!exclusions.is_empty());
    drop(db);
    drop(lock);
    fs::write(
        project.join("different.jsonl"),
        line("new-message", "removed:opaque", "new synthetic body")
            + &line("independent", "unrelated", "admitted independent body"),
    )
    .unwrap();
    converge(&target, &home, vec![store.to_path_buf()]);
    let reads = ReadPool::open(&restored).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .messages(
                &polylogue_model::SessionId::parse("claude-code:removed:opaque").unwrap(),
                false
            )
            .unwrap()
            .is_empty()
    );
    let messages = reads
        .reader()
        .unwrap()
        .messages(
            &polylogue_model::SessionId::parse("claude-code:unrelated").unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "admitted independent body");
}
